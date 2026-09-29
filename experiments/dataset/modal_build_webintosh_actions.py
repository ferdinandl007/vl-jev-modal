"""Build screenshot-to-click decision rows from Webintosh on Modal."""

import json
import modal

app = modal.App("vl-jev-webintosh-actions")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("requests==2.32.5", "pillow==12.1.1"))
data = modal.Volume.from_name("vl-jev-general-v1-data", create_if_missing=True)
media = modal.Volume.from_name("vl-jev-general-v1-media", create_if_missing=True)
REPO = "Chengheng/Webintosh"
REVISION = "7709afcb895aa7a3e39e2aa10406b930ad3956f8"
SPLITS = {"train": "train", "val": "dev", "test": "test"}


def _bbox(row):
    box = row.get("bbox")
    size = row.get("img_size")
    if (not isinstance(box, list) or len(box) != 4 or
            not isinstance(size, list) or len(size) != 2):
        return None
    x1, y1, x2, y2 = box
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return tuple(float(value) for value in box)


def _overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


@app.function(image=image, volumes={"/general-data": data,
                                    "/general-media": media},
              timeout=7200, memory=8192)
def build(split):
    import hashlib
    import io
    import time
    import requests
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from PIL import Image

    if split not in SPLITS:
        raise ValueError(split)
    source = requests.get(f"https://huggingface.co/api/datasets/{REPO}", timeout=60)
    source.raise_for_status()
    if source.json()["sha"] != REVISION:
        raise ValueError("Webintosh source revision changed")
    response = requests.get(
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{split}.jsonl",
        stream=True, timeout=300)
    response.raise_for_status()
    groups = defaultdict(list)
    for line in response.iter_lines():
        if not line:
            continue
        row = json.loads(line)
        if _bbox(row) is not None:
            groups[row["img_filename"]].append(row)
    response.close()
    part = Path("/general-data/v1/parts") / f"webintosh-{SPLITS[split]}.jsonl"
    part.parent.mkdir(parents=True, exist_ok=True)
    items = []
    counts = Counter({"screenshots_seen": len(groups)})
    for filename, elements in sorted(groups.items()):
        targets = [item for item in elements
                   if item.get("is_action_target") and item.get("style") == "imperative"
                   and item.get("task") and _bbox(item) is not None]
        if not targets:
            continue
        target = sorted(targets, key=lambda item: item["id"])[0]
        target_box = _bbox(target)
        size = target["img_size"]
        candidates = [item for item in elements
                      if item["id"] != target["id"] and _bbox(item) is not None
                      and item.get("role") not in {"img", "span"}
                      and _overlap(target_box, _bbox(item)) < 0.1]
        target_center = ((target_box[0] + target_box[2]) / 2,
                         (target_box[1] + target_box[3]) / 2)
        candidates.sort(key=lambda item: (
            ((item["bbox"][0] + item["bbox"][2]) / 2 - target_center[0]) ** 2
            + ((item["bbox"][1] + item["bbox"][3]) / 2 - target_center[1]) ** 2,
            item["id"]))
        chosen = [target]
        for item in candidates:
            if all(_overlap(_bbox(item), _bbox(existing)) < 0.1 for existing in chosen):
                chosen.append(item)
            if len(chosen) == 4:
                break
        if len(chosen) != 4:
            counts["too_few_distinct_options"] += 1
            continue
        shift = int(hashlib.sha256(target["id"].encode()).hexdigest()[:8], 16) % 4
        chosen = chosen[shift:] + chosen[:shift]
        labels = "ABCD"

        def description(item):
            x1, y1, x2, y2 = _bbox(item)
            x = round(1000 * (x1 + x2) / (2 * size[0]))
            y = round(1000 * (y1 + y2) / (2 * size[1]))
            return f"Click screen position ({x}, {y}) on a 1000 by 1000 coordinate grid."

        if len({description(item) for item in chosen}) != 4:
            counts["duplicate_option_coordinates"] += 1
            continue

        identifier = hashlib.sha256(f"{REPO}:{split}:{target['id']}".encode()).hexdigest()[:24]
        destination = Path("/general-media/webintosh") / split / f"{identifier}.jpg"
        row = {"id": identifier, "split": SPLITS[split], "modality": "image",
               "task_family": "gui_next_click",
               "source": {"repo": REPO, "revision": REVISION,
                          "row": target["id"], "group": target["trajectory_id"],
                          "original_split": split},
               "media": {"kind": "modal_image", "path": str(destination)},
               "state": target["task"],
               "question": {"type": "choice",
                            "instructions": "To continue the task, which visible screen position should be clicked next?",
                            "criteria": {label: description(item)
                                         for label, item in zip(labels, chosen)}},
               "gold": {"key": labels[next(index for index, item in enumerate(chosen)
                                              if item["id"] == target["id"])]},
               "label_quality": "source_action_target", "usage": "research_only",
               "extra": {"target_instruction": target["instruction"],
                         "screenshot": filename,
                         "option_element_ids": [item["id"] for item in chosen],
                         "source_task": "single_step_click_choice",
                         "image_sha256": None}}
        items.append((filename, destination, row))

    def fetch(item):
        filename, destination, row = item
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}"
            for attempt in range(8):
                try:
                    downloaded = requests.get(url, timeout=180)
                    downloaded.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 7:
                        raise
                    time.sleep(min(60, 2 ** attempt))
            if len(downloaded.content) > 30_000_000:
                raise ValueError(f"Screenshot is too large: {filename}")
            with Image.open(io.BytesIO(downloaded.content)) as source_image:
                source_image.convert("RGB").save(destination, "JPEG", quality=92)
        row["extra"]["image_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        row["media"]["sha256"] = row["extra"]["image_sha256"]
        return row

    with part.open("w") as output:
        for start in range(0, len(items), 100):
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(pool.map(fetch, items[start:start + 100]))
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts["kept"] += len(rows)
            output.flush()
            data.commit()
            media.commit()
            print(json.dumps({"split": split, "kept": counts["kept"],
                              "total": len(items)}), flush=True)
    return {"part": str(part), "revision": REVISION, "counts": dict(counts)}


@app.local_entrypoint()
def main(split: str = "train"):
    print(json.dumps(build.remote(split), indent=2))
