"""Build a task-conditioned, labeled-control Mind2Web pilot on Modal only."""

import json
import modal

app = modal.App("vl-jev-mind2web-semantic-pilot")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("huggingface_hub==0.36.0", "pyarrow==20.0.0",
                      "beautifulsoup4==4.13.5", "pillow==12.1.1"))
data = modal.Volume.from_name("vl-jev-gui-semantic-pilot-data", create_if_missing=True)
media = modal.Volume.from_name("vl-jev-gui-semantic-pilot-media", create_if_missing=True)
REPO = "osunlp/Multimodal-Mind2Web"
REVISION = "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889"
SHARD = "data/train-00000-of-00027-4d11798d7219186d.parquet"
SHARD_SHA256 = "544d46778967fbf95b508a742d418dc1b5fbe72dcf99c014b58412d13da694d7"


def _candidate(item, nodes):
    item = json.loads(item) if isinstance(item, str) else item
    attrs = json.loads(item.get("attributes") or "{}")
    element_id = str(item.get("backend_node_id") or attrs.get("backend_node_id") or "")
    if not element_id:
        return None
    node = nodes.get(element_id)
    text = node.get_text(" ", strip=True) if node else ""
    name = next((str(value).strip() for value in
                 (attrs.get("aria_label"), text, attrs.get("placeholder"),
                  attrs.get("title"), attrs.get("alt")) if value and str(value).strip()), "")
    if not name:
        return None
    name = " ".join(name.split())[:100]
    role = str(attrs.get("role") or item.get("tag") or "control").strip().lower()
    box = attrs.get("bounding_box_rect")
    try:
        left, top, width, height = (float(value) for value in box.split(","))
    except (AttributeError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return {"id": element_id, "role": role, "name": name,
            "box": (left, top, width, height)}


def _make_row(source, width, height):
    import hashlib
    from bs4 import BeautifulSoup

    operation = json.loads(source["operation"])
    if operation.get("op") != "CLICK" or operation.get("original_op") != "CLICK":
        return None, "non_click"
    html = BeautifulSoup(source["cleaned_html"], "html.parser")
    nodes = {str(node.get("backend_node_id")): node
             for node in html.find_all(attrs={"backend_node_id": True})}
    positive_raw = [json.loads(item) for item in source.get("pos_candidates") or []]
    original_ids = {str(item.get("backend_node_id")) for item in positive_raw
                    if item.get("is_original_target")}
    positives = [item for item in (_candidate(item, nodes) for item in positive_raw) if item]
    if not positives:
        return None, "no_named_positive"
    positives.sort(key=lambda item: (item["id"] not in original_ids, item["id"]))
    target = positives[0]
    tx, ty = target["box"][0] + target["box"][2] / 2, target["box"][1] + target["box"][3] / 2
    if not 0 <= tx < width or not 0 <= ty < height:
        return None, "target_not_visible"
    negatives = []
    seen = {(target["role"], target["name"])}
    for raw in source.get("neg_candidates") or []:
        item = _candidate(raw, nodes)
        if not item or item["id"] == target["id"]:
            continue
        signature = (item["role"], item["name"])
        if signature in seen:
            continue
        x, y = item["box"][0] + item["box"][2] / 2, item["box"][1] + item["box"][3] / 2
        if not 0 <= x < width or not 0 <= y < height:
            continue
        if item["box"][2] * item["box"][3] > width * height * 0.2:
            continue
        negatives.append(item)
        seen.add(signature)
    negatives.sort(key=lambda item: (
        item["role"] != target["role"],
        (item["box"][0] + item["box"][2] / 2 - tx) ** 2 +
        (item["box"][1] + item["box"][3] / 2 - ty) ** 2,
        item["id"]))
    if len(negatives) < 3:
        return None, "too_few_named_negatives"
    choices = [target, *negatives[:3]]
    shift = int(hashlib.sha256(source["action_uid"].encode()).hexdigest()[:8], 16) % 4
    choices = choices[shift:] + choices[:shift]
    key = "ABCD"[next(index for index, item in enumerate(choices)
                       if item["id"] == target["id"])]
    index = int(source["target_action_index"])
    prior = source["action_reprs"][:index][-5:]
    task_id = source["annotation_id"]
    split = "dev" if int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
    row = {"id": source["action_uid"], "split": split, "modality": "image",
           "task_family": "gui_named_control_next_click",
           "source": {"repo": REPO, "revision": REVISION,
                      "row": source["action_uid"], "group": task_id,
                      "shard": SHARD, "website": source["website"]},
           "state": "Goal: " + source["confirmed_task"] + "\nPrevious actions: " +
                    ("; ".join(prior) if prior else "none"),
           "question": {"type": "choice",
                        "instructions": "Which available control should be clicked next?",
                        "criteria": {label: f"{item['role']}: {item['name']}"
                                     for label, item in zip("ABCD", choices)}},
           "gold": {"key": key}, "label_quality": "human_annotated_action",
           "usage": "research_only",
           "extra": {"option_element_ids": [item["id"] for item in choices],
                     "option_boxes": [item["box"] for item in choices],
                     "operation": operation["original_op"],
                     "history_length": len(prior)}}
    return row, None


@app.function(image=image, timeout=3600, memory=8192,
              volumes={"/gui-data": data, "/gui-media": media})
def build():
    import hashlib
    import io
    from collections import Counter
    from pathlib import Path
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw, ImageFont
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(REPO, SHARD, revision=REVISION, repo_type="dataset")
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if digest != SHARD_SHA256:
        raise ValueError("Pinned source shard changed")
    root = Path("/gui-data/mind2web_semantic_pilot_v1")
    root.mkdir(parents=True, exist_ok=True)
    files = {split: (root / f"{split}.jsonl").open("w") for split in ("train", "dev")}
    counts = Counter()
    groups = {"train": set(), "dev": set()}
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=8):
            for source_row in batch.to_pylist():
                counts["source_rows"] += 1
                shot = source_row.get("screenshot") or {}
                content = shot.get("bytes")
                if not content:
                    counts["missing_screenshot"] += 1
                    continue
                try:
                    screenshot = Image.open(io.BytesIO(content)).convert("RGB")
                except Exception:
                    counts["invalid_screenshot"] += 1
                    continue
                row, reason = _make_row(source_row, *screenshot.size)
                if reason:
                    counts[reason] += 1
                    continue
                destination = Path("/gui-media/mind2web_semantic_pilot_v1") / row["split"] / (row["id"] + ".jpg")
                destination.parent.mkdir(parents=True, exist_ok=True)
                screenshot.save(destination, "JPEG", quality=90)
                marked = screenshot.copy()
                draw = ImageDraw.Draw(marked)
                font = ImageFont.load_default(size=max(18, min(marked.size) // 45))
                for label, box in zip("ABCD", row["extra"]["option_boxes"]):
                    x, y, width, height = box
                    draw.rectangle((x, y, x + width, y + height),
                                   outline="#0066ff", width=max(3, marked.width // 400))
                    text_box = draw.textbbox((0, 0), label, font=font)
                    text_width = text_box[2] - text_box[0] + 12
                    text_height = text_box[3] - text_box[1] + 8
                    text_x, text_y = max(0, int(x)), max(0, int(y) - text_height)
                    draw.rectangle((text_x, text_y, text_x + text_width,
                                    text_y + text_height), fill="#0066ff")
                    draw.text((text_x + 6, text_y + 2), label, fill="white", font=font)
                marked_destination = destination.with_name(destination.stem + "-marked.jpg")
                marked.save(marked_destination, "JPEG", quality=90)
                row["media"] = {"kind": "modal_image", "path": str(destination),
                                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
                row["extra"]["marked_media_path"] = str(marked_destination)
                files[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["split"]] += 1
                groups[row["split"]].add(row["source"]["group"])
            for file in files.values():
                file.flush()
            data.commit()
            media.commit()
            print(json.dumps({"processed": counts["source_rows"],
                              "train": counts["train"], "dev": counts["dev"]}), flush=True)
    finally:
        for file in files.values():
            file.close()
    if groups["train"] & groups["dev"]:
        raise ValueError("Task leakage between pilot splits")
    summary = {"repo": REPO, "revision": REVISION, "shard": SHARD,
               "shard_sha256": digest, "counts": dict(counts),
               "task_groups": {split: len(value) for split, value in groups.items()},
               "split_sha256": {split: hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
                                for split in files}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    data.commit()
    media.commit()
    return summary


@app.local_entrypoint()
def main():
    print(json.dumps(build.remote(), indent=2))
