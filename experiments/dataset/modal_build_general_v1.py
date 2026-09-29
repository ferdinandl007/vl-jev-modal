"""Build a broad, source-pinned Jev decision dataset entirely on Modal.

The two large sources retain their official train/validation/test splits.
Images and metadata are stored in private Modal Volumes, never locally.
"""

import json
import modal

app = modal.App("vl-jev-general-v1-builder")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("requests==2.32.5", "pillow==12.1.1"))
data = modal.Volume.from_name("vl-jev-general-v1-data", create_if_missing=True)
media = modal.Volume.from_name("vl-jev-general-v1-media", create_if_missing=True)
pilot_data = modal.Volume.from_name("vl-jev-pilot-data")
pilot_media = modal.Volume.from_name("vl-jev-pilot-media")

SOURCES = {
    "aokvqa": ("HuggingFaceM4/A-OKVQA", "d1b0efa3a436e9101dfbde3752db7607da696c35"),
    "scienceqa": ("Gisiyuan/ScienceQA", "9e9e9fa2c80909c6ec9fafe9c09bc0775e1e188f"),
}
SPLITS = {"train": "train", "validation": "dev", "test": "test"}


@app.function(image=image, volumes={"/general-data": data,
                                    "/general-media": media}, timeout=3600, memory=4096)
def build_part(source_key, original_split):
    import hashlib
    import io
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import requests
    from PIL import Image
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    repo, revision = SOURCES[source_key]
    if original_split not in SPLITS:
        raise ValueError(original_split)
    output_split = SPLITS[original_split]
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=10, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True)))
    current = session.get(f"https://huggingface.co/api/datasets/{repo}", timeout=60)
    current.raise_for_status()
    if current.json()["sha"] != revision:
        raise ValueError(f"Source revision changed for {repo}")

    part = Path("/general-data/v1/parts") / f"{source_key}-{output_split}.jsonl"
    part.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    page_size = 100

    def convert(offset, item):
        choices = item.get("choices")
        answer = item.get("correct_choice_idx") if source_key == "aokvqa" else item.get("answer")
        if answer is None:
            return None, "unlabeled"
        if (not isinstance(choices, list) or not 2 <= len(choices) <= 10 or
                not all(isinstance(x, str) and x.strip() for x in choices) or
                len({x.strip().casefold() for x in choices}) != len(choices) or
                not isinstance(answer, int) or not 0 <= answer < len(choices)):
            return None, "invalid_choices_or_answer"
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            return None, "invalid_question"
        identifier = hashlib.sha256(f"{repo}:{original_split}:{offset}".encode()).hexdigest()[:24]
        labels = "ABCDEFGHIJ"[:len(choices)]
        source_image = item.get("image")
        image_url = source_image.get("src") if isinstance(source_image, dict) else None
        media_info = None
        modality = "text"
        if image_url:
            destination = Path("/general-media") / source_key / original_split / f"{identifier}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                for attempt in range(5):
                    try:
                        response = requests.get(image_url, timeout=120)
                        response.raise_for_status()
                        break
                    except requests.RequestException:
                        if attempt == 4:
                            raise
                        time.sleep(2 ** attempt)
                if not 0 < len(response.content) <= 20_000_000:
                    raise ValueError(f"Invalid image byte count at {source_key}:{original_split}:{offset}")
                with Image.open(io.BytesIO(response.content)) as decoded:
                    rgb = decoded.convert("RGB")
                    if not 32 <= rgb.width <= 8192 or not 32 <= rgb.height <= 8192:
                        raise ValueError(f"Invalid image dimensions at {source_key}:{original_split}:{offset}")
                    rgb.save(destination, format="JPEG", quality=92)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            media_info = {"kind": "modal_image", "path": str(destination), "sha256": digest}
            modality = "image"
        state = str(item.get("hint") or "").strip() if source_key == "scienceqa" else ""
        row = {"id": identifier, "split": output_split, "modality": modality,
               "task_family": "visual_commonsense" if source_key == "aokvqa"
                              else "science_reasoning",
               "source": {"repo": repo, "revision": revision, "row": offset,
                          "group": (f"{original_split}:{offset}" if source_key == "scienceqa"
                                    else str(item.get("question_id") or
                                             hashlib.sha256(question.encode()).hexdigest())),
                          "original_split": original_split},
               "media": media_info, "state": state,
               "question": {"type": "choice", "instructions": question.strip(),
                            "criteria": {key: text.strip() for key, text in zip(labels, choices)}},
               "gold": {"key": labels[answer]},
               "label_quality": "source_annotation", "usage": "research_only",
               "extra": {"image_sha256": media_info["sha256"] if media_info else None,
                         "source_family": source_key}}
        return row, None

    with part.open("w") as output:
        offset = 0
        total = None
        while total is None or offset < total:
            response = session.get("https://datasets-server.huggingface.co/rows",
                                   params={"dataset": repo, "config": "default",
                                           "split": original_split, "offset": offset,
                                           "length": page_size}, timeout=120)
            response.raise_for_status()
            payload = response.json()
            if total is None:
                total = payload["num_rows_total"]
            rows = payload["rows"]
            if not rows or rows[0]["row_idx"] != offset:
                raise ValueError(f"Viewer pagination failed at {repo}:{original_split}:{offset}")
            with ThreadPoolExecutor(max_workers=12) as pool:
                converted = list(pool.map(lambda entry: convert(entry["row_idx"], entry["row"]), rows))
            for row, skipped in converted:
                if skipped:
                    counts[skipped] += 1
                else:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[f"{row['modality']}_rows"] += 1
            offset += len(rows)
            counts["source_rows_seen"] = offset
            if offset % 500 == 0 or offset == total:
                output.flush()
                data.commit()
                media.commit()
                print(json.dumps({"source": source_key, "split": original_split,
                                  "seen": offset, "total": total,
                                  "kept": counts["image_rows"] + counts["text_rows"]}), flush=True)
            time.sleep(0.25)
    return {"repo": repo, "revision": revision, "original_split": original_split,
            "output_split": output_split, "part": str(part), "counts": dict(counts)}


@app.function(image=image, volumes={"/general-data": data}, timeout=1800)
def repair_science_groups():
    """Replace ScienceQA question-text groups with official split/row identities.

    ScienceQA exposes no problem ID. Repeated templates can represent distinct
    questions; exact prompt/option/media and image hashes remain overlap gates.
    """
    import hashlib
    from pathlib import Path

    root = Path("/general-data/v1/parts")
    changed = {}
    repo, revision = SOURCES["scienceqa"]
    for original_split, split in SPLITS.items():
        path = root / f"scienceqa-{split}.jsonl"
        temporary = path.with_suffix(".jsonl.repaired")
        count = 0
        with path.open() as source, temporary.open("w") as output:
            for line in source:
                row = json.loads(line)
                origin = row["source"]
                offset = origin["row"]
                expected_id = hashlib.sha256(
                    f"{repo}:{original_split}:{offset}".encode()).hexdigest()[:24]
                if (origin["repo"] != repo or origin["revision"] != revision or
                        origin["original_split"] != original_split or row["id"] != expected_id):
                    raise ValueError(f"Unexpected ScienceQA row in {path}: {row['id']}")
                origin["group"] = f"{original_split}:{offset}"
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temporary.replace(path)
        changed[split] = count
    data.commit()
    return {"source": repo, "revision": revision, "rows_repaired": changed,
            "group_policy": "official_split:row_index; exact decision and image hash overlap gates remain"}


@app.function(image=image, volumes={"/general-data": data,
                                    "/general-media": media,
                                    "/pilot-data": pilot_data,
                                    "/pilot-media": pilot_media}, timeout=1800)
def finalize():
    import hashlib
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    root = Path("/general-data/v1")
    parts = root / "parts"
    selected = defaultdict(list)
    skipped = Counter()
    for source_key in SOURCES:
        for original_split, split in SPLITS.items():
            if source_key == "aokvqa" and original_split == "test":
                # Official A-OKVQA test labels are withheld; its viewer rows
                # cannot contribute scored examples.
                continue
            path = parts / f"{source_key}-{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            selected[split].extend(json.loads(line) for line in path.read_text().splitlines())
    for split in ("train", "dev", "test"):
        path = parts / f"webintosh-{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        selected[split].extend(json.loads(line) for line in path.read_text().splitlines())
    for split in ("train", "dev", "calibration", "test"):
        path = Path("/pilot-data/pilot_v1") / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["modality"] in {"image", "image_sequence", "video"}:
                folder = Path("/pilot-media") / split / row["id"]
                record = folder / "row.json"
                if not record.is_file():
                    skipped[f"pilot_missing_media_{split}"] += 1
                    continue
                materialized = json.loads(record.read_text())
                if materialized["id"] != row["id"] or materialized["source"] != row["source"]:
                    raise ValueError(f"Pilot materialization mismatch: {row['id']}")
                if row["modality"] == "video" and not (folder / "media.mp4").is_file():
                    skipped[f"pilot_missing_video_{split}"] += 1
                    continue
                if row["modality"] in {"image", "image_sequence"} and not (
                        folder / "image_0.jpg").is_file():
                    skipped[f"pilot_missing_image_{split}"] += 1
                    continue
            selected[split].append(row)
    images = [row for rows in selected.values() for row in rows
              if (row.get("media") or {}).get("kind") == "modal_image"]

    def verify_image(row):
        asset = row["media"]
        path = Path(asset["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['id']}")

    with ThreadPoolExecutor(max_workers=24) as pool:
        for index, _ in enumerate(pool.map(verify_image, images), 1):
            if index % 5000 == 0:
                print(json.dumps({"verified_images": index, "total_images": len(images)}),
                      flush=True)

    def keys(row):
        source = row["source"]
        result = {("group", source["repo"], str(source["group"]))}
        asset = row.get("media") or {}
        digest = asset.get("sha256")
        if digest:
            result.add(("image", digest))
        fingerprint = hashlib.sha256(json.dumps(
            {"state": row["state"], "question": row["question"],
             "media_sha256": digest}, sort_keys=True).encode()).hexdigest()
        result.add(("exact_decision", fingerprint))
        return result

    heldout = set().union(*(keys(row) for split in ("dev", "calibration", "test")
                            for row in selected[split]))
    clean_train = []
    for row in selected["train"]:
        collisions = keys(row) & heldout
        if collisions:
            for key in collisions:
                skipped[f"train_holdout_overlap_{key[0]}"] += 1
        else:
            clean_train.append(row)
    selected["train"] = clean_train
    ids = set()
    for split, rows in selected.items():
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Duplicate id: {row['id']}")
            ids.add(row["id"])
            if row["split"] != split:
                raise ValueError(f"Split mismatch: {row['id']}")
    root.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    for split in ("train", "dev", "calibration", "test"):
        output = root / f"{split}.jsonl"
        output.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected[split]))
        split_hashes[split] = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {"counts": {split: len(selected[split]) for split in
                          ("train", "dev", "calibration", "test")},
               "by_family": {split: dict(Counter(row["task_family"] for row in selected[split]))
                             for split in ("train", "dev", "calibration", "test")},
               "by_modality": {split: dict(Counter(row["modality"] for row in selected[split]))
                               for split in ("train", "dev", "calibration", "test")},
               "choice_gold_position": {split: dict(Counter(
                   list(row["question"]["criteria"]).index(row["gold"]["key"])
                   for row in selected[split] if row["question"]["type"] == "choice"))
                   for split in ("train", "dev", "calibration", "test")},
               "skipped": dict(skipped), "source_revisions": SOURCES,
               "webintosh_revision": "7709afcb895aa7a3e39e2aa10406b930ad3956f8",
               "scienceqa_group_policy": "official_split:row_index; exact decision and image hash overlap gates",
               "split_sha256": split_hashes,
               "usage": "research_only"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    data.commit()
    return summary


@app.function(image=image, volumes={"/general-data": data,
                                    "/general-media": media}, timeout=300)
def status():
    from pathlib import Path

    root = Path("/general-data/v1")
    parts = {}
    for path in sorted((root / "parts").glob("*.jsonl")):
        parts[path.name] = sum(1 for _ in path.open())
    summary = root / "summary.json"
    return {"parts": parts,
            "summary": json.loads(summary.read_text()) if summary.is_file() else None}


@app.local_entrypoint()
def main(mode: str = "build", source: str = "aokvqa", split: str = "train"):
    if mode == "build":
        if source not in SOURCES or split not in SPLITS:
            raise ValueError("Unsupported source or split")
        result = build_part.remote(source, split)
    elif mode == "finalize":
        result = finalize.remote()
    elif mode == "repair-science-groups":
        result = repair_science_groups.remote()
    elif mode == "status":
        result = status.remote()
    else:
        raise ValueError("mode must be build or finalize")
    print(json.dumps(result, ensure_ascii=False, indent=2))
