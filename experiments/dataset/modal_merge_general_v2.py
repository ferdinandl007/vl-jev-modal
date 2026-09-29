"""Freeze verified broad v1 + full UCF101 + labeled human actions on Modal."""

import json
import modal

app = modal.App("vl-jev-general-v2-merger")
image = modal.Image.debian_slim(python_version="3.11")
general_data = modal.Volume.from_name("vl-jev-general-v1-data")
general_media = modal.Volume.from_name("vl-jev-general-v1-media")
ucf_data = modal.Volume.from_name("vl-jev-ucf101-full-data")
ucf_media = modal.Volume.from_name("vl-jev-ucf101-full-media")
SPLITS = ("train", "dev", "calibration", "test")


@app.function(image=image, volumes={"/general-data": general_data,
                                    "/general-media": general_media,
                                    "/ucf-data": ucf_data,
                                    "/ucf-media": ucf_media},
              timeout=86400, memory=8192)
def finalize():
    import hashlib
    from collections import Counter, defaultdict
    from pathlib import Path

    v1 = Path("/general-data/v1")
    human = v1 / "human_actions_v1"
    ucf = Path("/ucf-data/ucf101_full_v1")
    for root in (v1, human, ucf):
        if not (root / "summary.json").is_file():
            raise FileNotFoundError(f"Missing completed manifest: {root}")
    v1_summary = json.loads((v1 / "summary.json").read_text())
    human_summary = json.loads((human / "summary.json").read_text())
    ucf_summary = json.loads((ucf / "summary.json").read_text())
    if human_summary["source_rows"]["genuine_labeled_train"] != 12600:
        raise ValueError("Human-action source train not complete")
    if human_summary["source_rows"]["excluded_placeholder_test"] != 5400:
        raise ValueError("Human-action placeholder test audit missing")
    selected = defaultdict(list)
    skipped = Counter()
    for split in SPLITS:
        v1_file = v1 / f"{split}.jsonl"
        if hashlib.sha256(v1_file.read_bytes()).hexdigest() != v1_summary["split_sha256"][split]:
            raise ValueError(f"v1 split changed: {split}")
        for line in v1_file.open():
            row = json.loads(line)
            if row["source"]["repo"] == "guyuchao/UCF101":
                skipped["replace_pilot_ucf101"] += 1
                continue
            selected[split].append(row)
        human_file = human / f"{split}.jsonl"
        if hashlib.sha256(human_file.read_bytes()).hexdigest() != human_summary["split_sha256"][split]:
            raise ValueError(f"Human-action split changed: {split}")
        for index, line in enumerate(human_file.open(), start=1):
            row = json.loads(line)
            asset = row["media"]
            path = Path(asset["path"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Human-action image missing/changed: {row['id']}")
            selected[split].append(row)
            if index % 1000 == 0:
                print(json.dumps({"phase": "verified_human_images", "split": split,
                                  "count": index}), flush=True)
        ucf_file = ucf / f"{split}.jsonl"
        for index, line in enumerate(ucf_file.open(), start=1):
            row = json.loads(line)
            folder = Path("/ucf-media") / split / row["id"]
            record_path = folder / "row.json"
            target = folder / "media.avi"
            if not record_path.is_file() or not target.is_file():
                skipped[f"ucf_missing_media_{split}"] += 1
                continue
            record = json.loads(record_path.read_text())
            if record["id"] != row["id"] or record["source"] != row["source"]:
                raise ValueError(f"UCF materialization mismatch: {row['id']}")
            files = record["materialized_media"]
            if len(files) != 1 or files[0]["path"] != "media.avi":
                raise ValueError(f"UCF invalid materialization record: {row['id']}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != files[0]["sha256"]:
                raise ValueError(f"UCF media hash mismatch: {row['id']}")
            row["media"] = {"kind": "modal_video", "path": str(target),
                            "sha256": files[0]["sha256"],
                            "duration_seconds": files[0]["duration_seconds"]}
            selected[split].append(row)
            if index % 500 == 0:
                print(json.dumps({"phase": "verified_ucf_videos", "split": split,
                                  "count": index}), flush=True)
    if any(skipped[f"ucf_missing_media_{split}"] for split in SPLITS):
        raise ValueError(f"Full UCF101 media is incomplete: {dict(skipped)}")

    # Assign each source-video group, exact image/video, and exact decision to
    # its most protected split. This prevents train and evaluation leakage.
    def keys(row):
        source = row["source"]
        asset = row.get("media") or {}
        digest = asset.get("sha256")
        result = {("group", source["repo"], str(source["group"]))}
        if digest:
            result.add(("media_sha256", digest))
        exact = hashlib.sha256(json.dumps(
            {"state": row["state"], "question": row["question"],
             "media_sha256": digest}, sort_keys=True).encode()).hexdigest()
        result.add(("decision", exact))
        return result

    priority = {"train": 0, "dev": 1, "calibration": 2, "test": 3}
    max_priority = {}
    for split in SPLITS:
        for row in selected[split]:
            for key in keys(row):
                max_priority[key] = max(priority[split], max_priority.get(key, -1))
    clean = {}
    for split in SPLITS:
        clean[split] = []
        for row in selected[split]:
            highest = max(max_priority[key] for key in keys(row))
            if highest > priority[split]:
                skipped[f"{split}_cross_split_overlap"] += 1
            else:
                clean[split].append(row)
    seen_ids = set()
    for split in SPLITS:
        for row in clean[split]:
            if row["split"] != split or row["id"] in seen_ids:
                raise ValueError(f"Invalid or duplicate ID in {split}: {row['id']}")
            seen_ids.add(row["id"])
    root = Path("/general-data/v2")
    root.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                for row in clean[split]))
        split_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "counts": {split: len(clean[split]) for split in SPLITS},
        "by_family": {split: dict(Counter(row["task_family"] for row in clean[split]))
                      for split in SPLITS},
        "by_modality": {split: dict(Counter(row["modality"] for row in clean[split]))
                        for split in SPLITS},
        "choice_gold_position": {split: dict(Counter(
            list(row["question"]["criteria"]).index(row["gold"]["key"])
            for row in clean[split] if row["question"]["type"] == "choice"))
            for split in SPLITS},
        "skipped": dict(skipped), "split_sha256": split_hashes,
        "inherited_v1_split_sha256": v1_summary["split_sha256"],
        "human_action_source_revision": human_summary["revision"],
        "ucf101_source_revision": ucf_summary["revision"],
        "usage": "research_only", "status": "complete"
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    general_data.commit()
    return summary


@app.function(image=image, volumes={"/general-data": general_data,
                                    "/ucf-data": ucf_data,
                                    "/ucf-media": ucf_media}, timeout=300)
def status():
    from pathlib import Path
    root = Path("/general-data/v2")
    human = Path("/general-data/v1/human_actions_v1/summary.json")
    ucf = Path("/ucf-data/ucf101_full_v1/summary.json")
    return {"v2": json.loads((root / "summary.json").read_text())
            if (root / "summary.json").is_file() else None,
            "human_actions": json.loads(human.read_text())["counts"] if human.is_file() else None,
            "ucf101": json.loads(ucf.read_text())["split_counts"] if ucf.is_file() else None}


@app.local_entrypoint()
def main(mode: str = "status"):
    result = finalize.remote() if mode == "finalize" else status.remote() if mode == "status" else None
    if result is None:
        raise ValueError("mode must be status or finalize")
    print(json.dumps(result, indent=2, ensure_ascii=False))
