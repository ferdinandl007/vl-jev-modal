"""Build the full multimodal Mind2Web training split for general-head replay.

All source downloads, screenshots, and output files remain on Modal.
Official test splits are never read here. The development set is grouped by task.
"""

import json
import modal

from experiments.dataset.modal_build_mind2web_semantic_pilot import (
    REPO, REVISION, _make_row, image,
)

app = modal.App("vl-jev-mind2web-general-data")
data = modal.Volume.from_name("vl-jev-gui-general-data", create_if_missing=True)
media = modal.Volume.from_name("vl-jev-gui-general-media", create_if_missing=True)
ROOT = "/gui-data/mind2web_general_v1"
MEDIA_ROOT = "/gui-media/mind2web_general_v1"


@app.function(image=image, timeout=7200, memory=8192, max_containers=8,
              volumes={"/gui-data": data, "/gui-media": media})
def build_shard(shard):
    import hashlib
    import io
    from collections import Counter
    from pathlib import Path
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw, ImageFont
    from huggingface_hub import hf_hub_download

    if not shard.startswith("data/train-") or not shard.endswith(".parquet"):
        raise ValueError("Only the official training shards may be used")
    index = int(shard.split("train-")[1].split("-")[0])
    part = Path(ROOT) / "parts" / f"{index:02d}"
    if (part / "summary.json").is_file():
        saved = json.loads((part / "summary.json").read_text())
        if saved["shard"] != shard or saved["revision"] != REVISION:
            raise ValueError("Shard summary mismatch")
        return saved
    part.mkdir(parents=True, exist_ok=True)
    source = hf_hub_download(REPO, shard, revision=REVISION, repo_type="dataset")
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    files = {split: (part / f"{split}.jsonl").open("w") for split in ("train", "dev")}
    counts = Counter()
    groups = {"train": set(), "dev": set()}
    try:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=8):
            for source_row in batch.to_pylist():
                counts["source_rows"] += 1
                content = (source_row.get("screenshot") or {}).get("bytes")
                if not content:
                    counts["missing_screenshot"] += 1
                    continue
                try:
                    opened = Image.open(io.BytesIO(content))
                    if opened.width * opened.height > 20_000_000:
                        counts["oversize_screenshot"] += 1
                        continue
                    screenshot = opened.convert("RGB")
                except Exception:
                    counts["invalid_screenshot"] += 1
                    continue
                row, reason = _make_row(source_row, *screenshot.size, shard=shard,
                                        allowed_operations=("CLICK", "TYPE", "SELECT"))
                if reason:
                    counts[reason] += 1
                    continue
                destination = (Path(MEDIA_ROOT) / row["split"] /
                               f"{row['id']}-marked.jpg")
                destination.parent.mkdir(parents=True, exist_ok=True)
                marked = screenshot.copy()
                draw = ImageDraw.Draw(marked)
                font = ImageFont.load_default(size=max(18, min(marked.size) // 45))
                for label, box in zip("ABCD", row["extra"]["option_boxes"]):
                    x, y, width, height = box
                    draw.rectangle((x, y, x + width, y + height),
                                   outline="#0066ff", width=max(3, marked.width // 400))
                    bounds = draw.textbbox((0, 0), label, font=font)
                    w, h = bounds[2] - bounds[0] + 12, bounds[3] - bounds[1] + 8
                    lx, ly = max(0, int(x)), max(0, int(y) - h)
                    draw.rectangle((lx, ly, lx + w, ly + h), fill="#0066ff")
                    draw.text((lx + 6, ly + 2), label, fill="white", font=font)
                marked.save(destination, "JPEG", quality=90)
                row["media"] = {"kind": "modal_image", "path": str(destination),
                                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
                row["extra"].pop("option_boxes", None)
                files[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["split"]] += 1
                counts[f"operation:{row['extra']['operation']}"] += 1
                groups[row["split"]].add(row["source"]["group"])
            for handle in files.values():
                handle.flush()
            data.commit()
            media.commit()
        if groups["train"] & groups["dev"]:
            raise ValueError("Task leakage between shard splits")
    finally:
        for handle in files.values():
            handle.close()
    result = {"repo": REPO, "revision": REVISION, "shard": shard,
              "shard_sha256": digest, "counts": dict(counts),
              "task_groups": {key: len(value) for key, value in groups.items()},
              "split_sha256": {split: hashlib.sha256((part / f"{split}.jsonl").read_bytes()).hexdigest()
                               for split in ("train", "dev")}}
    (part / "summary.json").write_text(json.dumps(result, indent=2))
    data.commit()
    media.commit()
    return result


@app.function(image=image, timeout=86400, volumes={"/gui-data": data})
def build_all():
    import hashlib
    from collections import Counter
    from pathlib import Path
    from huggingface_hub import HfApi

    paths = sorted(path for path in HfApi().list_repo_files(
        REPO, repo_type="dataset", revision=REVISION)
        if path.startswith("data/train-") and path.endswith(".parquet"))
    if len(paths) != 27:
        raise ValueError(f"Expected 27 official train shards, found {len(paths)}")
    results = list(build_shard.map(paths))
    data.reload()
    root = Path(ROOT)
    counts = Counter()
    ids = set()
    groups = {"train": set(), "dev": set()}
    for split in ("train", "dev"):
        destination = root / f"{split}.jsonl"
        with destination.open("w") as output:
            for index, result in enumerate(results):
                part = root / "parts" / f"{index:02d}"
                path = part / f"{split}.jsonl"
                if hashlib.sha256(path.read_bytes()).hexdigest() != result["split_sha256"][split]:
                    raise ValueError(f"Part changed: {path}")
                for line in path.open():
                    row = json.loads(line)
                    if row["id"] in ids:
                        raise ValueError(f"Duplicate action ID: {row['id']}")
                    ids.add(row["id"])
                    groups[split].add(row["source"]["group"])
                    counts[split] += 1
                    counts[f"operation:{row['extra']['operation']}"] += 1
                    output.write(line)
    if groups["train"] & groups["dev"]:
        raise ValueError("Task leakage across shards")
    summary = {"repo": REPO, "revision": REVISION, "shards": results,
               "counts": dict(counts), "task_groups": {key: len(value)
                                                     for key, value in groups.items()},
               "split_sha256": {split: hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
                                for split in ("train", "dev")},
               "usage": "research_only", "official_test_splits_read": False}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    data.commit()
    return {key: summary[key] for key in ("repo", "revision", "counts", "task_groups",
                                         "split_sha256", "official_test_splits_read")}


@app.local_entrypoint()
def main():
    print(json.dumps(build_all.remote(), indent=2))
