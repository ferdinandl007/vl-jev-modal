"""Build an untouched, derived four-choice Mind2Web test-task set on Modal."""

import json
import modal

from experiments.dataset.modal_build_mind2web_semantic_pilot import (
    REPO, REVISION, _make_row, image,
)

app = modal.App("vl-jev-mind2web-test-task")
data = modal.Volume.from_name("vl-jev-gui-test-task-data", create_if_missing=True)
media = modal.Volume.from_name("vl-jev-gui-test-task-media", create_if_missing=True)
ROOT = "/gui-test-data/mind2web_test_task_v1"
MEDIA_ROOT = "/gui-test-media/mind2web_test_task_v1"


@app.function(image=image, timeout=7200, memory=8192, max_containers=5,
              volumes={"/gui-test-data": data, "/gui-test-media": media})
def build_shard(shard):
    import hashlib
    import io
    from collections import Counter
    from pathlib import Path
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw, ImageFont
    from huggingface_hub import hf_hub_download

    if not shard.startswith("data/test_task-") or not shard.endswith(".parquet"):
        raise ValueError("Only official test_task shards may be used")
    index = int(shard.split("test_task-")[1].split("-")[0])
    root = Path(ROOT) / "parts" / f"{index:02d}"
    if (root / "summary.json").is_file():
        result = json.loads((root / "summary.json").read_text())
        if result["shard"] != shard or result["revision"] != REVISION:
            raise ValueError("Test shard summary mismatch")
        return result
    root.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REVISION)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    output = (root / "rows.jsonl").open("w")
    counts = Counter()
    groups = set()
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for source in batch.to_pylist():
                counts["source_rows"] += 1
                content = (source.get("screenshot") or {}).get("bytes")
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
                row, reason = _make_row(source, *screenshot.size, shard=shard,
                                        allowed_operations=("CLICK", "TYPE", "SELECT"))
                if reason:
                    counts[reason] += 1
                    continue
                row["split"] = "test_task"
                destination = Path(MEDIA_ROOT) / f"{row['id']}-marked.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                marked = screenshot.copy()
                draw = ImageDraw.Draw(marked)
                font = ImageFont.load_default(size=max(18, min(marked.size) // 45))
                for label, box in zip("ABCD", row["extra"]["option_boxes"]):
                    x, y, width, height = box
                    draw.rectangle((x, y, x + width, y + height), outline="#0066ff",
                                   width=max(3, marked.width // 400))
                    bounds = draw.textbbox((0, 0), label, font=font)
                    w, h = bounds[2] - bounds[0] + 12, bounds[3] - bounds[1] + 8
                    lx, ly = max(0, int(x)), max(0, int(y) - h)
                    draw.rectangle((lx, ly, lx + w, ly + h), fill="#0066ff")
                    draw.text((lx + 6, ly + 2), label, fill="white", font=font)
                marked.save(destination, "JPEG", quality=90)
                row["media"] = {"kind": "modal_image", "path": str(destination),
                                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
                row["extra"].pop("option_boxes", None)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["kept"] += 1
                counts[f"operation:{row['extra']['operation']}"] += 1
                groups.add(row["source"]["group"])
            output.flush()
            data.commit()
            media.commit()
    finally:
        output.close()
    result = {"repo": REPO, "revision": REVISION, "shard": shard,
              "shard_sha256": digest, "counts": dict(counts),
              "task_groups": len(groups),
              "rows_sha256": hashlib.sha256((root / "rows.jsonl").read_bytes()).hexdigest()}
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    data.commit()
    media.commit()
    return result


@app.function(image=image, timeout=86400, volumes={"/gui-test-data": data})
def build_all():
    import hashlib
    from collections import Counter
    from pathlib import Path
    from huggingface_hub import HfApi

    paths = sorted(path for path in HfApi().list_repo_files(
        REPO, repo_type="dataset", revision=REVISION)
        if path.startswith("data/test_task-") and path.endswith(".parquet"))
    if not paths:
        raise ValueError("No official test_task shards")
    results = list(build_shard.map(paths))
    data.reload()
    root = Path(ROOT)
    ids = set()
    groups = set()
    counts = Counter()
    with (root / "test_task.jsonl").open("w") as output:
        for index, result in enumerate(results):
            path = root / "parts" / f"{index:02d}" / "rows.jsonl"
            if hashlib.sha256(path.read_bytes()).hexdigest() != result["rows_sha256"]:
                raise ValueError(f"Test part changed: {path}")
            for line in path.open():
                row = json.loads(line)
                if row["id"] in ids:
                    raise ValueError("Duplicate test action ID")
                ids.add(row["id"])
                groups.add(row["source"]["group"])
                counts["kept"] += 1
                counts[f"operation:{row['extra']['operation']}"] += 1
                output.write(line)
    summary = {"repo": REPO, "revision": REVISION,
               "source_split": "test_task", "counts": dict(counts),
               "task_groups": len(groups), "shards": results,
               "split_sha256": hashlib.sha256((root / "test_task.jsonl").read_bytes()).hexdigest(),
               "training_access": False}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    data.commit()
    return {key: summary[key] for key in ("source_split", "counts", "task_groups",
                                         "split_sha256", "training_access")}


@app.local_entrypoint()
def main():
    print(json.dumps(build_all.remote(), indent=2))
