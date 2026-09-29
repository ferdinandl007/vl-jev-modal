"""Inspect mixed visual pilot coverage and materialization on Modal."""

import json
import modal

app = modal.App("vl-jev-visual-pilot-inspection")
image = modal.Image.debian_slim(python_version="3.11")
data = modal.Volume.from_name("vl-jev-pilot-data")
media = modal.Volume.from_name("vl-jev-pilot-media")


@app.function(image=image, volumes={"/dataset": data, "/pilot": media}, timeout=300)
def inspect():
    from collections import Counter, defaultdict
    from pathlib import Path

    result = defaultdict(lambda: {"splits": Counter(), "media_kinds": Counter(),
                                  "question_types": Counter(), "label_quality": Counter(),
                                  "usage": Counter(), "materialized": Counter(),
                                  "gold": Counter()})
    for split in ("train", "dev", "calibration", "test"):
        path = Path("/dataset/pilot_v1") / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["modality"] not in {"video", "image", "image_sequence"}:
                continue
            item = result[row["task_family"]]
            item["splits"][split] += 1
            item["media_kinds"][row["media"]["kind"]] += 1
            item["question_types"][row["question"]["type"]] += 1
            item["label_quality"][row["label_quality"]] += 1
            item["usage"][row["usage"]] += 1
            item["gold"][row["gold"]["key"]] += 1
            folder = Path("/pilot") / split / row["id"]
            if (folder / "row.json").is_file():
                item["materialized"][split] += 1
    return {name: {key: dict(value) for key, value in details.items()}
            for name, details in sorted(result.items())}


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2))
