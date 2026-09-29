"""Audit UCF101 numeric labels against video_path class names on Modal."""

import json
import modal

app = modal.App("vl-jev-ucf101-label-audit")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub==1.32.0")
volume = modal.Volume.from_name("vl-jev-ucf101-full-data", create_if_missing=True)
REPO = "guyuchao/UCF101"
REVISION = "057753e5d0709d3f5b8104a803b91a420a069103"


@app.function(image=image, volumes={"/ucf-data": volume}, timeout=1800, memory=2048)
def audit():
    from collections import Counter, defaultdict
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    result = {}
    for split, filename in (("train", "train03.json"), ("test", "test03.json")):
        path = hf_hub_download(REPO, filename, revision=REVISION,
                               repo_type="dataset", cache_dir="/ucf-data/hf-cache")
        records = json.loads(Path(path).read_text())
        counts = Counter()
        by_numeric = defaultdict(Counter)
        mismatches = []
        missing = 0
        for index, item in enumerate(records):
            video_path = item.get("video_path", "")
            parts = video_path.replace("\\", "/").split("/")
            class_name = parts[-2] if len(parts) >= 2 else None
            numeric = item.get("label")
            if class_name is None or numeric is None:
                missing += 1
                continue
            counts[str(numeric)] += 1
            by_numeric[str(numeric)][class_name] += 1
            if isinstance(numeric, str) and not numeric.isdigit():
                if numeric != class_name:
                    mismatches.append({"row": index, "label": numeric, "path_class": class_name})
        collisions = {key: dict(value) for key, value in by_numeric.items() if len(value) > 1}
        reverse = defaultdict(Counter)
        for numeric, classes in by_numeric.items():
            for class_name, count in classes.items():
                reverse[class_name][numeric] += count
        class_collisions = {key: dict(value) for key, value in reverse.items() if len(value) > 1}
        result[split] = {"rows": len(records), "label_type_counts": dict(Counter(type(item.get("label")).__name__ for item in records)),
                         "missing_label_or_path_class": missing, "numeric_label_class_counts": dict(counts),
                         "numeric_to_path_class": {key: dict(value) for key, value in by_numeric.items()},
                         "numeric_to_class_collisions": collisions, "class_to_numeric_collisions": class_collisions,
                         "string_label_path_mismatches": mismatches[:30], "string_label_mismatch_count": len(mismatches)}
    report = {"schema_version": "ucf101_source_label_audit_v1",
              "repo": REPO, "revision": REVISION, "splits": result}
    output = Path("/ucf-data/ucf101_full_v1/source_label_audit_v1.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(audit.remote(), indent=2, ensure_ascii=False))
