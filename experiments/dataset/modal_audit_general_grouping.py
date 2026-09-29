"""Inspect source row identities and overlap quarantine on Modal only."""

import json
import modal

app = modal.App("vl-jev-general-group-audit")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")
data = modal.Volume.from_name("vl-jev-general-v1-data")


@app.function(image=image, volumes={"/general-data": data}, timeout=600)
def audit():
    from collections import Counter, defaultdict
    from pathlib import Path
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    root = Path("/general-data/v1/parts")
    parts = {}
    for name in ("scienceqa-train", "scienceqa-dev", "scienceqa-test"):
        rows = [json.loads(line) for line in (root / f"{name}.jsonl").open()]
        parts[name] = rows
    heldout_groups = {row["source"]["group"] for name in ("scienceqa-dev", "scienceqa-test")
                      for row in parts[name]}
    heldout_images = {row["media"]["sha256"] for name in ("scienceqa-dev", "scienceqa-test")
                      for row in parts[name] if row["media"]}
    collision = [row for row in parts["scienceqa-train"]
                 if row["source"]["group"] in heldout_groups]
    group_counts = Counter(row["source"]["group"] for row in parts["scienceqa-train"])
    samples = []
    for row in collision[:12]:
        samples.append({"row": row["source"]["row"], "group": row["source"]["group"],
                        "question": row["question"]["instructions"],
                        "media_sha256": (row["media"] or {}).get("sha256")})
    source_rows = []
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=8, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True)))
    for split, offset in (("train", 0), ("train", 250), ("validation", 0), ("test", 0)):
        response = session.get("https://datasets-server.huggingface.co/rows",
                                params={"dataset": "Gisiyuan/ScienceQA", "config": "default",
                                        "split": split, "offset": offset, "length": 2}, timeout=60)
        if response.status_code != 200:
            raise ValueError(f"ScienceQA viewer status {response.status_code} at {split}:{offset}")
        for entry in response.json()["rows"]:
            item = entry["row"]
            source_rows.append({"split": split, "row_idx": entry["row_idx"],
                                "keys": list(item),
                                "identity_values": {key: item.get(key) for key in
                                                    ("id", "question_id", "image", "task", "grade")
                                                    if key in item and key != "image"},
                                "question": item.get("question")})
    return {"part_counts": {name: len(rows) for name, rows in parts.items()},
            "train_group_collisions": len(collision),
            "train_unique_groups": len(group_counts),
            "train_largest_group_sizes": group_counts.most_common(8),
            "train_image_overlap": sum(bool(row["media"] and row["media"]["sha256"] in heldout_images)
                                       for row in parts["scienceqa-train"]),
            "collision_samples": samples, "viewer_rows": source_rows}


@app.local_entrypoint()
def main():
    print(json.dumps(audit.remote(), ensure_ascii=False, indent=2))
