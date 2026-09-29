"""Count usable Webintosh screenshot-to-action rows on Modal."""

import json
import modal

app = modal.App("vl-jev-webintosh-action-audit")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")
REVISION = "7709afcb895aa7a3e39e2aa10406b930ad3956f8"


@app.function(image=image, timeout=1800, memory=4096)
def audit(split):
    import hashlib
    from collections import Counter, defaultdict
    import requests

    url = f"https://huggingface.co/datasets/Chengheng/Webintosh/resolve/{REVISION}/{split}.jsonl"
    counts = Counter()
    groups = defaultdict(lambda: {"elements": 0, "action_targets": 0,
                                  "usable_targets": 0, "task": False})
    samples = []
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            row = json.loads(line)
            counts["rows"] += 1
            key = row["img_filename"]
            group = groups[key]
            group["elements"] += 1
            group["task"] |= bool(row.get("task"))
            if row.get("is_action_target"):
                counts["action_targets"] += 1
                group["action_targets"] += 1
                if row.get("style") == "imperative" and row.get("task"):
                    counts["usable_targets"] += 1
                    group["usable_targets"] += 1
                    if len(samples) < 3:
                        samples.append({key: row.get(key) for key in
                                        ("id", "instruction", "task", "bbox", "style",
                                         "trajectory_id", "img_filename")})
            if counts["rows"] % 100000 == 0:
                print(json.dumps({"split": split, "seen": counts["rows"]}), flush=True)
    counts["screenshots"] = len(groups)
    counts["screenshots_with_usable_target"] = sum(
        x["usable_targets"] > 0 and x["elements"] >= 4 for x in groups.values())
    counts["multi_target_screenshots"] = sum(x["action_targets"] > 1 for x in groups.values())
    counts["screenshots_with_task"] = sum(x["task"] for x in groups.values())
    return {"split": split, "revision": REVISION, "counts": dict(counts), "samples": samples}


@app.local_entrypoint()
def main():
    for result in audit.map(["train", "val", "test"]):
        print(json.dumps(result, indent=2))
