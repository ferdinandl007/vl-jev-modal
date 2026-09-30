"""Inspect screenshot/action and semantic candidate fields on Modal only."""

import json
import modal

app = modal.App("vl-jev-mind2web-semantic-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")
REPO = "osunlp/Multimodal-Mind2Web"


@app.function(image=image, timeout=300)
def probe():
    import requests

    metadata = requests.get(f"https://huggingface.co/api/datasets/{REPO}", timeout=60)
    metadata.raise_for_status()
    result = {"repo": REPO, "revision": metadata.json()["sha"], "splits": {}}
    for split in ("train", "test_task", "test_website", "test_domain"):
        response = requests.get("https://datasets-server.huggingface.co/rows",
                                params={"dataset": REPO, "config": "default",
                                        "split": split, "offset": 0, "length": 2}, timeout=120)
        if not response.ok:
            result["splits"][split] = {"status": response.status_code,
                                       "detail": response.text[:200]}
            continue
        payload = response.json()
        samples = []
        for entry in payload["rows"]:
            row = entry["row"]
            samples.append({"fields": list(row), "task": row.get("confirmed_task"),
                            "operation": row.get("operation"),
                            "history_step": row.get("target_action_index"),
                            "target_action_repr": row.get("target_action_reprs"),
                            "positive_candidates": row.get("pos_candidates", [])[:2],
                            "negative_candidates": row.get("neg_candidates", [])[:2],
                            "screenshot_available": bool(row.get("screenshot"))})
        result["splits"][split] = {"rows": payload["num_rows_total"],
                                   "samples": samples}
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
