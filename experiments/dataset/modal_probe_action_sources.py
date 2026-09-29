"""Inspect public driving and GUI-action dataset metadata on Modal only."""

import json
import modal

app = modal.App("vl-jev-action-source-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")


@app.function(image=image, timeout=300)
def probe():
    import requests

    result = {}
    for repo in ("WaltonFuture/DrivingVQA", "Chengheng/Webintosh",
                 "OpenGVLab/ScaleCUA-Data", "immanuelpeter/carla-autopilot-images",
                 "urjc-deepracer/carla-expert-racing"):
        response = requests.get(f"https://huggingface.co/api/datasets/{repo}",
                                timeout=60)
        response.raise_for_status()
        metadata = response.json()
        siblings = metadata.get("siblings") or []
        result[repo] = {
            "sha": metadata["sha"],
            "private": metadata.get("private"),
            "gated": metadata.get("gated"),
            "license": (metadata.get("cardData") or {}).get("license"),
            "files": len(siblings),
            "sample_files": [item["rfilename"] for item in siblings[:12]],
            "parquet_files": [item["rfilename"] for item in siblings
                              if item["rfilename"].endswith(".parquet")][:12],
            "jsonl_files": [item["rfilename"] for item in siblings
                            if item["rfilename"].endswith(".jsonl")][:12],
            "split_like_files": [item["rfilename"] for item in siblings
                                 if any(piece in item["rfilename"].lower()
                                        for piece in ("train", "validation", "test", ".csv"))][:20],
        }
        for split in (("train", "test") if repo.endswith("DrivingVQA")
                      else ("train", "validation", "test")):
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": repo, "config": "default", "split": split,
                        "offset": 0, "length": 2}, timeout=90)
            if response.ok:
                payload = response.json()
                result[repo][split] = {
                    "rows": payload["num_rows_total"],
                    "examples": [{"row_idx": entry["row_idx"],
                                  "row": {key: (value if key not in {"image", "images"}
                                                else str(value)[:250])
                                          for key, value in entry["row"].items()}}
                                 for entry in payload["rows"]],
                }
            else:
                result[repo][split] = {"error": response.status_code,
                                       "detail": response.text[:250]}
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
