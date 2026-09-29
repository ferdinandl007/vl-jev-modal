"""Inspect Jev-style text data schema and splits on Modal."""

import json
import modal

app = modal.App("vl-jev-typed-source-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")


@app.function(image=image, timeout=300)
def probe():
    import requests
    repo = "tasksource/tasksource-jev-typed-decisions"
    metadata = requests.get(f"https://huggingface.co/api/datasets/{repo}", timeout=60)
    metadata.raise_for_status()
    splits = requests.get("https://datasets-server.huggingface.co/splits",
                          params={"dataset": repo}, timeout=60)
    splits.raise_for_status()
    result = {"repo": repo, "sha": metadata.json()["sha"],
              "license": (metadata.json().get("cardData") or {}).get("license"),
              "splits": splits.json().get("splits"), "rows": {}}
    for split in ("train", "validation", "test"):
        response = requests.get("https://datasets-server.huggingface.co/rows",
                                params={"dataset": repo, "config": "default",
                                        "split": split, "offset": 0, "length": 4},
                                timeout=90)
        result["rows"][split] = (response.json() if response.ok else
                                 {"error": response.status_code,
                                  "detail": response.text[:300]})
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
