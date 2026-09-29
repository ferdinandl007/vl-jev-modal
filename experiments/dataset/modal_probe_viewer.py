"""Probe selected Hugging Face Dataset Viewer rows on Modal."""

import json
import modal

app = modal.App("vl-jev-viewer-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")


@app.function(image=image, timeout=180)
def probe(repo, split, offsets):
    import requests

    session = requests.Session()
    result = []
    for offset in offsets:
        response = session.get("https://datasets-server.huggingface.co/rows",
                               params={"dataset": repo, "config": "default", "split": split,
                                       "offset": offset, "length": 1}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        item = payload["rows"][0]["row"]
        result.append({"offset": offset, "num_rows_total": payload["num_rows_total"],
                       "fields": {key: ("image_url" if isinstance(value, dict) and value.get("src")
                                          else "null" if value is None else type(value).__name__)
                                  for key, value in item.items()},
                       "choice_count": len(item.get("choices", []))
                       if isinstance(item.get("choices"), list) else None,
                       "image_src_prefix": str(item.get("image", {}).get("src", ""))[:100]
                       if isinstance(item.get("image"), dict) else None})
    return {"repo": repo, "split": split, "rows": result}


@app.local_entrypoint()
def main():
    cases = [("HuggingFaceM4/A-OKVQA", "train", [0, 5000, 17000]),
             ("HuggingFaceM4/A-OKVQA", "validation", [0]),
             ("Gisiyuan/ScienceQA", "train", [0, 100, 1000, 12000]),
             ("Gisiyuan/ScienceQA", "validation", [0, 100, 1000])]
    for repo, split, offsets in cases:
        print(json.dumps(probe.remote(repo, split, offsets)))
