"""Inspect the raw Webintosh records without downloading data locally."""

import json
import modal

app = modal.App("vl-jev-webintosh-raw-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")


@app.function(image=image, timeout=300)
def probe():
    import requests

    revision = "7709afcb895aa7a3e39e2aa10406b930ad3956f8"
    results = {}
    for split in ("train", "val", "test"):
        url = f"https://huggingface.co/datasets/Chengheng/Webintosh/resolve/{revision}/{split}.jsonl"
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(json.loads(line))
                if len(lines) == 3:
                    break
            results[split] = {"status": response.status_code,
                              "content_length": response.headers.get("Content-Length"),
                              "examples": lines}
    return results


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
