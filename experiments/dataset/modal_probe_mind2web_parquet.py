"""Inspect one pinned Multimodal Mind2Web train shard inside Modal."""

import json
import modal

app = modal.App("vl-jev-mind2web-parquet-probe")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("huggingface_hub==0.36.0", "pyarrow==20.0.0", "beautifulsoup4==4.13.5"))
REPO = "osunlp/Multimodal-Mind2Web"
REVISION = "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889"
SHARD = "data/train-00000-of-00027-4d11798d7219186d.parquet"


@app.function(image=image, timeout=1800, memory=8192)
def probe():
    import hashlib
    from pathlib import Path
    import pyarrow.parquet as pq
    from bs4 import BeautifulSoup
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(REPO, SHARD, revision=REVISION, repo_type="dataset")
    parquet = pq.ParquetFile(source)
    batch = next(parquet.iter_batches(batch_size=3))
    rows = batch.to_pylist()
    samples = []
    for row in rows:
        screenshot = row.get("screenshot")
        positive = [json.loads(item) for item in row.get("pos_candidates") or []]
        html = BeautifulSoup(row["cleaned_html"], "html.parser")
        choices = []
        for item in positive[:2]:
            attributes = json.loads(item.get("attributes") or "{}")
            node = html.find(attrs={"backend_node_id": item["backend_node_id"]})
            choices.append({"id": item["backend_node_id"],
                            "tag": item.get("tag"), "attributes": attributes,
                            "text": node.get_text(" ", strip=True)[:120] if node else None})
        samples.append({"action_uid": row["action_uid"],
                        "screenshot_type": type(screenshot).__name__,
                        "screenshot_keys": list(screenshot) if isinstance(screenshot, dict) else None,
                        "screenshot_byte_len": len(screenshot.get("bytes") or b"")
                        if isinstance(screenshot, dict) else None,
                        "operation": row["operation"],
                        "positive": choices,
                        "negative_count": len(row.get("neg_candidates") or []),
                        "history_index": row["target_action_index"]})
    return {"repo": REPO, "revision": REVISION, "shard": SHARD,
            "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "row_groups": parquet.num_row_groups,
            "rows": parquet.metadata.num_rows, "samples": samples}


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
