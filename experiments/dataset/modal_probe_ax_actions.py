"""Inspect accessibility-tree action rows on Modal without local downloads."""

import json
import modal

app = modal.App("vl-jev-probe-ax-actions")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==0.36.0", "pyarrow==20.0.0")
REPO = "saital/browser-agent-phase1-sft-action-only"


@app.function(image=image, timeout=1200)
def probe():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    info = api.dataset_info(REPO)
    paths = sorted(path for path in api.list_repo_files(
        REPO, repo_type="dataset", revision=info.sha) if path.endswith(".parquet"))
    if not paths:
        raise ValueError("No parquet files")
    path = hf_hub_download(REPO, paths[0], repo_type="dataset", revision=info.sha)
    table = pq.ParquetFile(path)
    samples = table.read_row_group(0).slice(0, 2).to_pylist()
    return {"repo": REPO, "revision": info.sha, "files": paths,
            "rows_first_file": table.metadata.num_rows, "schema": str(table.schema),
            "samples": [json.dumps(row, ensure_ascii=False)[:7000] for row in samples]}


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
