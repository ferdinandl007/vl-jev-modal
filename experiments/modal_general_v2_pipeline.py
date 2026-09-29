"""Detached Modal orchestration: wait for UCF101, merge v2, train and evaluate."""

import json

import modal

from experiments.dataset.modal_merge_general_v2 import app as merger_app, finalize
from experiments.modal_jev_omni_sports_train import (
    app as trainer_app,
    complete_general_head_pipeline,
    extract_general_shard,
)

app = modal.App("vl-jev-general-v2-background-pipeline")
app.include(merger_app)
app.include(trainer_app)

ucf_data = modal.Volume.from_name("vl-jev-ucf101-full-data")
general_data = modal.Volume.from_name("vl-jev-general-v1-data")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/ucf-data": ucf_data, "/general-data": general_data},
    timeout=86400,
)
def finish_v2(run_name: str = "general-head-v2", shards: int = 4,
              epochs: int = 3, wait_for_external_merge: bool = False):
    import time
    from pathlib import Path

    if not run_name or "/" in run_name or not 1 <= shards <= 16:
        raise ValueError("Invalid run name or shard count")
    summary_path = Path("/ucf-data/ucf101_full_v1/materialization_summary.json")
    manifest_path = Path("/ucf-data/ucf101_full_v1/summary.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["total_usable_rows"]
    if expected != 13320 or manifest["split_counts"] != {
        "train": 6729, "dev": 1379, "calibration": 1516, "test": 3696
    }:
        raise ValueError("Unexpected UCF101 manifest")
    deadline = time.monotonic() + 22 * 3600
    while time.monotonic() < deadline:
        ucf_data.reload()
        materialized = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        print(json.dumps({"phase": "waiting_for_ucf_media", "summary": materialized}),
              flush=True)
        if materialized is not None:
            ready = (materialized.get("skipped_existing", 0) +
                     materialized.get("materialized", 0))
            if materialized.get("missing_members", 1) == 0 and ready == expected:
                break
            raise ValueError(f"UCF101 materialization finished incomplete: {materialized}")
        time.sleep(60)
    else:
        raise TimeoutError("UCF101 materialization did not finish within 22 hours")

    if wait_for_external_merge:
        merged_path = Path("/general-data/v2/summary.json")
        deadline = time.monotonic() + 20 * 3600
        while time.monotonic() < deadline:
            general_data.reload()
            if merged_path.is_file():
                break
            print(json.dumps({"phase": "waiting_for_v2_merge"}), flush=True)
            time.sleep(60)
        else:
            raise TimeoutError("V2 merge did not finish within 20 hours")
        merged = json.loads(merged_path.read_text())
    else:
        merged = finalize.remote()
        general_data.reload()
    if merged["status"] != "complete" or merged["counts"]["train"] < 40000:
        raise ValueError(f"Unexpected v2 merge: {merged.get('counts')}")
    print(json.dumps({"phase": "v2_merged", "counts": merged["counts"],
                      "skipped": merged["skipped"]}), flush=True)

    calls = [extract_general_shard.spawn(run_name, shard, shards,
                                         ("train", "dev"), "v2")
             for shard in range(shards)]
    print(json.dumps({"phase": "v2_extraction_started", "shards": len(calls)}),
          flush=True)
    result = complete_general_head_pipeline.remote(run_name, "v2", shards, epochs)
    return {"merge_counts": merged["counts"], "merge_skipped": merged["skipped"],
            "training_result": result}


@app.local_entrypoint()
def main(run_name: str = "general-head-v2", shards: int = 4, epochs: int = 3,
         wait_for_external_merge: bool = False):
    print(json.dumps(finish_v2.remote(run_name, shards, epochs,
                                      wait_for_external_merge), indent=2))
