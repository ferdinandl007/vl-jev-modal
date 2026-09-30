"""Compare the selected general v3 head with v2 on official Mind2Web test-task."""

import json
import modal

from experiments.modal_jev_omni_sports_train import (
    _general_feature, _general_metrics, _load_classifier, _options,
    base_image, general_training, model_cache,
)

app = modal.App("vl-jev-general-v3-gui-test")
test_data = modal.Volume.from_name("vl-jev-gui-test-task-data")
test_media = modal.Volume.from_name("vl-jev-gui-test-task-media")
gui_data = modal.Volume.from_name("vl-jev-gui-general-data")


@app.function(image=base_image, gpu="H100", timeout=86400,
              volumes={"/gui-test-data": test_data, "/gui-test-media": test_media,
                       "/gui-data": gui_data, "/general-runs": general_training,
                       "/model-cache": model_cache})
def evaluate():
    import copy
    import hashlib
    from pathlib import Path
    import torch

    test_root = Path("/gui-test-data/mind2web_test_task_v1")
    summary = json.loads((test_root / "summary.json").read_text())
    path = test_root / "test_task.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["split_sha256"]:
        raise ValueError("Official test-task split changed")
    rows = [json.loads(line) for line in path.open()]
    known_ids, known_groups = set(), set()
    for split in ("train", "dev"):
        for line in (Path("/gui-data/mind2web_general_v1") / f"{split}.jsonl").open():
            item = json.loads(line)
            known_ids.add(item["id"])
            known_groups.add(item["source"]["group"])
    if (known_ids & {row["id"] for row in rows} or
            known_groups & {row["source"]["group"] for row in rows}):
        raise ValueError("Training/test action or task overlap")
    run_root = Path("/general-runs/general-head-v3-gui")
    report = json.loads((run_root / "report.json").read_text())
    new_path = Path(report["checkpoint"])
    if hashlib.sha256(new_path.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("Selected v3 checkpoint changed")
    old_path = Path("/general-runs/general-head-v2/head-uniform.pt")
    backend, classifier, package = _load_classifier()
    old_head = copy.deepcopy(classifier.head)
    old_head.load_state_dict(torch.load(old_path, map_location="cuda", weights_only=True))
    new_head = copy.deepcopy(classifier.head)
    new_head.load_state_dict(torch.load(new_path, map_location="cuda", weights_only=True))
    items = []
    for index, row in enumerate(rows):
        labels, _ = _options(row)
        feature = _general_feature(backend, classifier, package, row)
        items.append({"feature": feature.half(), "target": labels.index(row["gold"]["key"]),
                      "count": len(labels), "modality": row["modality"],
                      "family": row["task_family"], "source_repo": row["source"]["repo"]})
        if (index + 1) % 100 == 0:
            print(json.dumps({"phase": "test_feature_extraction",
                              "processed": index + 1, "total": len(rows)}), flush=True)
    result = {"source_split": "official_test_task",
              "derived_task": "four_preselected_named_controls_with_marked_screenshot",
              "n": len(items), "dataset_sha256": summary["split_sha256"],
              "v2_checkpoint_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
              "v3_checkpoint_sha256": report["checkpoint_sha256"],
              "v3_selected_epoch": report["selected_epoch"],
              "baseline_v2": _general_metrics(backend, old_head, items),
              "selected_v3": _general_metrics(backend, new_head, items),
              "note": "Official source split, derived four-choice task; not the published Mind2Web metric or live task success."}
    destination = run_root / "official-test-task.json"
    if destination.is_file() and json.loads(destination.read_text()) != result:
        raise ValueError("Immutable official test result changed")
    destination.write_text(json.dumps(result, indent=2))
    general_training.commit()
    return {key: result[key] for key in ("source_split", "derived_task", "n",
                                        "v3_selected_epoch", "baseline_v2",
                                        "selected_v3", "note")}


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              timeout=86400, volumes={"/gui-test-data": test_data,
                                      "/general-runs": general_training})
def wait_and_evaluate():
    import time
    from pathlib import Path

    deadline = time.monotonic() + 20 * 3600
    while time.monotonic() < deadline:
        test_data.reload()
        general_training.reload()
        ready = {"test_data": Path("/gui-test-data/mind2web_test_task_v1/summary.json").is_file(),
                 "v3_report": Path("/general-runs/general-head-v3-gui/report.json").is_file()}
        if all(ready.values()):
            break
        print(json.dumps({"phase": "waiting_for_test_prerequisites", "ready": ready}),
              flush=True)
        time.sleep(60)
    else:
        raise TimeoutError("GUI test prerequisites did not finish")
    return evaluate.remote()


@app.local_entrypoint()
def main():
    print(json.dumps(wait_and_evaluate.remote(), indent=2))
