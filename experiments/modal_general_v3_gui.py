"""Retrain the same general Jev-Omni head with GUI data and full v2 replay.

All model inference, feature storage, training, and evaluation run on Modal.
The published v2 head remains immutable. Dataset builders create the GUI rows.
"""

import json
import modal

from experiments.modal_jev_omni_sports_train import (
    MODEL_ID, MODEL_REVISION, _decision_hash, _general_feature,
    _general_metrics, _load_classifier, _load_general_features, _options,
    base_image, general_data, general_training, model_cache,
)

app = modal.App("vl-jev-general-v3-gui-training")
gui_data = modal.Volume.from_name("vl-jev-gui-general-data")
gui_media = modal.Volume.from_name("vl-jev-gui-general-media")
RUN = "general-head-v3-gui"
SOURCES = {"mind2web": "mind2web_general_v1", "ax": "ax_actions_general_v1"}


def _gui_rows(split):
    import hashlib
    from pathlib import Path

    if split not in {"train", "dev"}:
        raise ValueError("GUI feature extraction uses train/dev only")
    rows = []
    for name, folder in SOURCES.items():
        root = Path("/gui-data") / folder
        summary = json.loads((root / "summary.json").read_text())
        path = root / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["split_sha256"][split]:
            raise ValueError(f"GUI split changed: {name}/{split}")
        rows.extend(json.loads(line) for line in path.open())
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate GUI IDs")
    return rows


def _worker_rows(split, worker, workers):
    import hashlib

    return [row for row in _gui_rows(split)
            if int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % workers == worker]


@app.function(image=base_image, gpu="H100", timeout=86400, max_containers=4,
              volumes={"/gui-data": gui_data, "/gui-media": gui_media,
                       "/model-cache": model_cache,
                       "/general-runs": general_training})
def extract_worker(worker: int, workers: int = 4):
    from pathlib import Path
    import torch

    if not 0 <= worker < workers <= 8:
        raise ValueError("Invalid worker count")
    root = Path("/general-runs") / RUN / "features"
    result = {}
    model = None
    for split in ("train", "dev"):
        rows = _worker_rows(split, worker, workers)
        made = reused = 0
        for offset in range(0, len(rows), 64):
            chunk = rows[offset:offset + 64]
            destination = root / split / f"pack-gui-{worker:02d}-{offset // 64:05d}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                saved = torch.load(destination, map_location="cpu", weights_only=True)["rows"]
                if len(saved) != len(chunk) or any(
                    item["id"] != row["id"] or item["decision_hash"] != _decision_hash(row)
                    for item, row in zip(saved, chunk)):
                    raise ValueError(f"Stale GUI feature pack: {destination}")
                reused += len(chunk)
            else:
                if model is None:
                    model = _load_classifier()
                backend, classifier, package = model
                saved = []
                for row in chunk:
                    labels, _ = _options(row)
                    feature = _general_feature(backend, classifier, package, row)
                    if not bool(torch.isfinite(feature).all()):
                        raise ValueError(f"Nonfinite GUI feature: {row['id']}")
                    saved.append({"id": row["id"], "feature": feature.half(),
                                  "target": labels.index(row["gold"]["key"]),
                                  "count": len(labels), "modality": row["modality"],
                                  "family": row["task_family"],
                                  "decision_hash": _decision_hash(row),
                                  "source_repo": row["source"]["repo"]})
                torch.save({"rows": saved, "worker": worker, "workers": workers},
                           destination)
                made += len(chunk)
            general_training.commit()
            print(json.dumps({"phase": "extract_gui", "worker": worker,
                              "split": split, "made": made, "reused": reused,
                              "total": len(rows)}), flush=True)
        result[split] = {"made": made, "reused": reused, "total": len(rows)}
    return {"worker": worker, "splits": result}


def _load_gui_features(torch, split):
    from pathlib import Path

    expected = {row["id"]: row for row in _gui_rows(split)}
    root = Path("/general-runs") / RUN / "features" / split
    found = {}
    for path in sorted(root.glob("pack-gui-*.pt")):
        pack = torch.load(path, map_location="cpu", weights_only=True)
        for item in pack["rows"]:
            row = expected.get(item["id"])
            if row is None or item["id"] in found:
                raise ValueError(f"Unexpected or duplicate GUI feature: {path}")
            labels, _ = _options(row)
            if (item["target"] != labels.index(row["gold"]["key"]) or
                    item["count"] != len(labels) or
                    item["modality"] != row["modality"] or
                    item["family"] != row["task_family"] or
                    item["decision_hash"] != _decision_hash(row) or
                    item["source_repo"] != row["source"]["repo"]):
                raise ValueError(f"GUI feature metadata mismatch: {path}")
            found[item["id"]] = item
    if set(found) != set(expected):
        raise ValueError(f"GUI features incomplete: {split}: {len(found)}/{len(expected)}")
    return list(found.values())


def _score(torch, head, general_dev, gui_dev):
    general = _general_metrics(torch, head, general_dev)
    gui = _general_metrics(torch, head, gui_dev)
    source_scores = [value["accuracy"] for key, value in gui.items()
                     if key.startswith("source:")]
    return {"general": general, "gui": gui,
            "gui_source_macro": sum(source_scores) / len(source_scores)}


@app.function(image=base_image, gpu="H100", timeout=86400,
              volumes={"/gui-data": gui_data, "/general-data": general_data,
                       "/general-runs": general_training, "/model-cache": model_cache})
def train(epochs: int = 3):
    import copy
    import hashlib
    import random
    import sys
    from collections import Counter
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download

    if not 1 <= epochs <= 6:
        raise ValueError("Invalid epoch count")
    v2_root = Path("/general-runs/general-head-v2")
    v2_report = json.loads((v2_root / "report-uniform.json").read_text())
    checkpoint = v2_root / "head-uniform.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != v2_report["checkpoint_sha256"]:
        raise ValueError("Published general v2 checkpoint changed")
    general_train = _load_general_features(torch, v2_root, "train", "v2")
    general_dev = _load_general_features(torch, v2_root, "dev", "v2")
    gui_train = _load_gui_features(torch, "train")
    gui_dev = _load_gui_features(torch, "dev")
    source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(source).parent))
    import jev_omni
    head = jev_omni._Head256(general_train[0]["feature"].numel()).to("cuda")
    head.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    baseline = _score(torch, head, general_dev, gui_dev)
    train_rows = general_train + gui_train
    optimum = torch.optim.AdamW(head.parameters(), lr=5e-5, weight_decay=0.01)
    history = []
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    best_gui = baseline["gui_source_macro"]
    for epoch in range(epochs):
        head.train()
        order = list(range(len(train_rows)))
        random.Random(3407 + epoch).shuffle(order)
        for start in range(0, len(order), 128):
            batch = [train_rows[index] for index in order[start:start + 128]]
            features = torch.stack([item["feature"] for item in batch]).to("cuda").float()
            counts = torch.tensor([item["count"] for item in batch], device="cuda")
            targets = torch.tensor([item["target"] for item in batch], device="cuda")
            loss = torch.nn.functional.cross_entropy(head(features, counts), targets)
            optimum.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimum.step()
        metrics = _score(torch, head, general_dev, gui_dev)
        old = baseline["general"]
        new = metrics["general"]
        guard = (new["text"]["accuracy"] >= old["text"]["accuracy"] - 0.015 and
                 new["vision"]["accuracy"] >= old["vision"]["accuracy"] - 0.015 and
                 new["overall"]["log_loss"] <= old["overall"]["log_loss"] + 0.03)
        history.append({"epoch": epoch + 1, "guard_passed": guard, **metrics})
        if guard and metrics["gui_source_macro"] > best_gui:
            best_state = copy.deepcopy(head.state_dict())
            best_gui = metrics["gui_source_macro"]
            best_epoch = epoch + 1
        print(json.dumps({"epoch": epoch + 1, "guard": guard,
                          "general": new["overall"], "gui": metrics["gui"]["overall"],
                          "gui_source_macro": metrics["gui_source_macro"]}), flush=True)
    head.load_state_dict(best_state)
    selected = _score(torch, head, general_dev, gui_dev)
    root = Path("/general-runs") / RUN
    root.mkdir(parents=True, exist_ok=True)
    output = root / "head-uniform.pt"
    torch.save(best_state, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {"run": RUN, "method": "frozen_backbone_general_head_replay",
              "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
              "parent_checkpoint_sha256": v2_report["checkpoint_sha256"],
              "checkpoint": str(output), "checkpoint_sha256": digest,
              "epochs": epochs, "selected_epoch": best_epoch,
              "train_rows": len(train_rows), "general_replay_rows": len(general_train),
              "gui_train_rows": len(gui_train), "general_dev_rows": len(general_dev),
              "gui_dev_rows": len(gui_dev),
              "train_by_source": dict(Counter(item["source_repo"] for item in train_rows)),
              "baseline_dev": baseline, "history": history,
              "selected_dev": selected,
              "selection_policy": "maximize equal-source GUI dev accuracy subject to general text and vision accuracy within 1.5pp of v2 and general log loss within 0.03; fall back to v2",
              "official_gui_test_used_for_selection": False,
              "source_validation_split_used_for_training": False}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    general_training.commit()
    return {key: report[key] for key in ("run", "checkpoint_sha256", "selected_epoch",
                                        "train_rows", "general_replay_rows", "gui_train_rows",
                                        "general_dev_rows", "gui_dev_rows", "baseline_dev",
                                        "selected_dev")}


@app.function(image=base_image, timeout=86400, volumes={"/gui-data": gui_data,
              "/general-runs": general_training})
def pipeline(workers: int = 4):
    import time
    from pathlib import Path

    if not 1 <= workers <= 8:
        raise ValueError("Invalid worker count")
    deadline = time.monotonic() + 20 * 3600
    while time.monotonic() < deadline:
        gui_data.reload()
        ready = {name: (Path("/gui-data") / folder / "summary.json").is_file()
                 for name, folder in SOURCES.items()}
        if all(ready.values()):
            break
        print(json.dumps({"phase": "waiting_for_gui_data", "ready": ready}), flush=True)
        time.sleep(60)
    else:
        raise TimeoutError("GUI data build did not finish")
    extraction = list(extract_worker.map(range(workers), [workers] * workers))
    general_training.reload()
    result = train.remote()
    return {"extraction": extraction,
            "training": {key: result[key] for key in ("run", "checkpoint_sha256",
                                                   "selected_epoch", "train_rows",
                                                   "gui_train_rows")},
            "report_path": f"/general-runs/{RUN}/report.json"}


@app.local_entrypoint()
def main():
    print(json.dumps(pipeline.remote(), indent=2))
