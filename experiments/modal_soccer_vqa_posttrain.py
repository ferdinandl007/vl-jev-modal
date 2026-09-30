"""Continue the published v2 Jev-Omni decision head on SoccerNet VQA.

All feature extraction, training, checkpoints, and evaluation run on Modal.
The frozen Jev-Omni backbone is shared with the v2 head. Existing v2 features
are replayed to limit forgetting; public SoccerNet test data is evaluation only.
"""

import json

import modal

from experiments.modal_jev_omni_sports_train import (
    MODEL_ID, MODEL_REVISION, _decision_hash, _general_feature, _general_metrics,
    _load_classifier, _load_general_features, _options, base_image, general_data,
    general_media, general_training, model_cache,
)

app = modal.App("vl-jev-soccer-vqa-20260930")
training_image = base_image.add_local_python_source("experiments")
RUN = "soccer-vqa-posttrain-v2"
RUNS = {"pilot": RUN, "all": "soccer-vqa-posttrain-full-v1"}
SOCCER_ROOT = "/general-data/soccer_vqa_2026"
BASE_RUN = "/general-runs/general-head-v2"
BASE_SHA256 = "e7771cb40f7cc41b5847e2a4880fb498572c86128c5d20cd0cce4f880211f780"
DIRECT_VISUAL_FAMILIES = frozenset({
    "soccer_vqa_action_classification",
    "soccer_vqa_camera_status_classification",
    "soccer_vqa_camera_status_switching",
    "soccer_vqa_game_state_relevant_qa",
    "soccer_vqa_jersy_color_relevant_qa",
    "soccer_vqa_multi_view_foul_recognition",
    "soccer_vqa_replay_grounding",
    "soccer_vqa_score_and_time_relevant_qa",
})


def _run_name(scope, focus):
    if scope not in RUNS or focus not in {"all", "direct_visual"}:
        raise ValueError("Invalid SoccerNet run selection")
    return RUNS[scope] + ("-direct-visual" if focus == "direct_visual" else "")


def _direct_visual_summary(metrics):
    groups = {key: value for key, value in metrics.items()
              if key.startswith("family:") and key.split(":", 1)[1] in DIRECT_VISUAL_FAMILIES}
    count = sum(value["n"] for value in groups.values())
    correct = sum(value["correct"] for value in groups.values())
    return {"n": count, "correct": correct,
            "accuracy": correct / count if count else None,
            "families": sorted(key.split(":", 1)[1] for key in groups)}


def _rows(split, per_type, scope="pilot"):
    import hashlib
    from pathlib import Path

    if scope not in RUNS:
        raise ValueError("Invalid SoccerNet scope")
    file = Path(SOCCER_ROOT) / f"{scope}-{per_type}-{split}.jsonl"
    descriptor = file.with_suffix(".json")
    if not file.is_file() or not descriptor.is_file():
        raise FileNotFoundError(f"SoccerNet {split} media has not finished materializing")
    record = json.loads(descriptor.read_text())
    if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"SoccerNet materialization changed: {split}")
    rows = [json.loads(line) for line in file.open()]
    if split != "test":
        protected_materials = set()
        protected_text_questions = set()
        for higher in (("test",) if split == "valid" else ("valid", "test")):
            for line in (Path(SOCCER_ROOT) / f"{higher}.jsonl").open():
                other = json.loads(line)
                protected_materials.update(other["source_materials"])
                if other["modality"] == "text":
                    protected_text_questions.add(json.dumps(other["question"], sort_keys=True))
        rows = [row for row in rows
                if not any(material in protected_materials for material in row["source_materials"])
                and not (row["modality"] == "text" and
                         json.dumps(row["question"], sort_keys=True) in protected_text_questions)]
    return rows, record["sha256"]


def _feature_path(split, shard, scope="pilot", part=None):
    from pathlib import Path

    name = f"shard-{shard:02d}.pt" if part is None else f"shard-{shard:02d}-part-{part:04d}.pt"
    return Path("/general-runs") / RUNS[scope] / "features" / split / name


@app.function(image=training_image, gpu="H100", timeout=86400,
              volumes={"/general-data": general_data, "/general-media": general_media,
                       "/general-runs": general_training, "/model-cache": model_cache})
def extract(split: str, per_type: int, shard: int, shards: int = 4,
            scope: str = "pilot"):
    import hashlib
    from pathlib import Path
    import torch

    if split not in {"train", "valid", "test"} or not 0 <= shard < shards <= 16:
        raise ValueError("Invalid feature extraction request")
    rows, source_sha = _rows(split, per_type, scope)
    selected = [row for row in rows if int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % shards == shard]
    chunk_size = 64 if scope == "all" else max(1, len(selected))
    chunks = [selected[offset:offset + chunk_size]
              for offset in range(0, len(selected), chunk_size)]
    reused = created = 0
    classifier = package = None
    for part, chunk in enumerate(chunks):
        destination = _feature_path(split, shard, scope, part if scope == "all" else None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            payload = torch.load(destination, map_location="cpu", weights_only=True)
            if (payload["source_sha256"] != source_sha or
                    payload["ids"] != [row["id"] for row in chunk] or
                    payload["model_revision"] != MODEL_REVISION):
                raise ValueError(f"SoccerNet feature cache mismatch: {destination}")
            reused += len(chunk)
            continue
        if classifier is None:
            torch, classifier, package = _load_classifier()
        packed = []
        for row in chunk:
            labels, _ = _options(row)
            feature = _general_feature(torch, classifier, package, row)
            if not bool(torch.isfinite(feature).all()):
                raise ValueError(f"Nonfinite SoccerNet feature: {row['id']}")
            packed.append({"id": row["id"], "feature": feature.half(),
                           "target": labels.index(row["gold"]["key"]), "count": len(labels),
                           "modality": row["modality"], "family": row["task_family"],
                           "source_repo": row["source"]["repo"],
                           "decision_hash": _decision_hash(row)})
        torch.save({"source_sha256": source_sha, "ids": [row["id"] for row in chunk],
                    "rows": packed, "model_revision": MODEL_REVISION}, destination)
        general_training.commit()
        created += len(chunk)
        print(json.dumps({"phase": "soccer_features", "scope": scope, "split": split,
                          "shard": shard, "done": reused + created,
                          "total": len(selected)}), flush=True)
    return {"scope": scope, "split": split, "shard": shard,
            "rows": len(selected), "reused": reused, "created": created}


def _load_soccer_features(torch, split, per_type, shards=4, scope="pilot"):
    rows, source_sha = _rows(split, per_type, scope)
    expected = {row["id"]: row for row in rows}
    found = {}
    for shard in range(shards):
        if scope == "all":
            base = _feature_path(split, shard, scope)
            paths = sorted(base.parent.glob(f"shard-{shard:02d}-part-*.pt"))
        else:
            paths = [_feature_path(split, shard, scope)]
        if not paths:
            raise FileNotFoundError(f"No SoccerNet feature packs for {split}:{shard}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["source_sha256"] != source_sha or payload["model_revision"] != MODEL_REVISION:
                raise ValueError(f"Stale SoccerNet feature pack: {path}")
            for item in payload["rows"]:
                row = expected.get(item["id"])
                if row is None or item["id"] in found or item["decision_hash"] != _decision_hash(row):
                    raise ValueError(f"SoccerNet feature mismatch: {path}")
                found[item["id"]] = item
    if set(found) != set(expected):
        raise ValueError(f"Incomplete SoccerNet {split} features: {len(found)}/{len(expected)}")
    return [found[row["id"]] for row in rows]


def _head(torch, feature_size):
    import hashlib
    import sys
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(source).parent))
    import jev_omni

    checkpoint = Path(BASE_RUN) / "head-uniform.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Published v2 head checkpoint changed")
    head = jev_omni._Head256(feature_size).to("cuda")
    head.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    return head


@app.function(image=training_image, gpu="H100", timeout=86400, memory=8192,
              volumes={"/general-data": general_data, "/general-runs": general_training,
                       "/model-cache": model_cache})
def train(train_per_type: int = 32, valid_per_type: int = 8, epochs: int = 3,
          replay_ratio: int = 2, scope: str = "pilot", focus: str = "all"):
    import copy
    import hashlib
    import random
    from pathlib import Path
    import torch

    run_name = _run_name(scope, focus)
    if not 1 <= epochs <= 8 or not 0 <= replay_ratio <= 8:
        raise ValueError("Invalid training settings")
    soccer_train = _load_soccer_features(torch, "train", train_per_type, scope=scope)
    soccer_valid = _load_soccer_features(torch, "valid", valid_per_type, scope=scope)
    if focus == "direct_visual":
        soccer_train = [item for item in soccer_train if item["family"] in DIRECT_VISUAL_FAMILIES]
        soccer_valid = [item for item in soccer_valid if item["family"] in DIRECT_VISUAL_FAMILIES]
    if not soccer_train or not soccer_valid:
        raise ValueError("No SoccerNet examples in selected focus")
    v2_train = _load_general_features(torch, BASE_RUN, "train", "v2")
    v2_dev = _load_general_features(torch, BASE_RUN, "dev", "v2")
    head = _head(torch, soccer_train[0]["feature"].numel())
    baseline_soccer = _general_metrics(torch, head, soccer_valid)
    baseline_v2 = _general_metrics(torch, head, v2_dev)
    root = Path("/general-runs") / run_name
    root.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-5, weight_decay=0.01)
    history = []
    best_state = None
    best_score = (-1, -1.0, float("-inf"))
    for epoch in range(epochs):
        replay = random.Random(3407 + epoch).sample(
            v2_train, min(len(v2_train), replay_ratio * len(soccer_train)))
        items = soccer_train + replay
        random.Random(7351 + epoch).shuffle(items)
        head.train()
        for start in range(0, len(items), 128):
            batch = items[start:start + 128]
            features = torch.stack([item["feature"] for item in batch]).to("cuda").float()
            counts = torch.tensor([item["count"] for item in batch], device="cuda")
            targets = torch.tensor([item["target"] for item in batch], device="cuda")
            logits = head(features, counts)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        soccer = _general_metrics(torch, head, soccer_valid)
        v2 = _general_metrics(torch, head, v2_dev)
        guard = (v2["overall"]["accuracy"] >= baseline_v2["overall"]["accuracy"] - 0.01 and
                 v2["vision"]["accuracy"] >= baseline_v2["vision"]["accuracy"] - 0.02)
        score = (int(guard), soccer["overall"]["accuracy"],
                 -soccer["overall"]["log_loss"])
        history.append({"epoch": epoch + 1, "soccer_valid": soccer,
                        "v2_dev": v2, "retention_guard": guard})
        if guard and score > best_score:
            best_score = score
            best_state = copy.deepcopy(head.state_dict())
        print(json.dumps({"epoch": epoch + 1, "soccer_accuracy": soccer["overall"]["accuracy"],
                          "v2_accuracy": v2["overall"]["accuracy"], "guard": guard}), flush=True)
    report = {"run": run_name, "scope": scope, "focus": focus,
              "method": "continued_v2_head_frozen_backbone",
              "base_checkpoint_sha256": BASE_SHA256, "backbone_revision": MODEL_REVISION,
              "train_rows": len(soccer_train), "valid_rows": len(soccer_valid),
              "v2_replay_ratio": replay_ratio, "epochs": epochs,
              "baseline_soccer_valid": baseline_soccer, "baseline_v2_dev": baseline_v2,
              "history": history, "selected_epoch": None, "status": "no_guarded_improvement"}
    if best_state is not None and best_score[1] > baseline_soccer["overall"]["accuracy"]:
        checkpoint = root / "head-continued.pt"
        torch.save(best_state, checkpoint)
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report["selected_epoch"] = next(item["epoch"] for item in history
                                        if item["soccer_valid"]["overall"]["accuracy"] == best_score[1]
                                        and item["retention_guard"])
        report["status"] = "guarded_improvement"
    (root / "report.json").write_text(json.dumps(report, indent=2))
    general_training.commit()
    return {"status": report["status"], "selected_epoch": report["selected_epoch"],
            "baseline_soccer_accuracy": baseline_soccer["overall"]["accuracy"],
            "best_soccer_accuracy": best_score[1],
            "checkpoint_sha256": report.get("checkpoint_sha256")}


@app.function(image=training_image, gpu="H100", timeout=86400,
              volumes={"/general-data": general_data, "/general-runs": general_training,
                       "/model-cache": model_cache})
def evaluate(test_per_type: int = 8, scope: str = "pilot", focus: str = "all"):
    import hashlib
    import random
    from pathlib import Path
    import torch

    run_name = _run_name(scope, focus)
    # The public test features and labels are identical for both training scopes.
    test = _load_soccer_features(torch, "test", test_per_type, scope="pilot")
    base = _head(torch, test[0]["feature"].numel())

    def correctness(head):
        head.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(test), 128):
                batch = test[start:start + 128]
                features = torch.stack([item["feature"] for item in batch]).to("cuda").float()
                counts = torch.tensor([item["count"] for item in batch], device="cuda")
                predictions = head(features, counts).argmax(-1).tolist()
                values.extend(int(prediction == item["target"])
                              for prediction, item in zip(predictions, batch))
        return values

    baseline = _general_metrics(torch, base, test)
    base_correct = correctness(base)
    root = Path("/general-runs") / run_name
    report = json.loads((root / "report.json").read_text())
    continued = None
    if report["status"] == "guarded_improvement":
        path = Path(report["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise ValueError("Continued head checkpoint changed")
        base.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
        continued = _general_metrics(torch, base, test)
        continued_correct = correctness(base)
        gains = [after - before for before, after in zip(base_correct, continued_correct)]
        bootstrap = []
        rng = random.Random(3407)
        for _ in range(2000):
            bootstrap.append(sum(gains[rng.randrange(len(gains))] for _ in gains) / len(gains))
        bootstrap.sort()
        paired = {"improved_only": sum(before == 0 and after == 1
                                        for before, after in zip(base_correct, continued_correct)),
                  "regressed_only": sum(before == 1 and after == 0
                                         for before, after in zip(base_correct, continued_correct)),
                  "delta_accuracy": sum(gains) / len(gains),
                  "bootstrap_95pct_delta": [bootstrap[50], bootstrap[1949]]}
    else:
        paired = None
    output = {"run": run_name, "scope": scope, "focus": focus,
              "official_test_source_rows": 500,
              "scorable_rows": len(test), "unscorable_source_rows": 500 - len(test),
              "base_v2": baseline, "continued": continued,
              "paired_comparison": paired,
              "status": report["status"], "source": "SoccerNet/SN-VQA-2026",
              "note": "Public test is not the 2026 challenge hidden set; 4 malformed source rows are unscorable."}
    (root / "soccer-test-evaluation.json").write_text(json.dumps(output, indent=2))
    general_training.commit()
    return {"scorable_rows": len(test), "baseline_accuracy": baseline["overall"]["accuracy"],
            "continued_accuracy": continued["overall"]["accuracy"] if continued else None,
            "paired_comparison": paired,
            "status": report["status"]}


@app.function(image=training_image,
              volumes={"/general-runs": general_training}, timeout=300)
def status(scope: str = "pilot", focus: str = "all"):
    from pathlib import Path

    root = Path("/general-runs") / _run_name(scope, focus)
    return {"features": {split: len(list((root / "features" / split).glob("shard-*.pt")))
                         for split in ("train", "valid", "test")},
            "report": json.loads((root / "report.json").read_text())["status"]
            if (root / "report.json").is_file() else None,
            "test": json.loads((root / "soccer-test-evaluation.json").read_text())["status"]
            if (root / "soccer-test-evaluation.json").is_file() else None}


@app.function(image=training_image,
              volumes={"/general-runs": general_training}, timeout=300)
def report_summary(scope: str = "pilot", focus: str = "all"):
    from pathlib import Path

    root = Path("/general-runs") / _run_name(scope, focus)
    training = json.loads((root / "report.json").read_text())
    result = json.loads((root / "soccer-test-evaluation.json").read_text())
    selected = (training["history"][training["selected_epoch"] - 1]["soccer_valid"]
                if training["selected_epoch"] else {})
    return {"scope": scope, "focus": focus, "train_rows": training["train_rows"],
            "valid_rows": training["valid_rows"],
            "selected_epoch": training["selected_epoch"],
            "checkpoint_sha256": training.get("checkpoint_sha256"),
            "v2_dev_baseline": training["baseline_v2_dev"]["overall"],
            "v2_dev_after": training["history"][training["selected_epoch"] - 1]["v2_dev"]["overall"]
            if training["selected_epoch"] else None,
            "valid_base": training["baseline_soccer_valid"]["overall"],
            "valid_direct_visual_base": _direct_visual_summary(training["baseline_soccer_valid"]),
            "valid_direct_visual_after": _direct_visual_summary(selected) if selected else None,
            "valid_after": training["history"][training["selected_epoch"] - 1]["soccer_valid"]["overall"]
            if training["selected_epoch"] else None,
            "valid_by_task_base": {key: value for key, value in
                                   training["baseline_soccer_valid"].items()
                                   if key.startswith("family:")},
            "valid_by_task_after": {key: value for key, value in
                                    (training["history"][training["selected_epoch"] - 1]["soccer_valid"]
                                     if training["selected_epoch"] else {}).items()
                                    if key.startswith("family:")},
            "test_base": {key: value for key, value in result["base_v2"].items()
                          if key in {"overall", "text", "vision"} or
                          key.startswith(("modality:", "family:"))},
            "test_after": {key: value for key, value in (result["continued"] or {}).items()
                           if key in {"overall", "text", "vision"} or
                           key.startswith(("modality:", "family:"))},
            "paired": result["paired_comparison"]}


@app.function(image=training_image, gpu="H100", timeout=3600,
              volumes={"/general-data": general_data,
                       "/general-runs": general_training,
                       "/model-cache": model_cache})
def valid_baseline(scope: str = "all"):
    """Report frozen Gemma/Jev-head accuracy before any SoccerNet continuation."""
    import torch

    if scope not in RUNS:
        raise ValueError("Invalid SoccerNet scope")
    rows = _load_soccer_features(torch, "valid", 8, scope=scope)
    head = _head(torch, rows[0]["feature"].numel())
    metrics = _general_metrics(torch, head, rows)
    return {"scope": scope, "all": metrics["overall"],
            "direct_visual": _direct_visual_summary(metrics),
            "by_task": {key: value for key, value in metrics.items()
                        if key.startswith("family:")}}


@app.function(image=training_image, timeout=86400,
              volumes={"/general-data": general_data,
                       "/general-runs": general_training})
def full_pipeline():
    """Wait for complete source materialization, then run the remote study."""
    import time
    from pathlib import Path

    required = [Path(SOCCER_ROOT) / "all-32-train.json",
                Path(SOCCER_ROOT) / "all-8-valid.json"]
    deadline = time.monotonic() + 20 * 3600
    while time.monotonic() < deadline:
        general_data.reload()
        if all(path.is_file() for path in required):
            break
        print(json.dumps({"phase": "waiting_for_full_media",
                          "ready": [path.is_file() for path in required]}), flush=True)
        time.sleep(60)
    else:
        raise TimeoutError("Full SoccerNet materialization did not finish within 20 hours")
    jobs = [("train", 32, shard) for shard in range(4)] + [
        ("valid", 8, shard) for shard in range(4)]
    extracted = list(extract.map(
        [job[0] for job in jobs], [job[1] for job in jobs],
        [job[2] for job in jobs], [4] * len(jobs), ["all"] * len(jobs)))
    general_training.reload()
    training = train.remote(scope="all")
    general_training.reload()
    evaluation = evaluate.remote(scope="all")
    result = {"scope": "all", "extracted": extracted,
              "training": training, "evaluation": evaluation}
    root = Path("/general-runs") / RUNS["all"]
    (root / "pipeline-result.json").write_text(json.dumps(result, indent=2))
    general_training.commit()
    return result


@app.function(image=training_image, timeout=86400,
              volumes={"/general-runs": general_training})
def direct_visual_pipeline():
    """Reuse the full Modal feature packs for a visual-evidence head comparison."""
    import time
    from pathlib import Path

    prerequisite = Path("/general-runs") / RUNS["all"] / "pipeline-result.json"
    deadline = time.monotonic() + 20 * 3600
    while time.monotonic() < deadline:
        general_training.reload()
        if prerequisite.is_file():
            break
        print(json.dumps({"phase": "waiting_for_full_features"}), flush=True)
        time.sleep(60)
    else:
        raise TimeoutError("Full SoccerNet feature extraction did not finish")
    training = train.remote(scope="all", focus="direct_visual")
    general_training.reload()
    evaluation = evaluate.remote(scope="all", focus="direct_visual")
    result = {"scope": "all", "focus": "direct_visual",
              "training": training, "evaluation": evaluation}
    root = Path("/general-runs") / _run_name("all", "direct_visual")
    (root / "pipeline-result.json").write_text(json.dumps(result, indent=2))
    general_training.commit()
    return result


@app.local_entrypoint()
def main(mode: str = "status", split: str = "train", per_type: int = 32,
         shard: int = 0, shards: int = 4, scope: str = "pilot",
         focus: str = "all"):
    if mode == "extract":
        result = extract.remote(split, per_type, shard, shards, scope)
    elif mode == "train":
        result = train.remote(scope=scope, focus=focus)
    elif mode == "evaluate":
        result = evaluate.remote(scope=scope, focus=focus)
    elif mode == "status":
        result = status.remote(scope, focus)
    elif mode == "report":
        result = report_summary.remote(scope, focus)
    elif mode == "valid-baseline":
        result = valid_baseline.remote(scope)
    elif mode == "full-pipeline":
        result = full_pipeline.remote()
    elif mode == "direct-visual-pipeline":
        result = direct_visual_pipeline.remote()
    else:
        raise ValueError("Invalid SoccerNet mode")
    print(json.dumps(result, indent=2))
