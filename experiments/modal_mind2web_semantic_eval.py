"""Evaluate task-conditioned element choices on a Modal-only pilot dev split."""

import json
import modal

from experiments.modal_jev_omni_sports_train import (
    _general_feature, _load_classifier, base_image, general_training, model_cache,
)

app = modal.App("vl-jev-mind2web-semantic-eval")
data = modal.Volume.from_name("vl-jev-gui-semantic-pilot-data")
media = modal.Volume.from_name("vl-jev-gui-semantic-pilot-media")


@app.function(image=base_image, gpu="H100", timeout=7200,
              volumes={"/gui-data": data, "/gui-media": media,
                       "/model-cache": model_cache,
                       "/general-runs": general_training})
def evaluate():
    import copy
    import hashlib
    from collections import Counter
    from pathlib import Path
    import torch

    root = Path("/gui-data/mind2web_semantic_pilot_v1")
    summary = json.loads((root / "summary.json").read_text())
    path = root / "dev.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["split_sha256"]["dev"]:
        raise ValueError("Pilot dev split changed")
    rows = [json.loads(line) for line in path.open()]
    if not rows:
        raise ValueError("No pilot development rows")
    torch, classifier, package = _load_classifier()
    trained = copy.deepcopy(classifier.head)
    trained.load_state_dict(torch.load(
        "/general-runs/general-head-v2/head-uniform.pt", map_location="cuda",
        weights_only=True))
    heads = {"pretrained": classifier.head, "general_v2": trained}
    counts = {view: {name: Counter() for name in heads}
              for view in ("image_plus_labels", "marked_image_plus_labels", "labels_only")}
    gold_slots = Counter()
    for index, row in enumerate(rows):
        labels = list(row["question"]["criteria"])
        target = labels.index(row["gold"]["key"])
        gold_slots[labels[target]] += 1
        size = torch.tensor([len(labels)], device="cuda")
        for view in counts:
            if view == "image_plus_labels":
                item = row
            elif view == "marked_image_plus_labels":
                item = dict(row, media={"kind": "modal_image",
                                        "path": row["extra"]["marked_media_path"]})
            else:
                item = dict(row, modality="text", media=None)
            feature = _general_feature(torch, classifier, package, item).to("cuda").unsqueeze(0)
            with torch.no_grad():
                for name, head in heads.items():
                    logits = head(feature, size)[0, :len(labels)]
                    counts[view][name]["correct"] += int(logits.argmax().item() == target)
                    counts[view][name]["rows"] += 1
        if (index + 1) % 20 == 0:
            print(json.dumps({"processed": index + 1, "total": len(rows),
                              "image_correct": {name: counts["image_plus_labels"][name]["correct"]
                                                for name in heads}}), flush=True)
    result = {"task": "task_history_named_control_next_click", "split": "pilot_dev",
              "source_shard": summary["shard"], "rows": len(rows),
              "chance": 0.25, "majority_slot_accuracy": max(gold_slots.values()) / len(rows),
              "gold_slots": dict(gold_slots),
              "results": {view: {name: {"correct": value["correct"], "n": value["rows"],
                                         "accuracy": value["correct"] / value["rows"]}
                                 for name, value in by_head.items()}
                          for view, by_head in counts.items()},
              "note": "A one-shard task-grouped development pilot, not an official Mind2Web test score."}
    (root / "evaluation-dev.json").write_text(json.dumps(result, indent=2))
    data.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(evaluate.remote(), indent=2))
