"""Inspect GUI decision supervision and saved metrics entirely on Modal."""

import json
from collections import Counter
from pathlib import Path

import modal

from experiments.modal_jev_omni_sports_train import base_image, general_media, model_cache

app = modal.App("vl-jev-gui-diagnostics")
data = modal.Volume.from_name("vl-jev-general-v1-data")
training = modal.Volume.from_name("vl-jev-general-v1-training")


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              volumes={"/general-data": data, "/general-runs": training}, timeout=300)
def inspect():
    root = Path("/general-runs/general-head-v2")
    output = {"model_metrics": {}, "splits": {}}
    for weighting in ("uniform", "sqrt_inverse_source"):
        report = json.loads((root / f"report-{weighting}.json").read_text())
        family = "family:gui_next_click"
        output["model_metrics"][weighting] = {
            "baseline_dev": report["baseline_dev"].get(family),
            "epochs_dev": [epoch["dev"].get(family) for epoch in report["history"]],
            "selected_dev": report["selected_dev"].get(family),
            "selected_vision": report["selected_dev"].get("vision"),
            "selected_text": report["selected_dev"].get("text"),
        }
    for split in ("train", "dev", "calibration", "test"):
        path = Path("/general-data/v2") / f"{split}.jsonl"
        rows = (json.loads(line) for line in path.open())
        gui = [row for row in rows if row["task_family"] == "gui_next_click"]
        gold = Counter(row["gold"]["key"] for row in gui)
        groups = Counter(row["source"].get("group") for row in gui)
        samples = [{"task": row["state"],
                    "local_target_instruction": row["extra"].get("target_instruction"),
                    "options": row["question"]["criteria"],
                    "gold": row["gold"]["key"]}
                   for row in gui[:12]]
        output["splits"][split] = {"rows": len(gui), "gold_positions": dict(gold),
                                    "trajectories": len(groups), "samples": samples}
    return output


@app.function(image=base_image,
              gpu="H100", timeout=7200,
              volumes={"/general-data": data, "/general-runs": training,
                       "/general-media": general_media,
                       "/model-cache": model_cache})
def ablate_local_intent():
    """An oracle-intent grounding diagnostic on dev only, never a next-action score."""
    import copy
    import torch
    from experiments.modal_jev_omni_sports_train import _general_feature, _load_classifier

    path = Path("/general-data/v2/dev.jsonl")
    rows = [row for row in (json.loads(line) for line in path.open())
            if row["task_family"] == "gui_next_click"]
    torch, classifier, package = _load_classifier()
    trained = copy.deepcopy(classifier.head)
    trained.load_state_dict(torch.load(
        "/general-runs/general-head-v2/head-uniform.pt", map_location="cuda",
        weights_only=True))
    heads = {"pretrained": classifier.head, "general_v2": trained}
    counts = {name: Counter() for name in heads}
    for index, row in enumerate(rows):
        local = dict(row)
        local["state"] = row["extra"]["target_instruction"]
        feature = _general_feature(torch, classifier, package, local).to("cuda").unsqueeze(0)
        labels = list(row["question"]["criteria"])
        target = labels.index(row["gold"]["key"])
        size = torch.tensor([len(labels)], device="cuda")
        with torch.no_grad():
            for name, head in heads.items():
                logits = head(feature, size)[0, :len(labels)]
                counts[name]["correct"] += int(logits.argmax().item() == target)
                counts[name]["rows"] += 1
        if (index + 1) % 20 == 0:
            print(json.dumps({"processed": index + 1, "total": len(rows),
                              "correct": {name: counts[name]["correct"]
                                          for name in counts}}), flush=True)
    report = {"split": "dev", "task": "oracle_local_instruction_grounding",
              "note": "The correct element instruction is provided as input; this is not next-click planning.",
              "results": {name: {"correct": value["correct"], "n": value["rows"],
                                 "accuracy": value["correct"] / value["rows"]}
                          for name, value in counts.items()}}
    destination = Path("/general-runs/general-head-v2/gui-oracle-intent-ablation.json")
    destination.write_text(json.dumps(report, indent=2))
    training.commit()
    return report


@app.function(image=base_image, gpu="H100", timeout=7200,
              volumes={"/general-data": data, "/general-runs": training,
                       "/general-media": general_media,
                       "/model-cache": model_cache})
def benchmark_semantic_dev():
    """Choose named accessible elements; report image and text-only scores."""
    import copy
    import requests
    import torch
    from experiments.modal_jev_omni_sports_train import _general_feature, _load_classifier

    rows = [row for row in (json.loads(line) for line in
            Path("/general-data/v2/dev.jsonl").open())
            if row["task_family"] == "gui_next_click"]
    required = {element_id for row in rows
                for element_id in row["extra"]["option_element_ids"]}
    url = ("https://huggingface.co/datasets/Chengheng/Webintosh/resolve/"
           "7709afcb895aa7a3e39e2aa10406b930ad3956f8/val.jsonl")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    elements = {}
    for line in response.iter_lines():
        if line:
            element = json.loads(line)
            if element["id"] in required:
                elements[element["id"]] = element
    response.close()
    if set(elements) != required:
        raise ValueError("Missing pinned source elements")
    eligible = []
    rejected = Counter()
    for row in rows:
        ids = row["extra"]["option_element_ids"]
        choices = [elements[element_id] for element_id in ids]
        names = [str(item.get("dom_name") or "").strip() for item in choices]
        if any(not name for name in names):
            rejected["unnamed_candidate"] += 1
            continue
        descriptions = [f"{item.get('role') or 'control'}: {name}"
                        for item, name in zip(choices, names)]
        if len(set(descriptions)) != len(descriptions):
            rejected["duplicate_label"] += 1
            continue
        updated = copy.deepcopy(row)
        updated["state"] = row["extra"]["target_instruction"]
        updated["question"]["instructions"] = "Which available control performs this action?"
        updated["question"]["criteria"] = dict(zip("ABCD", descriptions))
        eligible.append(updated)
    if not eligible:
        raise ValueError("No eligible named-element examples")
    torch, classifier, package = _load_classifier()
    trained = copy.deepcopy(classifier.head)
    trained.load_state_dict(torch.load(
        "/general-runs/general-head-v2/head-uniform.pt", map_location="cuda",
        weights_only=True))
    heads = {"pretrained": classifier.head, "general_v2": trained}
    counts = {view: {name: Counter() for name in heads}
              for view in ("image_plus_labels", "labels_only")}
    for index, row in enumerate(eligible):
        labels = list(row["question"]["criteria"])
        target = labels.index(row["gold"]["key"])
        size = torch.tensor([len(labels)], device="cuda")
        for view in counts:
            item = row
            if view == "labels_only":
                item = dict(row, modality="text", media=None)
            feature = _general_feature(torch, classifier, package, item).to("cuda").unsqueeze(0)
            with torch.no_grad():
                for name, head in heads.items():
                    logits = head(feature, size)[0, :len(labels)]
                    counts[view][name]["correct"] += int(logits.argmax().item() == target)
                    counts[view][name]["rows"] += 1
        if (index + 1) % 20 == 0:
            print(json.dumps({"processed": index + 1, "total": len(eligible),
                              "correct": {view: {name: counts[view][name]["correct"]
                                                 for name in heads}
                                          for view in counts}}), flush=True)
    report = {"split": "dev", "task": "oracle_local_intent_named_element_choice",
              "note": "Target-local intent is supplied. This measures element choice, not next-action planning.",
              "source_rows": len(rows), "eligible_rows": len(eligible),
              "rejected": dict(rejected),
              "results": {view: {name: {"correct": value["correct"],
                                         "n": value["rows"],
                                         "accuracy": value["correct"] / value["rows"]}
                                 for name, value in by_head.items()}
                          for view, by_head in counts.items()}}
    destination = Path("/general-runs/general-head-v2/gui-semantic-dev.json")
    destination.write_text(json.dumps(report, indent=2))
    training.commit()
    return report


@app.local_entrypoint()
def main(mode: str = "inspect"):
    if mode == "inspect":
        result = inspect.remote()
    elif mode == "ablate":
        result = ablate_local_intent.remote()
    elif mode == "semantic":
        result = benchmark_semantic_dev.remote()
    else:
        raise ValueError("Choose inspect, ablate, or semantic")
    print(json.dumps(result, indent=2))
