"""One audited head: full general + GUI replay + visual-only SoccerNet.

Selection uses development splits only. Benchmark splits are never mounted.
Prior checkpoints and feature packs remain immutable.
"""
import json
import modal
from experiments.modal_jev_omni_sports_train import (
    base_image, general_data, general_training, model_cache,
    _load_general_features, _general_metrics)
from experiments.modal_general_v3_gui import gui_data, _gui_rows, _load_gui_features
from experiments.modal_soccer_vqa_posttrain import _load_soccer_features, DIRECT_VISUAL_FAMILIES
from experiments.modal_verify_v3_public import REPO, REVISION, SHA256

app = modal.App("vl-jev-general-v4-unified-20260930")
RUN = "general-head-v4-unified-audited"
mounts = {"/general-data": general_data, "/gui-data": gui_data,
          "/general-runs": general_training, "/model-cache": model_cache}


@app.function(image=base_image, gpu="H100", memory=16384, volumes=mounts, timeout=14400)
def train(epochs: int = 3):
    import copy
    import hashlib
    import random
    import sys
    from collections import Counter
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download
    from experiments.modal_jev_omni_sports_train import MODEL_ID, MODEL_REVISION
    if not 1 <= epochs <= 6:
        raise ValueError("Invalid epoch count")
    root = Path("/general-runs") / RUN
    if (root / "report.json").exists():
        return json.loads((root / "report.json").read_text())
    general_root = Path("/general-runs/general-head-v2")
    general_train = _load_general_features(torch, general_root, "train", "v2")
    general_dev = _load_general_features(torch, general_root, "dev", "v2")
    gui_train = _load_gui_features(torch, "train")
    gui_dev_all = _load_gui_features(torch, "dev")
    # Hash protection is independent of task IDs and catches copied screenshots.
    training_hashes = {r["media"]["sha256"] for r in _gui_rows("train") if r.get("media", {}).get("sha256")}
    excluded_ids = {r["id"] for r in _gui_rows("dev") if r.get("media", {}).get("sha256") in training_hashes}
    gui_dev = [r for r in gui_dev_all if r["id"] not in excluded_ids]
    soccer_train = [r for r in _load_soccer_features(torch, "train", 32, scope="all") if r["family"] in DIRECT_VISUAL_FAMILIES]
    soccer_dev = [r for r in _load_soccer_features(torch, "valid", 8, scope="all") if r["family"] in DIRECT_VISUAL_FAMILIES]
    if (len(general_train), len(gui_train), len(soccer_train), len(gui_dev), len(soccer_dev)) != (47680,7962,3358,2105,847):
        raise ValueError("Audited counts changed")
    parent = Path(hf_hub_download(REPO, "decision_head.pt", revision=REVISION, token=False))
    if hashlib.sha256(parent.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Public v3 parent changed")
    source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(source).parent))
    import jev_omni
    head = jev_omni._Head256(3840).cuda()
    head.load_state_dict(torch.load(parent, map_location="cuda", weights_only=True))
    def measure():
        head.eval()
        groups = {"general": _general_metrics(torch, head, general_dev),
                  "gui": _general_metrics(torch, head, gui_dev),
                  "soccer_visual": _general_metrics(torch, head, soccer_dev)}
        def macro(metrics, prefix):
            values = [v["accuracy"] for k,v in metrics.items() if k.startswith(prefix)]
            return sum(values)/len(values)
        groups["balanced_objective"] = (groups["general"]["vision_macro"]["accuracy"] +
                                         macro(groups["gui"], "source:") + macro(groups["soccer_visual"], "family:"))/3
        return groups
    baseline = measure(); best = baseline; best_state = copy.deepcopy(head.state_dict()); best_epoch = 0
    rows = general_train + gui_train + soccer_train
    optimum = torch.optim.AdamW(head.parameters(), lr=5e-5, weight_decay=0.01)
    history = []
    for epoch in range(epochs):
        head.train(); order = list(range(len(rows))); random.Random(3407+epoch).shuffle(order)
        for start in range(0, len(order), 128):
            batch = [rows[i] for i in order[start:start+128]]
            features = torch.stack([r["feature"] for r in batch]).cuda().float()
            counts = torch.tensor([r["count"] for r in batch], device="cuda")
            targets = torch.tensor([r["target"] for r in batch], device="cuda")
            loss = torch.nn.functional.cross_entropy(head(features, counts), targets)
            optimum.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); optimum.step()
        metrics = measure()
        gates = {
            "general_text": metrics["general"]["text"]["accuracy"] >= baseline["general"]["text"]["accuracy"]-0.01,
            "general_vision": metrics["general"]["vision"]["accuracy"] >= baseline["general"]["vision"]["accuracy"]-0.01,
            "gui": metrics["gui"]["overall"]["accuracy"] >= baseline["gui"]["overall"]["accuracy"]-0.01,
            "soccer_improves": metrics["soccer_visual"]["overall"]["accuracy"] > baseline["soccer_visual"]["overall"]["accuracy"],
            "general_log_loss": metrics["general"]["overall"]["log_loss"] <= baseline["general"]["overall"]["log_loss"]+0.03}
        if all(gates.values()) and metrics["balanced_objective"] > best["balanced_objective"]:
            best = metrics; best_epoch = epoch+1; best_state = copy.deepcopy(head.state_dict())
        history.append({"epoch":epoch+1,"gates":gates,"metrics":metrics})
        print(json.dumps({"epoch":epoch+1,"gates":gates,"objective":metrics["balanced_objective"],
                          "soccer":metrics["soccer_visual"]["overall"]}),flush=True)
    root.mkdir(parents=True,exist_ok=True); checkpoint = root/"head.pt"
    if best_epoch:
        torch.save(best_state,checkpoint)
    else:
        import shutil
        shutil.copyfile(parent,checkpoint)
    head.load_state_dict(best_state)
    # Tests are first read after the epoch and fallback decision is locked.
    general_test = _load_general_features(torch,general_root,"test","v2")
    soccer_test = _load_soccer_features(torch,"test",8,scope="pilot")
    selected_test = {"general":_general_metrics(torch,head,general_test),
                     "soccer_public_usable":_general_metrics(torch,head,soccer_test)}
    head.load_state_dict(torch.load(parent,map_location="cuda",weights_only=True))
    baseline_test = {"general":_general_metrics(torch,head,general_test),
                     "soccer_public_usable":_general_metrics(torch,head,soccer_test)}
    report = {"run":RUN,"parent_repo":REPO,"parent_revision":REVISION,"parent_sha256":SHA256,
              "checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "method":"frozen_backbone_one_head_full_general_gui_visual_soccer_replay",
              "train_rows":len(rows),"train_by_source":dict(Counter(r["source_repo"] for r in rows)),
              "gui_dev_excluded_shared_screenshot_ids":sorted(excluded_ids),
              "dev_counts":{"general":len(general_dev),"gui":len(gui_dev),"soccer_visual":len(soccer_dev)},
              "selected_epoch":best_epoch,"status":"guarded_development_improvement" if best_epoch else "fallback_to_v3",
              "baseline_dev":baseline,"selected_dev":best,"history":history,
              "baseline_test":baseline_test,"selected_test":selected_test,
              "test_used_for_selection":False,"benchmark_data_used_for_training_or_selection":False,
              "selection_policy":"maximize equal-weight general visual family macro, GUI source macro and soccer family macro, with 1pp retention gates, improved visual soccer accuracy and general log loss <= parent +0.03"}
    (root/"report.json").write_text(json.dumps(report,indent=2)); general_training.commit()
    return {k:report[k] for k in ("run","train_rows","selected_epoch","status","checkpoint_sha256","dev_counts","baseline_test","selected_test")}


@app.local_entrypoint()
def main(epochs: int = 3):
    print(json.dumps(train.remote(epochs),indent=2))
