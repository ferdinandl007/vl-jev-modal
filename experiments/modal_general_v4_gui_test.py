"""Held-out GUI comparison, with exact train screenshot overlap exclusions."""
import json
import modal
from experiments.modal_general_v3_gui_test import test_data,test_media,gui_data
from experiments.modal_jev_omni_sports_train import base_image,general_training,model_cache,_load_classifier,_general_feature,_options,_decision_hash,_general_metrics
from experiments.modal_general_v4_unified import RUN
from experiments.modal_verify_v3_public import SHA256

app=modal.App("vl-jev-v4-audited-gui-test-20260930")


@app.function(image=base_image,gpu="H100",timeout=14400,volumes={"/gui-test-data":test_data,"/gui-test-media":test_media,"/gui-data":gui_data,"/general-runs":general_training,"/model-cache":model_cache})
def evaluate():
    import copy,hashlib
    from pathlib import Path
    import torch
    root=Path("/general-runs")/RUN
    report=json.loads((root/"report.json").read_text())
    if (root/"gui-test.json").exists():
        return json.loads((root/"gui-test.json").read_text())
    test_root=Path("/gui-test-data/mind2web_test_task_v1")
    manifest=test_root/"test_task.jsonl"
    rows=[json.loads(line) for line in manifest.open()]
    summary=json.loads((test_root/"summary.json").read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=summary["split_sha256"]:
        raise ValueError("Test manifest changed")
    train=[json.loads(line) for line in Path("/gui-data/mind2web_general_v1/train.jsonl").open()]
    if {r["source"]["group"] for r in rows}&{r["source"]["group"] for r in train}:
        raise ValueError("Test task overlap")
    train_hashes={r["media"]["sha256"] for r in train}
    excluded={r["id"] for r in rows if r["media"]["sha256"] in train_hashes}
    _,classifier,package=_load_classifier()
    v3=copy.deepcopy(classifier.head); parent=Path("/general-runs/general-head-v3-gui/head-uniform.pt")
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=SHA256:
        raise ValueError("Parent changed")
    v3.load_state_dict(torch.load(parent,map_location="cuda",weights_only=True))
    v4=copy.deepcopy(classifier.head); checkpoint=Path(report["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report["checkpoint_sha256"]:
        raise ValueError("Selected head changed")
    v4.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True))
    feature_root=root/"gui-test-features"; feature_root.mkdir(exist_ok=True)
    items=[]
    for offset in range(0,len(rows),32):
        batch=rows[offset:offset+32]; destination=feature_root/f"pack-{offset//32:04d}.pt"
        if destination.exists():
            saved=torch.load(destination,map_location="cpu",weights_only=True)
            if [r["decision_hash"] for r in saved]!=[_decision_hash(r) for r in batch]:
                raise ValueError("Stale GUI test features")
        else:
            saved=[]
            for row in batch:
                labels,_=_options(row)
                feature=_general_feature(torch,classifier,package,row)
                saved.append({"id":row["id"],"feature":feature.half(),"decision_hash":_decision_hash(row),
                              "target":labels.index(row["gold"]["key"]),"count":len(labels),
                              "modality":row["modality"],"family":row["task_family"],"source_repo":row["source"]["repo"]})
            torch.save(saved,destination); general_training.commit()
        items.extend(saved)
        print(f"audited GUI test features: {len(items)}/{len(rows)}",flush=True)
    clean=[r for r in items if r["id"] not in excluded]
    output={"n_original":len(items),"n_clean":len(clean),"excluded_shared_screenshot_ids":sorted(excluded),
            "checkpoint_sha256":report["checkpoint_sha256"],"parent_sha256":SHA256,
            "full_source_split":{"v3":_general_metrics(torch,v3,items),"v4":_general_metrics(torch,v4,items)},
            "clean_screenshot_subset":{"v3":_general_metrics(torch,v3,clean),"v4":_general_metrics(torch,v4,clean)},
            "note":"Derived four-candidate marked-screenshot decisions; not official Mind2Web task-success metric. Excludes exact training screenshot reuse from primary diagnostic."}
    (root/"gui-test.json").write_text(json.dumps(output,indent=2));general_training.commit()
    return {"n_original":len(items),"n_clean":len(clean),"scores":{k:v["overall"] for k,v in output["clean_screenshot_subset"].items()}}


@app.local_entrypoint()
def main():
    print(json.dumps(evaluate.remote(),indent=2))
