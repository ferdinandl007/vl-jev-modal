"""Re-evaluate the public head on held-out Modal feature packs."""

import hashlib
import json

import modal

from experiments.modal_jev_omni_sports_train import (
    MODEL_ID, MODEL_REVISION, _general_metrics, _load_classifier,
    _load_general_features, base_image, general_data, general_training, model_cache,
)


app = modal.App("vl-jev-verify-public-head")
HEAD_ID = "ferdinandl007/jev-omni-general-head-v2"
EXPECTED_SHA256 = "e7771cb40f7cc41b5847e2a4880fb498572c86128c5d20cd0cce4f880211f780"


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/general-data": general_data, "/general-runs": general_training})
def verify(split: str = "test", revision: str = "main"):
    from pathlib import Path
    import sys
    import torch
    from huggingface_hub import hf_hub_download

    if split not in {"calibration", "test"}:
        raise ValueError("Choose calibration or test")
    root = Path("/general-runs/general-head-v2")
    published = hf_hub_download(HEAD_ID, "decision_head.pt", revision=revision)
    digest = hashlib.sha256(Path(published).read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Published checkpoint hash differs from trained checkpoint")
    upstream = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(upstream).parent))
    import jev_omni

    items = _load_general_features(torch, root, split, "v2")
    head = jev_omni._Head256(items[0]["feature"].numel()).to("cuda")
    head.load_state_dict(torch.load(published, map_location="cuda", weights_only=True))
    observed = _general_metrics(torch, head, items)
    expected = json.loads((root / f"evaluation-{split}-uniform.json").read_text())["metrics"]
    for group in ("overall", "vision", "text", "modality:image", "modality:video"):
        if group not in observed or group not in expected:
            if group in ("modality:image", "modality:video") and group not in expected:
                continue
            raise ValueError(f"Missing metric group: {group}")
        for key in ("n", "correct"):
            if observed[group][key] != expected[group][key]:
                raise ValueError(f"Published head differs on {split} {group} {key}")
        for key in ("accuracy", "log_loss", "brier"):
            if abs(observed[group][key] - expected[group][key]) > 1e-6:
                raise ValueError(f"Published head differs on {split} {group} {key}")
    return {"repo_id": HEAD_ID, "revision": revision, "split": split,
            "checkpoint_sha256": digest, "rows": len(items),
            "matched_saved_evaluation": True,
            "overall": observed["overall"], "vision": observed["vision"],
            "video": observed.get("modality:video"), "text": observed["text"]}


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/model-cache": model_cache})
def smoke_predict(revision: str = "main"):
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download

    torch, classifier, _package = _load_classifier()
    checkpoint = hf_hub_download(HEAD_ID, "decision_head.pt", revision=revision)
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Published checkpoint hash mismatch")
    classifier.head.load_state_dict(
        torch.load(checkpoint, map_location=classifier.device, weights_only=True))
    classifier.head.eval()
    answer = classifier.predict(state="The match has started.",
                                question="Has play begun?", options=["Yes", "No"])
    if not isinstance(answer, dict):
        raise ValueError("Published head did not produce a decision mapping")
    return {"repo_id": HEAD_ID, "revision": revision,
            "checkpoint_sha256": digest, "answer": answer}


@app.local_entrypoint()
def main(split: str = "test", revision: str = "main", mode: str = "verify"):
    if mode == "verify":
        result = verify.remote(split, revision)
    elif mode == "smoke":
        result = smoke_predict.remote(revision)
    else:
        raise ValueError("Choose verify or smoke")
    print(json.dumps(result, indent=2))
