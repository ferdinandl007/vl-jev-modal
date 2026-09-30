"""Load the anonymous public v3 checkpoint and exercise all three types on Modal."""

import json
import modal

from experiments.modal_jev_omni_sports_train import _load_classifier, base_image, model_cache
from experiments.typed_decisions import evaluate_typed

app = modal.App("vl-jev-verify-public-v3-head")
REPO = "ferdinandl007/jev-omni-general-head-v3"
REVISION = "10e061b811f4add247dc0cbf06a67cb5c0233ad8"
SHA256 = "f1f2b4ba26776198bf0cf716f38159419fb2ba35a881943237de11624182f3e9"


@app.function(image=base_image, gpu="H100", timeout=1800,
              volumes={"/model-cache": model_cache})
def verify():
    import hashlib
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "decision_head.pt", revision=REVISION, token=False)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError("Public v3 head checksum mismatch")
    _backend, classifier, _package = _load_classifier()
    classifier.head.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    classifier.head.eval()
    request = {"state": "The meeting starts at 10:00. It is now 09:00.",
               "questions": {
                   "started": {"type": "noul", "instructions": "Has it started?"},
                   "phase": {"type": "choice", "instructions": "Choose the phase",
                             "criteria": {"before": "Before", "during": "During"}},
                   "progress": {"type": "score", "instructions": "Rate progress",
                                "criteria": ["Not started", "In progress", "Finished"]}}}
    answer = evaluate_typed(classifier, request, REPO)
    if set(answer["answers"]) != set(request["questions"]):
        raise ValueError("Missing typed answers")
    return {"repo": REPO, "revision": REVISION, "checkpoint_sha256": digest,
            "all_typed_answers_present": True,
            "answer_types": {name: result["type"] for name, result in answer["answers"].items()}}


@app.local_entrypoint()
def main():
    print(json.dumps(verify.remote(), indent=2))
