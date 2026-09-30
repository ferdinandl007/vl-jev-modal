"""Run all three typed answer kinds against the published v2 head on Modal."""

import json
import modal

from experiments.modal_jev_omni_sports_train import (
    _load_classifier, base_image, general_training, model_cache,
)
from experiments.typed_decisions import evaluate_typed

app = modal.App("vl-jev-typed-decisions-real-smoke")


@app.function(image=base_image, gpu="H100", timeout=1800,
              volumes={"/model-cache": model_cache,
                       "/general-runs": general_training})
def smoke():
    import torch

    _torch, classifier, _package = _load_classifier()
    classifier.head.load_state_dict(torch.load(
        "/general-runs/general-head-v2/head-uniform.pt",
        map_location="cuda", weights_only=True))
    classifier.head.eval()
    request = {"state": "The meeting begins at 10:00. It is now 09:00.",
               "questions": {
                   "started": {"type": "noul", "instructions": "Has it started?"},
                   "phase": {"type": "choice", "instructions": "What is the meeting phase?",
                             "criteria": {"before": "Before start", "during": "In progress"}},
                   "progress": {"type": "score", "instructions": "Rate the meeting progress",
                                "criteria": ["Not started", "In progress", "Finished"]}}}
    result = evaluate_typed(classifier, request,
                            "ferdinandl007/jev-omni-general-head-v2")
    if set(result["answers"]) != set(request["questions"]):
        raise ValueError("Incomplete typed answers")
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(smoke.remote(), indent=2))
