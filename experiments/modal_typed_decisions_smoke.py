"""Modal-only contract smoke for the Choice/Noul/Score adapter."""

import json
import modal

from experiments.typed_decisions import evaluate_typed

app = modal.App("vl-jev-typed-decisions-smoke")


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=120)
def smoke():
    class Stub:
        def predict(self, *, state, question, options, media, modality):
            if state != '{"scene": "browser"}' or media is not None or modality != "text":
                raise ValueError("Unexpected input")
            weights = [index + 1 for index in range(len(options))]
            total = sum(weights)
            return {"probabilities": {option: weight / total
                                       for option, weight in zip(options, weights)},
                    "confidence": 0.4}

    request = {"state": {"scene": "browser"}, "questions": {
        "continue": {"type": "noul", "instructions": "Continue?"},
        "control": {"type": "choice", "instructions": "Choose a control",
                    "criteria": {"a": "Back", "b": "Submit"}},
        "priority": {"type": "score", "instructions": "Rate priority",
                     "criteria": ["Low", "Medium", "High"]}}}
    result = evaluate_typed(Stub(), request, "stub")
    assert set(result["answers"]) == set(request["questions"])
    assert result["answers"]["control"]["choice"] == "b"
    assert abs(result["answers"]["continue"]["noul"] - 2 / 3) < 1e-9
    assert abs(result["answers"]["priority"]["score"] - 4 / 3) < 1e-9
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(smoke.remote(), indent=2))
