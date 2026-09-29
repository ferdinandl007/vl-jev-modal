"""Run typed-logit calibration and evaluation entirely inside Modal.

Prediction files are paths inside vl-jev-pilot-data, relative to the volume root.
"""

from pathlib import Path

import modal


app = modal.App("vl-jev-pilot-evaluation")
evaluator = Path(__file__).with_name("evaluate_logits.py")
image = modal.Image.debian_slim(python_version="3.11").add_local_file(str(evaluator), "/root/evaluate_logits.py")
data_volume = modal.Volume.from_name("vl-jev-pilot-data", create_if_missing=True)


@app.function(image=image, volumes={"/dataset": data_volume}, timeout=600)
def evaluate(calibration_file: str, test_file: str, output_file: str, smoke: bool = False):
    import hashlib
    import json
    import os
    import runpy
    import sys
    from pathlib import Path

    root = Path("/dataset").resolve()

    def inside(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("prediction and report paths must stay in the Modal data Volume")
        return path

    calibration_path = inside(calibration_file)
    test_path = inside(test_file)
    output_path = inside(output_file)
    if smoke:
        for split, target in [("calibration", calibration_path), ("test", test_path)]:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as out:
                source = root / "pilot_v1" / f"{split}.jsonl"
                for line in source.read_text().splitlines():
                    row = json.loads(line)
                    question = row["question"]
                    keys = (list(question["criteria"]) if question["type"] == "choice" else
                            ["false", "true"] if question["type"] == "noul" else
                            [str(i) for i in range(len(question["criteria"]))])
                    logits = {key: int(hashlib.sha256((row["id"] + key).encode()).hexdigest()[:4], 16) / 65535
                              for key in keys}
                    out.write(json.dumps({"id": row["id"], "logits": logits}) + "\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    argv = sys.argv
    try:
        os.environ["VL_JEV_MODAL"] = "1"
        sys.argv = ["evaluate_logits.py", "--dataset-dir", str(root / "pilot_v1"),
                    "--calibration-logits", str(calibration_path), "--test-logits", str(test_path),
                    "--output", str(output_path)]
        runpy.run_path("/root/evaluate_logits.py", run_name="__main__")
    finally:
        sys.argv = argv
    report = json.loads(output_path.read_text())
    if smoke:
        for path in [calibration_path, test_path, output_path]:
            path.unlink(missing_ok=True)
    data_volume.commit()
    return {"report": "synthetic smoke result removed" if smoke else str(output_path), "smoke_only": smoke,
            "counts": {kind: {"calibration": data["calibration_n"], "test": data["test_n"]}
                       for kind, data in report.items()}}


@app.local_entrypoint()
def main(calibration_file: str = "predictions/calibration.jsonl",
         test_file: str = "predictions/test.jsonl",
         output_file: str = "reports/calibration.json", smoke: bool = False):
    import json

    print(json.dumps(evaluate.remote(calibration_file, test_file, output_file, smoke), indent=2))
