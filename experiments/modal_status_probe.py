"""Read small training status artifacts inside Modal without downloading model data."""

import json
from pathlib import Path

import modal


app = modal.App("vl-jev-training-status-probe")
training = modal.Volume.from_name("vl-jev-general-v1-training")
dataset = modal.Volume.from_name("vl-jev-general-v1-data")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/training": training, "/dataset": dataset},
    timeout=300,
)
def inspect():
    output = {}
    for version in ("v1", "v2"):
        root = Path("/training") / f"general-head-{version}"
        result_file = root / "pipeline-result.json"
        result = json.loads(result_file.read_text()) if result_file.is_file() else None
        summary_file = Path("/dataset") / version / "summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.is_file() else None
        output[version] = {
            "dataset_counts": summary.get("counts") if summary else None,
            "trained": result is not None,
        }
        if result:
            selection = result["selection"]
            output[version].update({
                "selected_weighting": result["selected_weighting"],
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "text_guard_passed": selection["text_guard_passed"],
                "baseline_dev_text": selection["baseline_text"],
                "selected_dev": {k: v for k, v in selection["dev"][result["selected_weighting"]].items()
                                 if k in ("overall", "vision", "vision_macro", "text", "modality:image", "modality:video", "family:gui_next_click")},
                "test": {k: v for k, v in result["test"]["metrics"].items()
                         if k in ("overall", "vision", "vision_macro", "text", "modality:image", "modality:video", "family:gui_next_click")},
                "calibration": {k: v for k, v in result["calibration"]["metrics"].items()
                                if k in ("overall", "vision", "vision_macro", "text")},
            })
    return output


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2))
