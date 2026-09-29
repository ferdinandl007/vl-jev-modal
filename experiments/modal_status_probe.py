"""Read small training status artifacts inside Modal without downloading model data."""

import json
from pathlib import Path

import modal


app = modal.App("vl-jev-training-status-probe")
training = modal.Volume.from_name("vl-jev-general-v1-training")
dataset = modal.Volume.from_name("vl-jev-general-v1-data")
ucf_data = modal.Volume.from_name("vl-jev-ucf101-full-data")
ucf_media = modal.Volume.from_name("vl-jev-ucf101-full-media")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/training": training, "/dataset": dataset,
             "/ucf-data": ucf_data, "/ucf-media": ucf_media},
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
    missing = []
    manifest = Path("/ucf-data/ucf101_full_v1/calibration.jsonl")
    if manifest.is_file():
        for line in manifest.open():
            row = json.loads(line)
            folder = Path("/ucf-media/calibration") / row["id"]
            if not (folder / "row.json").is_file() or not (folder / "media.avi").is_file():
                missing.append({"id": row["id"], "source_path": row["media"]["path"],
                                "folder_exists": folder.is_dir(),
                                "record_exists": (folder / "row.json").is_file(),
                                "video_exists": (folder / "media.avi").is_file()})
    output["missing_ucf_calibration"] = missing
    return output


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2))
