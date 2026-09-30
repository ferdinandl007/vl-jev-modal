"""Small read-only Modal status report for the general v3 GUI mixture."""

import json
import modal

app = modal.App("vl-jev-general-v3-status")
gui_data = modal.Volume.from_name("vl-jev-gui-general-data")
test_data = modal.Volume.from_name("vl-jev-gui-test-task-data")
general_training = modal.Volume.from_name("vl-jev-general-v1-training")


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=120,
              volumes={"/gui-data": gui_data, "/gui-test-data": test_data,
                       "/general-runs": general_training})
def status():
    from pathlib import Path

    def compact(metrics):
        return {key: value for key, value in metrics.items()
                if key in {"overall", "text", "vision", "vision_macro"} or
                key.startswith("family:gui_")}

    root = Path("/general-runs/general-head-v3-gui")
    sources = {}
    for name, folder in (("mind2web", "mind2web_general_v1"),
                         ("ax", "ax_actions_general_v1")):
        path = Path("/gui-data") / folder / "summary.json"
        sources[name] = json.loads(path.read_text())["counts"] if path.is_file() else None
    test_path = Path("/gui-test-data/mind2web_test_task_v1/summary.json")
    report_path = root / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    gui_test = root / "official-test-task.json"
    retention = root / "retention-test.json"
    return {"sources": sources,
            "official_test_source": json.loads(test_path.read_text())["counts"]
            if test_path.is_file() else None,
            "feature_packs": {split: len(list((root / "features" / split).glob("pack-gui-*.pt")))
                              for split in ("train", "dev")},
            "training": {**{key: report[key] for key in
                         ("selected_epoch", "train_rows", "gui_train_rows",
                          "checkpoint_sha256")},
                         "baseline_dev": {kind: compact(report["baseline_dev"][kind])
                                          if kind in {"general", "gui"} else
                                          report["baseline_dev"][kind]
                                          for kind in ("general", "gui", "gui_source_macro")},
                         "selected_dev": {kind: compact(report["selected_dev"][kind])
                                          if kind in {"general", "gui"} else
                                          report["selected_dev"][kind]
                                          for kind in ("general", "gui", "gui_source_macro")}}
            if report else None,
            "official_gui_test": {key: (compact(json.loads(gui_test.read_text())[key])
                                         if key in {"baseline_v2", "selected_v3"}
                                         else json.loads(gui_test.read_text())[key])
                                  for key in ("n", "v3_selected_epoch",
                                              "baseline_v2", "selected_v3")}
            if gui_test.is_file() else None,
            "general_retention_test": {key: (compact(json.loads(retention.read_text())[key])
                                            if key in {"baseline_v2", "selected_v3"}
                                            else json.loads(retention.read_text())[key])
                                       for key in ("n", "v3_selected_epoch",
                                                   "baseline_v2", "selected_v3")}
            if retention.is_file() else None}


@app.local_entrypoint()
def main():
    print(json.dumps(status.remote(), indent=2))
