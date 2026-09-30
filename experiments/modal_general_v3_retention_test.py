"""Compare general v2 and selected v3 on the unchanged v2 held-out test mix."""

import json
import modal

from experiments.modal_jev_omni_sports_train import (
    MODEL_ID, MODEL_REVISION, _general_metrics, _load_general_features,
    base_image, general_data, general_training, model_cache,
)

app = modal.App("vl-jev-general-v3-retention-test")


@app.function(image=base_image, gpu="H100", timeout=7200,
              volumes={"/general-data": general_data,
                       "/general-runs": general_training,
                       "/model-cache": model_cache})
def evaluate():
    import hashlib
    import sys
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download

    root = Path("/general-runs")
    v2 = root / "general-head-v2" / "head-uniform.pt"
    report = json.loads((root / "general-head-v3-gui" / "report.json").read_text())
    v3 = Path(report["checkpoint"])
    if hashlib.sha256(v3.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("V3 checkpoint changed")
    items = _load_general_features(torch, root / "general-head-v2", "test", "v2")
    source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(source).parent))
    import jev_omni
    heads = {}
    for name, path in (("baseline_v2", v2), ("selected_v3", v3)):
        head = jev_omni._Head256(items[0]["feature"].numel()).to("cuda")
        head.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
        heads[name] = _general_metrics(torch, head, items)
    result = {"split": "general_v2_test", "n": len(items),
              "v3_selected_epoch": report["selected_epoch"],
              "v2_checkpoint_sha256": hashlib.sha256(v2.read_bytes()).hexdigest(),
              "v3_checkpoint_sha256": report["checkpoint_sha256"], **heads}
    destination = root / "general-head-v3-gui" / "retention-test.json"
    if destination.is_file() and json.loads(destination.read_text()) != result:
        raise ValueError("Immutable retention test result changed")
    destination.write_text(json.dumps(result, indent=2))
    general_training.commit()
    return result


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              timeout=86400, volumes={"/general-runs": general_training})
def wait_and_evaluate():
    import time
    from pathlib import Path

    deadline = time.monotonic() + 20 * 3600
    while time.monotonic() < deadline:
        general_training.reload()
        if Path("/general-runs/general-head-v3-gui/report.json").is_file():
            return evaluate.remote()
        print(json.dumps({"phase": "waiting_for_v3_report"}), flush=True)
        time.sleep(60)
    raise TimeoutError("V3 report did not finish")


@app.local_entrypoint()
def main():
    print(json.dumps(wait_and_evaluate.remote(), indent=2))
