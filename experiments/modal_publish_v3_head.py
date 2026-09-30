"""Package and publish the validated general v3 decision head from Modal."""

import json
import os
import modal

app = modal.App("vl-jev-publish-general-v3-head")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==1.33.0")
training = modal.Volume.from_name("vl-jev-general-v1-training")
release_volume = modal.Volume.from_name("vl-jev-publication-artifacts", create_if_missing=True)
HEAD_REPO = "ferdinandl007/jev-omni-general-head-v3"
BASE_ID = "akhilaaa3/Jev-Omni"
BASE_REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
RUN = "general-head-v3-gui"


@app.function(image=image, volumes={"/training": training,
                                    "/release": release_volume}, timeout=300)
def prepare_release(source_commit: str):
    import hashlib
    import shutil
    from pathlib import Path

    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("Provide an exact GitHub source commit")
    root = Path("/training") / RUN
    report = json.loads((root / "report.json").read_text())
    gui_test = json.loads((root / "official-test-task.json").read_text())
    retention = json.loads((root / "retention-test.json").read_text())
    checkpoint = Path(report["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (digest != report["checkpoint_sha256"] or
            report["selected_epoch"] < 1 or
            report["train_rows"] != 55_642 or
            report["general_replay_rows"] != 47_680 or
            report["gui_train_rows"] != 7_962 or
            gui_test["v3_checkpoint_sha256"] != digest or
            retention["v3_checkpoint_sha256"] != digest or
            gui_test["n"] != 1_086 or retention["n"] != 9_284):
        raise ValueError("Training and held-out artifacts disagree")
    release = Path("/release/jev-omni-general-head-v3")
    release.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, release / "decision_head.pt")
    config = {"artifact_type": "jev_omni_decision_head_only",
              "base_model": BASE_ID, "base_revision": BASE_REVISION,
              "head_class": "jev_omni._Head256", "feature_dimension": 3840,
              "maximum_options": 256, "checkpoint_file": "decision_head.pt",
              "checkpoint_sha256": digest, "checkpoint_bytes": checkpoint.stat().st_size,
              "training_method": "frozen_backbone_general_head_with_full_v2_replay",
              "parent_checkpoint_sha256": report["parent_checkpoint_sha256"],
              "selected_epoch": report["selected_epoch"],
              "train_rows": report["train_rows"],
              "general_replay_rows": report["general_replay_rows"],
              "gui_train_rows": report["gui_train_rows"], "code_commit": source_commit}
    metrics = {"train_by_source": report["train_by_source"],
               "selection_policy": report["selection_policy"],
               "baseline_dev": report["baseline_dev"],
               "selected_dev": report["selected_dev"],
               "official_test_task": gui_test,
               "general_retention_test": retention}
    (release / "config.json").write_text(json.dumps(config, indent=2))
    (release / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def accuracy(payload, key="overall"):
        value = payload[key]
        return f"{value['correct']:,}/{value['n']:,} ({value['accuracy']:.2%})"

    card = f"""---
base_model: {BASE_ID}
tags:
  - multimodal
  - image
  - video
  - decision-classification
  - research
---

# Jev-Omni general decision head v3

This repository contains a 256-slot decision head for [{BASE_ID}](https://huggingface.co/{BASE_ID}), not the 12B backbone. The backbone stayed frozen. The v2 general head was retrained with full replay of 47,680 mixed text, image, and video decisions plus 7,962 labeled-control computer-use decisions, totaling **55,642 training examples**. The new data came from [Multimodal Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) and [BrowserGym MiniWoB action-only trajectories](https://huggingface.co/datasets/saital/browser-agent-phase1-sft-action-only). Source media and datasets are not redistributed.

## Held-out evaluation

| Derived task / split | v2 head | v3 head |
| --- | ---: | ---: |
| Mind2Web official test-task source split, four named candidates | {accuracy(gui_test['baseline_v2'])} | {accuracy(gui_test['selected_v3'])} |
| Existing general test, all tasks | {accuracy(retention['baseline_v2'])} | {accuracy(retention['selected_v3'])} |
| Existing general test, vision | {accuracy(retention['baseline_v2'], 'vision')} | {accuracy(retention['selected_v3'], 'vision')} |
| Existing general test, text | {accuracy(retention['baseline_v2'], 'text')} | {accuracy(retention['selected_v3'], 'text')} |

The Mind2Web set uses the official test-task **source split**, but the four-candidate marked-screenshot task is derived and is not the published Mind2Web leaderboard metric. Its target is guaranteed among four preselected named controls. It does not measure full-page retrieval or completed browser tasks. The MiniWoB supervision is teacher generated and only retained click rows. See `metrics.json` for per-operation, per-family, log-loss, and checkpoint details. Raw probabilities and confidence are not demonstrated to be calibrated for deployment.

## Load

Run inference on a remote CUDA GPU such as Modal:

```python
from pathlib import Path
import sys
import torch
from huggingface_hub import hf_hub_download, snapshot_download

base = snapshot_download("{BASE_ID}", revision="{BASE_REVISION}")
source = hf_hub_download("{BASE_ID}", "jev_omni.py", revision="{BASE_REVISION}")
sys.path.insert(0, str(Path(source).parent))
import jev_omni
jev_omni.snapshot_download = lambda *_args, **_kwargs: base
classifier = jev_omni.load_jev_omni()
head = hf_hub_download("{HEAD_REPO}", "decision_head.pt")
classifier.head.load_state_dict(torch.load(head, map_location=classifier.device, weights_only=True))
classifier.head.eval()
print(classifier.predict(state="A meeting starts at 10:00; it is 09:00.",
                         question="Has it started?", options=["No", "Yes"]))
```

The [training and evaluation scripts at GitHub commit `{source_commit[:12]}`](https://github.com/ferdinandl007/vl-jev-modal/tree/{source_commit}) include a Python adapter for Choice, Noul, and Score response types. This model is independent of TypeSafe AI's Jev service. Code is MIT licensed; upstream model and dataset rights are separate. The Mind2Web source is research-only under OpenRAIL; inspect each upstream source before other uses.
"""
    (release / "README.md").write_text(card)
    release_volume.commit()
    return {"repo_id": HEAD_REPO, "checkpoint_sha256": digest,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "files": sorted(path.name for path in release.iterdir()),
            "source_commit": source_commit,
            "official_gui_test": {key: gui_test[key]["overall"]
                                  for key in ("baseline_v2", "selected_v3")}}


@app.function(image=image, volumes={"/release": release_volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1800)
def publish_release():
    import hashlib
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    token = next((os.environ[key] for key in
                  ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
                  if os.environ.get(key)), None)
    if not token:
        raise ValueError("No Hugging Face token in Modal secret")
    folder = Path("/release/jev-omni-general-head-v3")
    config = json.loads((folder / "config.json").read_text())
    digest = hashlib.sha256((folder / "decision_head.pt").read_bytes()).hexdigest()
    if digest != config["checkpoint_sha256"]:
        raise ValueError("Release checkpoint digest mismatch")
    api = HfApi(token=token)
    try:
        existing = api.model_info(HEAD_REPO, token=token)
    except RepositoryNotFoundError:
        existing = None
    if existing is not None and "decision_head.pt" in {item.rfilename for item in existing.siblings}:
        previous = hf_hub_download(HEAD_REPO, "decision_head.pt", token=token)
        if hashlib.sha256(Path(previous).read_bytes()).hexdigest() != digest:
            raise ValueError("Existing remote repository contains a different head")
    if existing is None:
        api.create_repo(HEAD_REPO, repo_type="model", private=False, token=token)
    commit = api.upload_folder(folder_path=str(folder), repo_id=HEAD_REPO,
                               repo_type="model", token=token,
                               commit_message="Publish v3 general decision head and held-out metrics")
    public = HfApi(token=False).model_info(HEAD_REPO, token=False)
    files = {item.rfilename for item in public.siblings}
    if not {"decision_head.pt", "README.md", "config.json", "metrics.json"} <= files:
        raise ValueError("Public release files incomplete")
    anonymous = hf_hub_download(HEAD_REPO, "decision_head.pt", token=False,
                                revision=public.sha)
    if hashlib.sha256(Path(anonymous).read_bytes()).hexdigest() != digest:
        raise ValueError("Anonymous checkpoint readback mismatch")
    return {"url": f"https://huggingface.co/{HEAD_REPO}",
            "revision": public.sha, "commit": str(commit.commit_url),
            "checkpoint_sha256": digest, "anonymous_readback": True}


@app.local_entrypoint()
def main(mode: str = "prepare", source_commit: str = ""):
    if mode == "prepare":
        result = prepare_release.remote(source_commit)
    elif mode == "publish":
        result = publish_release.remote()
    else:
        raise ValueError("Mode must be prepare or publish")
    print(json.dumps(result, indent=2))
