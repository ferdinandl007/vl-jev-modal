"""Inspect and publish a trained Jev-Omni decision head from Modal only."""

import json
import os

import modal


app = modal.App("vl-jev-publish-head")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==1.33.0")
training = modal.Volume.from_name("vl-jev-general-v1-training")
general_data = modal.Volume.from_name("vl-jev-general-v1-data")
release_volume = modal.Volume.from_name("vl-jev-publication-artifacts", create_if_missing=True)
HEAD_REPO = "ferdinandl007/jev-omni-general-head-v2"
BASE_ID = "akhilaaa3/Jev-Omni"
BASE_REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
EXPECTED_SHA256 = "e7771cb40f7cc41b5847e2a4880fb498572c86128c5d20cd0cce4f880211f780"


def _credential_status(secret_name):
    from huggingface_hub import HfApi

    keys = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    present = [key for key in keys if os.environ.get(key)]
    if not present:
        return {"secret": secret_name, "token_present": False}
    try:
        account = HfApi().whoami(token=os.environ[present[0]])
    except Exception as error:
        return {"secret": secret_name, "token_present": True,
                "error_type": type(error).__name__}
    return {"secret": secret_name, "token_present": True,
            "username": account.get("name"), "auth_type": account.get("auth", {}).get("type")}


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=120)
def credential_one():
    return _credential_status("huggingface-secret")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-token")], timeout=120)
def credential_two():
    return _credential_status("huggingface-token")


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              volumes={"/training": training}, timeout=120)
def artifact_status():
    import hashlib
    from pathlib import Path

    root = Path("/training/general-head-v2")
    selection = json.loads((root / "selection.json").read_text())
    pipeline = json.loads((root / "pipeline-result.json").read_text())
    checkpoint = root / f"head-{selection['selected_weighting']}.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {"checkpoint_exists": checkpoint.is_file(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": digest,
            "selection_sha256": selection["checkpoint_sha256"],
            "checkpoint_matches": digest == selection["checkpoint_sha256"],
            "selected_weighting": selection["selected_weighting"],
            "run": pipeline["run"], "dataset_version": pipeline["dataset_version"],
            "test_rows": pipeline["test"]["rows"]}


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              volumes={"/training": training, "/general-data": general_data,
                       "/release": release_volume}, timeout=300)
def prepare_release(source_commit: str):
    import hashlib
    import shutil
    from pathlib import Path

    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("Provide the exact 40-character GitHub source commit")
    root = Path("/training/general-head-v2")
    selection = json.loads((root / "selection.json").read_text())
    result = json.loads((root / "pipeline-result.json").read_text())
    report = json.loads((root / "report-uniform.json").read_text())
    v2 = json.loads(Path("/general-data/v2/summary.json").read_text())
    v1 = json.loads(Path("/general-data/v1/summary.json").read_text())
    if (result["dataset_version"] != "v2" or
            selection["selected_weighting"] != "uniform" or
            report["method"] != "frozen_backbone_head" or
            result["test"]["rows"] != v2["counts"]["test"] or
            report["dataset_split_sha256"] != v2["split_sha256"]):
        raise ValueError("Training artifacts disagree")
    checkpoint = root / "head-uniform.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256 or digest != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint digest mismatch")

    release = Path("/release/jev-omni-general-head-v2")
    release.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("rb") as source, (release / "decision_head.pt").open("wb") as target:
        shutil.copyfileobj(source, target)
    config = {
        "artifact_type": "jev_omni_decision_head_only",
        "base_model": BASE_ID, "base_revision": BASE_REVISION,
        "head_class": "jev_omni._Head256", "feature_dimension": 3840,
        "maximum_options": 256, "checkpoint_file": "decision_head.pt",
        "checkpoint_sha256": digest, "checkpoint_bytes": checkpoint.stat().st_size,
        "training_method": "frozen_backbone_supervised_head",
        "dataset_version": "v2", "selected_weighting": "uniform",
        "training_epochs": report["epochs"], "train_rows": report["train_rows"],
        "dev_rows": report["dev_rows"], "code_commit": source_commit,
    }
    metrics = {
        "train_rows": v2["counts"]["train"],
        "dev_rows": v2["counts"]["dev"],
        "calibration_rows": v2["counts"]["calibration"],
        "test_rows": v2["counts"]["test"],
        "dataset_split_sha256": v2["split_sha256"],
        "inherited_v1_split_sha256": v1["split_sha256"],
        "source_revisions": {
            "v1": v1["source_revisions"],
            "webintosh": v1["webintosh_revision"],
            "human_actions": v2["human_action_source_revision"],
            "ucf101": v2["ucf101_source_revision"],
        },
        "selection_policy": selection["selection_policy"],
        "text_guard_passed": selection["text_guard_passed"],
        "baseline_dev_text": selection["baseline_text"],
        "selected_dev": report["selected_dev"],
        "calibration": result["calibration"]["metrics"],
        "test": result["test"]["metrics"],
    }
    (release / "config.json").write_text(json.dumps(config, indent=2))
    (release / "metrics.json").write_text(json.dumps(metrics, indent=2))
    card = f"""---
base_model: {BASE_ID}
tags:
  - multimodal
  - image
  - video
  - decision-classification
  - research
---

# Jev-Omni general decision head v2

This repository contains **only a trained decision head** for [{BASE_ID}](https://huggingface.co/{BASE_ID}). It does not contain the 12B multimodal backbone or the training dataset. The backbone was frozen; the published `_Head256` head was trained for 3 epochs on 47,680 mixed decision examples. Base model revision: `{BASE_REVISION}`.

## Held-out results

| v2 test subset | Correct / total | Accuracy |
| --- | ---: | ---: |
| All | 7,983 / 9,284 | 85.99% |
| Vision | 5,677 / 6,754 | 84.05% |
| Image | 2,361 / 2,827 | 83.52% |
| Video | 3,214 / 3,782 | 84.98% |
| Text | 2,306 / 2,530 | 91.15% |

The equal-family vision average is **72.96%**; GUI next-click accuracy is **30.36%** (34/112). The pooled vision score does not mean every vision task reaches 83%. These are within-dataset closed-choice results, not JevBench/MVBench scores or a comparison to TypeSafe Jev. Raw option probabilities are not demonstrated to be calibrated for deployment. See `metrics.json` for per-family counts, log loss, Brier score, split hashes, and source revisions.

## Load the head

Run this on a remote CUDA GPU such as Modal. The example pins the upstream model and loads this repository's head in place of its published head:

```python
from pathlib import Path
import sys
import torch
from huggingface_hub import hf_hub_download, snapshot_download

BASE_ID = "{BASE_ID}"
BASE_REVISION = "{BASE_REVISION}"
HEAD_ID = "{HEAD_REPO}"

base_path = snapshot_download(BASE_ID, revision=BASE_REVISION)
source_path = hf_hub_download(BASE_ID, "jev_omni.py", revision=BASE_REVISION)
sys.path.insert(0, str(Path(source_path).parent))
import jev_omni

jev_omni.snapshot_download = lambda *_args, **_kwargs: base_path
classifier = jev_omni.load_jev_omni()
head_path = hf_hub_download(HEAD_ID, "decision_head.pt")
classifier.head.load_state_dict(
    torch.load(head_path, map_location=classifier.device, weights_only=True)
)
classifier.head.eval()

result = classifier.predict(
    state="The match has started.",
    question="Has play begun?",
    options=["Yes", "No"],
)
print(result)
```

The exact training, data-building, and Modal evaluation scripts are at [GitHub commit `{source_commit[:12]}`](https://github.com/ferdinandl007/vl-jev-modal/tree/{source_commit}). The checkpoint is `decision_head.pt` (SHA-256 `{digest}`). Code in that repository is MIT licensed; this does not change the licenses or terms of the upstream model or datasets.

## Data and rights

Training mixed A-OKVQA, ScienceQA, GUI tasks, sports and human-action data, UCF101, and text decisions. Source revisions and split hashes are recorded in `metrics.json`. The data and media are **not redistributed**. Some sources restrict use to research or have unresolved footage/image rights; this checkpoint is published for research evaluation and reproducibility, with no claim of commercial rights to source media or datasets. Review each upstream source before other uses.

## Scope

This is an independent research adaptation of an open model. It is not affiliated with TypeSafe AI, and it does not reproduce TypeSafe's unpublished training method. The frozen backbone means this artifact changes decision scoring, not visual feature extraction.
"""
    (release / "README.md").write_text(card)
    release_volume.commit()
    return {"repo_id": HEAD_REPO, "files": sorted(path.name for path in release.iterdir()),
            "checkpoint_sha256": digest, "checkpoint_bytes": checkpoint.stat().st_size,
            "source_commit": source_commit,
            "test_vision": metrics["test"]["vision"],
            "test_video": metrics["test"]["modality:video"]}


@app.function(image=image, volumes={"/release": release_volume},
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1800)
def publish_release():
    import hashlib
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    keys = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    token = next((os.environ[key] for key in keys if os.environ.get(key)), None)
    if not token:
        raise ValueError("No Hugging Face token in Modal secret")
    folder = Path("/release/jev-omni-general-head-v2")
    config = json.loads((folder / "config.json").read_text())
    if hashlib.sha256((folder / "decision_head.pt").read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Release checkpoint digest mismatch")
    api = HfApi(token=token)
    try:
        existing = api.model_info(HEAD_REPO, token=token)
    except RepositoryNotFoundError:
        existing = None
    if existing is not None:
        sibling_names = {item.rfilename for item in existing.siblings}
        if "decision_head.pt" in sibling_names:
            previous = hf_hub_download(HEAD_REPO, "decision_head.pt", token=token)
            if hashlib.sha256(Path(previous).read_bytes()).hexdigest() != EXPECTED_SHA256:
                raise ValueError("Remote repository contains a different checkpoint")
    else:
        api.create_repo(HEAD_REPO, repo_type="model", private=False, token=token)
    commit = api.upload_folder(folder_path=str(folder), repo_id=HEAD_REPO,
                               repo_type="model", token=token,
                               commit_message="Publish v2 decision head and evaluation metadata")
    public_info = HfApi(token=False).model_info(HEAD_REPO, token=False)
    public_files = {item.rfilename for item in public_info.siblings}
    if not {"decision_head.pt", "README.md", "config.json", "metrics.json"} <= public_files:
        raise ValueError("Public repository missing release files")
    downloaded = hf_hub_download(HEAD_REPO, "decision_head.pt", token=False,
                                 revision=public_info.sha)
    actual = hashlib.sha256(Path(downloaded).read_bytes()).hexdigest()
    if actual != config["checkpoint_sha256"]:
        raise ValueError("Public checkpoint readback hash differs")
    return {"repo_id": HEAD_REPO, "url": f"https://huggingface.co/{HEAD_REPO}",
            "commit": str(commit.commit_url), "revision": public_info.sha,
            "files": sorted(public_files), "checkpoint_sha256": actual,
            "public_readback": True}


@app.function(image=image, timeout=120)
def upstream_api():
    from huggingface_hub import hf_hub_download
    from pathlib import Path

    path = hf_hub_download("akhilaaa3/Jev-Omni", "jev_omni.py",
                           revision="5addda86ddee081a68fb067477ea100c221b8917")
    lines = Path(path).read_text().splitlines()
    selected = []
    for index, line in enumerate(lines):
        if any(needle in line for needle in ("class _Head256", "class JevOmni",
                                           "self.head", "def load_jev_omni", "def predict")):
            selected.append({"line": index + 1, "source": line.strip()})
    return selected[:35]


@app.local_entrypoint()
def main(mode: str = "inspect", source_commit: str = ""):
    if mode == "inspect":
        result = {"credentials": [credential_one.remote(), credential_two.remote()],
                  "artifact": artifact_status.remote()}
    elif mode == "upstream-api":
        result = upstream_api.remote()
    elif mode == "prepare":
        result = prepare_release.remote(source_commit)
    elif mode == "publish":
        result = publish_release.remote()
    else:
        raise ValueError("Supported modes: inspect, upstream-api, prepare, publish")
    print(json.dumps(result, indent=2))
