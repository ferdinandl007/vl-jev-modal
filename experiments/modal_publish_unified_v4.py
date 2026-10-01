"""Consolidated 12B + one-head release, prepared and verified entirely on Modal."""
import json
import modal
from experiments.modal_jev_omni_sports_train import base_image,model_cache,general_training,MODEL_ID,MODEL_REVISION
from experiments.modal_general_v4_unified import RUN

app=modal.App("vl-jev-consolidated-release-v4-20260930")
REPO="ferdinandl007/jev-omni-unified-v4"
release=modal.Volume.from_name("vl-jev-publication-artifacts",create_if_missing=True)
bench=modal.Volume.from_name("vl-jev-official-benchmark-results")
audit=modal.Volume.from_name("vl-jev-benchmark-audit")
image=modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub==1.33.0").env({"HF_HOME":"/model-cache"}).add_local_python_source("experiments")
mounts={"/release":release,"/model-cache":model_cache,"/general-runs":general_training,"/bench-results":bench,"/audit":audit}
text_data=modal.Volume.from_name("vl-jev-text-decisions",create_if_missing=True)


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):
            h.update(part)
    return h.hexdigest()


@app.function(image=image,volumes=mounts,timeout=7200,memory=4096)
def prepare():
    import shutil
    from pathlib import Path
    from huggingface_hub import snapshot_download
    training=json.loads((Path("/general-runs")/RUN/"report.json").read_text())
    head=Path(training["checkpoint"])
    if sha(head)!=training["checkpoint_sha256"] or training["train_rows"]!=59000 or training["selected_epoch"]<1:
        raise ValueError("Unified head did not pass development selection")
    folder=Path("/release")/"jev-omni-unified-v4"; folder.mkdir(parents=True,exist_ok=True)
    base=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,token=False,allow_patterns=[
        "config.json","generation_config.json","model*.safetensors*","processor_config.json",
        "tokenizer.json","tokenizer_config.json","chat_template.jinja","decision_config.json","jev_omni.py","LICENSE*","NOTICE*"]))
    for source in base.iterdir():
        if source.is_file():
            dest=folder/source.name
            if not dest.exists() or dest.stat().st_size!=source.stat().st_size:
                shutil.copyfile(source,dest)
    shutil.copyfile(head,folder/"head.pt")
    shutil.copyfile("/root/experiments/typed_decisions.py",folder/"typed_decisions.py")
    decision=json.loads((folder/"decision_config.json").read_text())
    decision["post_training"]={"run":RUN,"method":training["method"],"train_rows":59000,
                               "parent_head_sha256":training["parent_sha256"],"head_sha256":training["checkpoint_sha256"],"selected_epoch":training["selected_epoch"]}
    (folder/"decision_config.json").write_text(json.dumps(decision,indent=2))
    wrapper='''"""Load a complete pinned repository, then answer Choice/Noul/Score."""
import importlib.util
from pathlib import Path
from huggingface_hub import snapshot_download

def load_unified_jev(repo_id="'''+REPO+'''", *, revision=None, device="cuda", token=False):
    path=Path(snapshot_download(repo_id,revision=revision,token=token))
    spec=importlib.util.spec_from_file_location("consolidated_jev_omni",path/"jev_omni.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.snapshot_download=lambda *args,**kwargs:str(path)
    classifier=module.load_jev_omni(model_id=repo_id,device=device)
    typed_spec=importlib.util.spec_from_file_location("consolidated_typed_decisions",path/"typed_decisions.py")
    typed=importlib.util.module_from_spec(typed_spec);typed_spec.loader.exec_module(typed)
    classifier.decide_typed=lambda request,**kwargs:typed.evaluate_typed(classifier,request,repo_id,**kwargs)
    return classifier
'''
    (folder/"unified_jev.py").write_text(wrapper)
    lock={"repo":REPO,"base_repo":MODEL_ID,"base_revision":MODEL_REVISION,
          "head_sha256":training["checkpoint_sha256"],"train_rows":59000,
          "files":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in folder.iterdir() if p.is_file() and p.name not in {"release.lock.json","README.md","metrics.json","dataset_audit.json"}}}
    (folder/"release.lock.json").write_text(json.dumps(lock,indent=2));release.commit();model_cache.commit()
    return {"prepared_repo":REPO,"head_sha256":training["checkpoint_sha256"],"weight_bytes":sum(v["bytes"] for k,v in lock["files"].items() if k.endswith(".safetensors")),"files":len(lock["files"])}


@app.function(image=image,volumes=mounts,secrets=[modal.Secret.from_name("huggingface-secret")],timeout=10800,memory=4096)
def publish():
    import os
    from pathlib import Path
    from huggingface_hub import HfApi,hf_hub_download
    folder=Path("/release/jev-omni-unified-v4")
    lock=json.loads((folder/"release.lock.json").read_text())
    training=json.loads((Path("/general-runs")/RUN/"report.json").read_text())
    benchmarks=json.loads(Path("/bench-results/aggregate.json").read_text())
    gui=json.loads((Path("/general-runs")/RUN/"gui-test.json").read_text())
    overlap=json.loads(Path("/audit/benchmark-overlap.json").read_text())
    if any(r["failures"] or r["claim_status"]!="complete_matched_fast_protocol" for r in benchmarks.values()):
        raise ValueError("External benchmark coverage is incomplete")
    if overlap["benchmark_exact_byte_overlaps"] or gui["n_clean"]!=1084:
        raise ValueError("Dataset audit changed")
    for name,metadata in lock["files"].items():
        if sha(folder/name)!=metadata["sha256"]:
            raise ValueError(f"Prepared artifact changed: {name}")
    metrics={"training_and_internal_evaluation":training,"gui_derived_evaluation":gui,"external_benchmarks":benchmarks,
             "benchmark_definition":"standard source splits, declared fast closed-choice protocol; not native-generation leaderboard scores",
             "calibration":"raw option probabilities; no held-out fitted calibration"}
    (folder/"metrics.json").write_text(json.dumps(metrics,indent=2))
    inventory=json.loads(Path("/audit/inventory.json").read_text())["inventory"]
    (folder/"dataset_audit.json").write_text(json.dumps({"inventory":{k:{"splits":v["splits"],"cross_split_source_groups":v["cross_split_source_groups"],"cross_split_media_hashes":v["cross_split_media_hashes"]} for k,v in inventory.items()},"benchmark_overlap":{"count":len(overlap["benchmark_exact_byte_overlaps"]),"limits":overlap["limitations"]},"excluded_GUI_dev_shared_screenshots":3,"excluded_GUI_test_shared_screenshots":2},indent=2))
    lines=[]
    for prefix,report in benchmarks.items():
        for key,score in report["scores"].items():
            model,suite=key.split("/",1)
            lines.append(f"| {model} | {suite} | {report['video_mode']} | {score['correct']}/{score['n']} | {score['accuracy']:.2%} | {score['task_macro_accuracy']:.2%} |")
    card=f'''---
base_model: {MODEL_ID}
tags:
  - multimodal
  - decision-classification
  - image
  - video
  - research
---

# Jev-Omni unified v4

One complete merged Gemma 4 12B checkpoint, one general decision head, one processor and a Choice/Noul/Score adapter. No separate base-model or head repository needs assembling. This is a research release independent of TypeSafe AI. The supplied state/media and bounded answer options are the decision input; no agent loop or generated explanation is used.

Continued head training used **59,000 examples**: all 47,680 general v2 examples, 7,962 GUI examples and 3,358 direct-visual-category SoccerNet examples. Our backbone stayed frozen; the upstream backbone already contains merged LoRA training. Training sources retain mixed terms. Source media, benchmark examples and labels are not redistributed. The upstream model's Apache-2.0 declaration applies to its artifacts; inspect source terms before commercial use. Wrapper code comes from the MIT-licensed project.

## Measured results

Internal general test: v3 and v4 both answered 8,051/9,284 (86.72%). Labeled visual SoccerNet development: 414/847 to 419/847 (48.88% to 49.47%). That five-answer gain is on the epoch-selection set and is not a demonstrated broad visual gain. Usable public SoccerNet test, all categories: 235/496 to 237/496 (47.38% to 47.78%). Four malformed source-test labels are excluded; this is not an official 500-row challenge submission. Clean derived Mind2Web test: v3 {gui['clean_screenshot_subset']['v3']['overall']['correct']}/1,084, v4 {gui['clean_screenshot_subset']['v4']['overall']['correct']}/1,084. The correct control is guaranteed among four preselected named candidates; this is not official Mind2Web task success.

External benchmark table uses all 1,901 labeled BLINK validation questions and all 1,580 TempCompass MC / 2,453 yes-no test questions. **Decider rows cover only eligible single-image BLINK questions**, not the full benchmark; compare the same IDs, not different denominators. Original option order is preserved. Batch one, H100, BF16, SDPA, thinking disabled, one decision forward; ordinary VLMs score allowed letter-token logits. FLA 0.5.2 is installed for Qwen, but causal convolution uses the reference fallback. Latency includes preprocessing and first-use kernel compilation; these are eager-harness results, not each vendor's best production server.

| Model | Source suite | Video representation | Correct / total | Micro accuracy | Task macro |
| --- | --- | --- | ---: | ---: | ---: |
'''+"\n".join(lines)+f'''

Sequence and timestamp modes use the same 16 midpoint frames. Native video mode uses the processor's sampling and 70 soft tokens per frame; its sample indices and token budget can differ. It also incurs the harness's initial midpoint decoding, so use a separate production latency profile before claiming native-video speed. These scores are under the declared fast protocol and are **not comparable without qualification to native-generation model-card/leaderboard scores**. No SOTA or hidden-test rank is claimed. BLINK hidden test, full MVBench, Video-MME, captioning and current sealed JevBench composites were not evaluated in this release.

Three repeated GUI development screenshots were excluded from v4 selection, and two training screenshots were excluded from the primary test diagnostic. Exact-byte benchmark overlap with recorded general/GUI training media was zero; crops, transcoding, near duplicates and unknown upstream pretraining remain limitations. See `dataset_audit.json`, `metrics.json` and `release.lock.json` for provenance, per-task scores, failures and latency scopes.

## Load on a remote CUDA GPU

Download `unified_jev.py` at the chosen release revision, inspect it, then use:

```python
from unified_jev import load_unified_jev
model = load_unified_jev("{REPO}", revision="<exact release revision>")
result = model.decide_typed({{
    "state": "The meeting starts at 10:00; it is 09:00.",
    "questions": {{"started": {{"type": "noul", "instructions": "Has it started?"}}}}
}})
```

The classifier's `predict(..., modality="image"|"video"|"audio", media=...)` uses the upstream loader. Multiple typed questions currently require one classifier forward per question. Public loading defaults to anonymous read. Scores/probabilities are raw, uncalibrated head outputs. A 256-slot structure does not establish quality for 256-option decisions; upstream quality evidence is bounded to smaller option sets.

Training/benchmark code: [vl-jev-modal](https://github.com/ferdinandl007/vl-jev-modal), especially `docs/UNIFIED_RELEASE_AND_BENCHMARKS.md`. All artifacts were built and tested on Modal.
'''
    (folder/"README.md").write_text(card)
    token=next((os.environ[k] for k in ("HF_TOKEN","HUGGINGFACE_TOKEN","HUGGINGFACE_HUB_TOKEN") if os.environ.get(k)),None)
    if not token:
        raise ValueError("No publication token in managed Modal secret")
    api=HfApi(token=token)
    if api.repo_exists(REPO):
        previous=Path(hf_hub_download(REPO,"head.pt",token=token))
        if sha(previous)!=lock["head_sha256"]:
            raise ValueError("Refusing to overwrite a different existing release")
    api.create_repo(REPO,private=False,exist_ok=True)
    api.upload_folder(repo_id=REPO,folder_path=str(folder),commit_message="Publish audited consolidated v4 model and declared benchmark results")
    public=HfApi(token=False).model_info(REPO,files_metadata=True)
    readback=Path(hf_hub_download(REPO,"head.pt",revision=public.sha,token=False))
    if sha(readback)!=lock["head_sha256"]:
        raise ValueError("Anonymous public head readback mismatch")
    publication={"repo":REPO,"revision":public.sha,"head_sha256":lock["head_sha256"],"anonymous_head_readback":True}
    (folder/"publication.json").write_text(json.dumps(publication,indent=2));release.commit()
    return publication


@app.function(image=base_image,volumes={"/release":release,"/model-cache":model_cache},gpu="H100",timeout=3600)
def verify(publication):
    import importlib.util
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    loader=Path(hf_hub_download(REPO,"unified_jev.py",revision=publication["revision"],token=False))
    spec=importlib.util.spec_from_file_location("released_unified_jev",loader)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    classifier=module.load_unified_jev(REPO,revision=publication["revision"],token=False)
    request={"state":"The meeting starts at 10:00. It is now 09:00.","questions":{
        "started":{"type":"noul","instructions":"Has it started?"},
        "phase":{"type":"choice","instructions":"Choose the phase","criteria":{"before":"Before","during":"During"}},
        "progress":{"type":"score","instructions":"Rate progress","criteria":["Not started","In progress","Finished"]}}}
    answer=classifier.decide_typed(request)
    if {k:v["type"] for k,v in answer["answers"].items()}!={k:v["type"] for k,v in request["questions"].items()}:
        raise ValueError("Released typed interface failed")
    result={**publication,"anonymous_complete_model_loaded":True,"all_three_typed_outputs_verified":True}
    (Path("/release/jev-omni-unified-v4")/"verification.json").write_text(json.dumps(result,indent=2));release.commit();model_cache.commit()
    return result


@app.function(image=image,volumes=mounts,timeout=86400)
def pipeline():
    import time
    from pathlib import Path
    prepare.remote();release.reload()
    deadline=time.monotonic()+8*3600
    while not (Path("/bench-results/aggregate.json").exists() and (Path("/general-runs")/RUN/"gui-test.json").exists()):
        if time.monotonic()>deadline:
            raise TimeoutError("Release evaluation did not finish; prepared model remains on Modal")
        time.sleep(30);bench.reload();general_training.reload()
    publication=publish.remote()
    return verify.remote(publication)


@app.function(image=image,volumes={**mounts,"/text-data":text_data},
              secrets=[modal.Secret.from_name("huggingface-secret")],timeout=1800)
def update_report(source_revision: str):
    """Add paired comparisons and public typed diagnostics without changing weights."""
    import os,re
    from pathlib import Path
    from huggingface_hub import HfApi,hf_hub_download
    if not re.fullmatch(r"[0-9a-f]{40}",source_revision):raise ValueError("Require an exact public code commit")
    folder=Path("/release/jev-omni-unified-v4")
    verification=json.loads((folder/"verification.json").read_text())
    if not verification["anonymous_complete_model_loaded"]:raise ValueError("Complete model not verified")
    comparison=json.loads(Path("/bench-results/comparison.json").read_text())
    public_text=Path("/text-data/jevbench/bb05a335bc809e61b20c0f745d25499a82b326fc/report.json")
    if not public_text.exists():raise ValueError("Public typed benchmark has not finished")
    typed=json.loads(public_text.read_text())
    if typed["head_sha256"]!=verification["head_sha256"]:raise ValueError("Typed benchmark candidate differs")
    metrics=json.loads((folder/"metrics.json").read_text())
    metrics.update({"paired_cluster_comparisons":comparison,"public_jevbench":typed,
                    "source_code_revision":source_revision,"public_load_verification":verification})
    (folder/"metrics.json").write_text(json.dumps(metrics,indent=2))
    card=(folder/"README.md").read_text()
    marker="## Paired comparisons and typed text diagnostics"
    card=card.split(marker)[0].rstrip()
    lines=["",marker,"", "Paired accuracy differences use 2,000 source-cluster bootstrap resamples; 95% intervals are exploratory and are not a leaderboard rank.","",
           "| Comparison | Suite | N | Accuracy difference | 95% interval |",
           "| --- | --- | ---: | ---: | ---: |"]
    for name,r in comparison["comparisons"].items():
        if not name.startswith("jev_unified_v4/"):continue
        lo,hi=r["cluster_bootstrap_95pct_delta"]
        suite=name.rsplit("/",1)[-1]
        lines.append(f"| {name} | {suite} | {r['n']} | {100*r['delta_accuracy']:+.2f} pp | [{100*lo:+.2f}, {100*hi:+.2f}] pp |")
    lines += ["", "Public JevBench uses the original 231 public tasks at source revision `"+typed["source_revision"]+"`, unmodified source correctness scoring, original label order and our native typed prompt. It does not evaluate the current sealed composite. No public benchmark tasks were used in our continued training.","",
              "| Head | Correct / total | Micro accuracy | Group macro | Choice | Noul | Score exact class |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,r in typed["models"].items():
        kinds=r["by_type"]
        lines.append(f"| {name} | {r['correct']}/{r['n']} | {r['micro_accuracy']:.2%} | {r['group_macro_accuracy']:.2%} | {kinds['choice']['accuracy']:.2%} | {kinds['noul']['accuracy']:.2%} | {kinds['score']['accuracy']:.2%} |")
    lines += ["", "The inherited audio interface has not been accuracy-benchmarked in this release. Multi-question typed requests still require one forward per question. Raw Score expectations and Noul/Choice probabilities are not calibrated.","",
              f"Reproducible source: [vl-jev-modal at {source_revision[:12]}](https://github.com/ferdinandl007/vl-jev-modal/tree/{source_revision}). Public loading of the complete checkpoint and all three typed interfaces was verified on Modal; the artifact head SHA-256 is `{verification['head_sha256']}`."]
    (folder/"README.md").write_text(card+"\n"+"\n".join(lines)+"\n")
    token=next((os.environ[k] for k in ("HF_TOKEN","HUGGINGFACE_TOKEN","HUGGINGFACE_HUB_TOKEN") if os.environ.get(k)),None)
    if not token:raise ValueError("Missing managed publication token")
    api=HfApi(token=token)
    # Only metadata goes up in this update; all 12B weights remain immutable.
    api.upload_folder(repo_id=REPO,folder_path=str(folder),allow_patterns=["README.md","metrics.json"],commit_message="Add paired benchmark uncertainty and public Choice/Noul/Score diagnostics")
    revision=HfApi(token=False).model_info(REPO).sha
    head=Path(hf_hub_download(REPO,"head.pt",revision=revision,token=False))
    if sha(head)!=verification["head_sha256"]:raise ValueError("Public metadata update changed the model")
    published_metrics=json.loads(Path(hf_hub_download(REPO,"metrics.json",revision=revision,token=False)).read_text())
    if published_metrics["source_code_revision"]!=source_revision or published_metrics["public_jevbench"]!=typed:
        raise ValueError("Anonymous public metrics readback differs")
    result={"repo":REPO,"revision":revision,"head_sha256":verification["head_sha256"],"source_revision":source_revision,"anonymous_metrics_readback":True}
    (folder/"publication-final.json").write_text(json.dumps(result,indent=2));release.commit()
    return result


@app.local_entrypoint()
def main(mode: str="prepare", source_revision: str=""):
    if mode=="status":answer=status.remote()
    elif mode=="update-report":answer=update_report.remote(source_revision)
    else:answer={"prepare":prepare,"publish":publish,"pipeline":pipeline}[mode].remote()
    print(json.dumps(answer,indent=2))


@app.function(image=image,volumes=mounts,timeout=300)
def status():
    from pathlib import Path
    from huggingface_hub import HfApi
    root=Path("/release/jev-omni-unified-v4")
    output={p.name:json.loads(p.read_text()) for p in root.glob("*.json") if p.name in {"publication.json","verification.json","publication-final.json"}}
    output["prepared_files"]={p.name:p.stat().st_size for p in root.iterdir() if p.is_file()} if root.exists() else {}
    if HfApi(token=False).repo_exists(REPO):output["public_revision"]=HfApi(token=False).model_info(REPO).sha
    return output
