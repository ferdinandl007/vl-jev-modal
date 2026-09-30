"""Pinned public benchmarks, remote materialization and matched decision evaluation.

No training or synthetic labels. Full source splits and original option order.
The fast protocol is explicitly distinct from a model's native generation protocol.
"""
import json
import modal
from experiments.modal_jev_omni_sports_train import base_image, model_cache

app = modal.App("vl-jev-official-benchmarks-20260930")
data = modal.Volume.from_name("vl-jev-official-benchmarks", create_if_missing=True)
results = modal.Volume.from_name("vl-jev-official-benchmark-results", create_if_missing=True)
cpu = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==1.33.0", "pyarrow==21.0.0", "pillow==12.1.1", "numpy==2.4.6")
gpu_image = base_image.pip_install("av==16.1.0", "flash-linear-attention==0.5.2", "fla-core==0.5.2")
BLINK = ("BLINK-Benchmark/BLINK", "a3666eb249237ba3d5eca8db21176cc47967e040")
TEMP = ("lmms-lab/TempCompass", "c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722")
MODELS = {
    "gemma12b": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    "qwen4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "decider2b": ("Mapika/decider-2b-vision", "863e290863655f1d6b69324d77d09ac972d21609"),
}
DECIDER_CODE = "f42a043b15a99a872bd625a9fc3a2cbd791c5ee4"
HEAD = ("ferdinandl007/jev-omni-general-head-v3", "10e061b811f4add247dc0cbf06a67cb5c0233ad8")
HEAD_SHA = "f1f2b4ba26776198bf0cf716f38159419fb2ba35a881943237de11624182f3e9"
MOUNTS = {"/bench": data, "/results": results, "/model-cache": model_cache}
UNIFIED_RUN = "general-head-v4-unified-audited"


def digest(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=cpu, volumes={"/bench": data}, timeout=3600)
def build():
    import re
    import zipfile
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    root = Path("/bench"); root.mkdir(exist_ok=True)
    def fetch(repo, revision, file):
        return Path(hf_hub_download(repo, file, repo_type="dataset", revision=revision,
                                   cache_dir="/bench/source-cache"))
    rows = []; files = []
    for file in sorted(HfApi(token=False).list_repo_files(BLINK[0], repo_type="dataset", revision=BLINK[1])):
        if not file.endswith("/val-00000-of-00001.parquet"):
            continue
        path = fetch(*BLINK, file); files.append({"repo": BLINK[0], "file": file, "sha256": digest(path)})
        task = file.split("/")[0]
        for item in pq.read_table(path).to_pylist():
            media = []
            for index in range(1, 5):
                asset = item.get(f"image_{index}")
                if not asset:
                    continue
                if not asset.get("bytes"):
                    raise ValueError(f"Missing image bytes: {item['idx']} {index}")
                folder = root / "images" / task; folder.mkdir(parents=True, exist_ok=True)
                dest = folder / f"{item['idx']}_{index}.image"
                dest.write_bytes(asset["bytes"])
                media.append({"path": str(dest), "sha256": digest(dest)})
            answer = re.fullmatch(r"\(?([A-Z])\)?", item["answer"].strip())
            if not answer or not media:
                raise ValueError(f"Invalid BLINK row: {item['idx']}; answer={item['answer']!r}; media={len(media)}; choices={item['choices']!r}")
            rows.append({"id": item["idx"], "suite": "blink_val", "task": task,
                         "question": item["question"], "prompt": item["prompt"],
                         "options": item["choices"], "gold": ord(answer[1])-65,
                         "images": media, "video": None})
        print(f"BLINK {task}: materialized", flush=True)
    blink_n = len(rows)
    archive = fetch(*TEMP, "tempcompass_videos.zip")
    files.append({"repo": TEMP[0], "file": archive.name, "sha256": digest(archive)})
    video_dir = root / "tempcompass-videos"; video_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (video_dir / member.filename).resolve()
            if not target.is_relative_to(video_dir.resolve()):
                raise ValueError("Unsafe archive member")
        z.extractall(video_dir)
    videos = {p.stem: p for p in video_dir.rglob("*.mp4")}
    video_hashes = {}
    for kind in ("multi-choice", "yes_no"):
        path = fetch(*TEMP, f"{kind}/test-00000-of-00001.parquet")
        files.append({"repo": TEMP[0], "file": f"{kind}/test-00000-of-00001.parquet", "sha256": digest(path)})
        for index, item in enumerate(pq.read_table(path).to_pylist()):
            vid = item["video_id"]
            if vid not in videos:
                raise FileNotFoundError(vid)
            if vid not in video_hashes:
                video_hashes[vid] = digest(videos[vid])
            if kind == "multi-choice":
                lines = item["question"].splitlines()
                option_lines = [re.fullmatch(r"([A-Z])\.\s*(.+)", line) for line in lines]
                option_lines = [m for m in option_lines if m]
                if [m[1] for m in option_lines] != [chr(65+i) for i in range(len(option_lines))]:
                    raise ValueError("Nonsequential options")
                options = [m[2] for m in option_lines]
                question = "\n".join(line for line in lines if not re.fullmatch(r"[A-Z]\.\s*.+", line))
                if item["answer"] not in [f"{m[1]}. {m[2]}" for m in option_lines]:
                    raise ValueError(f"Unmatched answer: {item['answer']}")
                gold = ord(item["answer"][0])-65
            else:
                question = item["question"]; options = ["yes", "no"]
                gold = options.index(item["answer"].strip().lower())
            rows.append({"id": f"temp_{kind}_{index}", "suite": f"tempcompass_{kind}",
                         "task": item["dim"], "question": question,
                         "prompt": item["question"], "options": options, "gold": gold,
                         "images": [], "video": {"path": str(videos[vid]), "sha256": video_hashes[vid], "source_id": vid}})
    for row in rows:
        if not 0 <= row["gold"] < len(row["options"]):
            raise ValueError("Invalid gold")
    if blink_n != 1901 or sum(r["suite"] == "tempcompass_multi-choice" for r in rows) != 1580 or sum(r["suite"] == "tempcompass_yes_no" for r in rows) != 2453:
        raise ValueError("Official split count changed")
    manifest = root / "rows.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    lock = {"datasets": {"blink": BLINK, "tempcompass": TEMP}, "source_files": files,
            "rows": len(rows), "manifest_sha256": digest(manifest), "training_allowed": False}
    (root / "lock.json").write_text(json.dumps(lock, indent=2)); data.commit()
    return lock


def frames_for(row):
    from PIL import Image
    if not row["video"]:
        return [Image.open(a["path"]).convert("RGB") for a in row["images"]], []
    import cv2
    capture = cv2.VideoCapture(row["video"]["path"])
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(capture.get(cv2.CAP_PROP_FPS))
    if count < 1 or fps <= 0:
        capture.release(); raise ValueError("Video has no valid frame count/fps")
    indices = [min(count-1, int((i+0.5)*count/16)) for i in range(16)]
    frames = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Failed decoding frame {index}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return frames, [round(index/fps, 6) for index in indices]


def report_for(records, expected):
    import numpy as np
    from collections import defaultdict
    groups = defaultdict(list)
    model_names = {m for r in records for m in r.get("predictions", {})}
    for row in records:
        for model in model_names:
            groups[(model, row["suite"])].append((row, row.get("predictions", {}).get(model)))
    output = {}
    for (model, suite), entries in groups.items():
        tasks = defaultdict(list)
        for row, prediction in entries:
            tasks[row["task"]].append(int(prediction is not None and prediction["selected"] == row["gold"]))
        latencies = [p["warm_total_ms"] for _, p in entries if p is not None]
        correct = sum(sum(x) for x in tasks.values()); n = len(entries); proportion = correct/n
        z = 1.959963984540054; denominator = 1+z*z/n
        center = (proportion+z*z/(2*n))/denominator
        half_width = z*((proportion*(1-proportion)/n+z*z/(4*n*n))**0.5)/denominator
        output[f"{model}/{suite}"] = {
            "n": n, "correct": correct, "accuracy": proportion,
            "micro_wilson_95pct": [center-half_width, center+half_width],
            "task_macro_accuracy": float(np.mean([np.mean(x) for x in tasks.values()])),
            "by_task": {k: {"n": len(v), "correct": sum(v), "accuracy": sum(v)/len(v)} for k,v in tasks.items()},
            "latency_ms": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95))}}
    return {"expected_rows": expected, "processed_rows": len(records),
            "failures": sum("error" in r for r in records), "scores": output,
            "claim_status": "diagnostic_only" if len(records) != expected or any("error" in r for r in records) else "complete_matched_fast_protocol",
            "protocol": "batch1_bf16_sdpa_no_thinking_original_option_order_16_midpoint_frames_v1",
            "latency_scope": "warm media decode + preprocessing + device transfer + forward + choice scoring; excludes model loading and network",
            "not_native_generation_leaderboard_score": True}


def _evaluate(model_key: str = "jev", limit: int = 0, timeline: bool = False, native_video: bool = False, shard: int = 0, shards: int = 1):
    import hashlib
    import time
    import torch
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor, AutoModelForMultimodalLM, Qwen3_5ForConditionalGeneration
    from experiments.modal_jev_omni_sports_train import _load_classifier, _feature_content
    rows = [json.loads(line) for line in Path("/bench/rows.jsonl").open()]
    if model_key == "decider2b":
        if native_video or timeline:
            raise ValueError("The native decider vision adapter supports single images")
        rows = [r for r in rows if r["suite"] == "blink_val" and len(r["images"]) == 1]
    unified = None
    if model_key == "jev_all":
        unified = json.loads((Path("/general-runs") / UNIFIED_RUN / "report.json").read_text())
        if digest(Path(unified["checkpoint"])) != unified["checkpoint_sha256"]:
            raise ValueError("Unified checkpoint changed")
    # Smoke sampling spans all tasks and formats and never produces a benchmark claim.
    if limit:
        by_suite = {s: [r for r in rows if r["suite"] == s] for s in sorted({r["suite"] for r in rows})}
        rows = [r for group in by_suite.values() for r in group[:limit]]
    if not 0 <= shard < shards <= 8:
        raise ValueError("Invalid evaluation shards")
    rows = [r for index,r in enumerate(rows) if index % shards == shard]
    run = f"{model_key}-{'native' if native_video else 'timeline' if timeline else 'sequence'}-{'smoke'+str(limit) if limit else 'full'}-v2"
    if shards > 1:
        run += f"-shard{shard:02d}of{shards:02d}"
    root = Path("/results") / run; root.mkdir(parents=True, exist_ok=True)
    locked = {"manifest_sha256": digest(Path("/bench/rows.jsonl")), "model": model_key,
              "model_revision": MODELS.get(model_key), "head": HEAD, "head_sha256": HEAD_SHA,
              "timeline": timeline, "native_video": native_video, "limit": limit, "protocol_version": 2,
              "unified_sha256": unified["checkpoint_sha256"] if unified else None}
    locked.update({"shard":shard,"shards":shards})
    lock_path = root / "lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        previous.setdefault("shard",0); previous.setdefault("shards",1)
        if previous != json.loads(json.dumps(locked)):
            raise ValueError("Cached run configuration differs")
    lock_path.write_text(json.dumps(locked, indent=2))
    records = []
    for pack in sorted(root.glob("pack-*.json")):
        records.extend(json.loads(pack.read_text()))
    completed = {r["id"] for r in records}
    if len(completed) != len(records) or not completed <= {r["id"] for r in rows}:
        raise ValueError("Invalid cached row identities")
    if model_key in {"jev", "jev_all"}:
        _, classifier, package = _load_classifier()
        source = hf_hub_download(*HEAD[:1], "decision_head.pt", revision=HEAD[1], token=False)
        if digest(Path(source)) != HEAD_SHA:
            raise ValueError("Public head hash differs")
        v3 = package._Head256(3840).cuda().eval()
        v3.load_state_dict(torch.load(source, map_location="cuda", weights_only=True))
        heads = [("jev_upstream", classifier.head), ("jev_general_v3", v3)]
        if unified:
            v4 = package._Head256(3840).cuda().eval()
            v4.load_state_dict(torch.load(unified["checkpoint"], map_location="cuda", weights_only=True))
            heads.append(("jev_unified_v4", v4))
    elif model_key == "decider2b":
        import io, sys, tarfile, urllib.request
        from huggingface_hub import snapshot_download
        code_root = Path("/model-cache/decider-code") / DECIDER_CODE
        if not code_root.exists():
            raw = urllib.request.urlopen(f"https://codeload.github.com/Mapika/decider/tar.gz/{DECIDER_CODE}",timeout=120).read()
            code_root.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
                archive.extractall(code_root,filter="data")
        sys.path.insert(0,str(next(code_root.iterdir())))
        from decider.vision import VisionDecisionModel
        from decider.infer import Example, Q
        path = snapshot_download(MODELS[model_key][0],revision=MODELS[model_key][1])
        decision_model = VisionDecisionModel(path,grad_ckpt=False).cuda().eval().requires_grad_(False)
    else:
        repo, revision = MODELS[model_key]
        processor = AutoProcessor.from_pretrained(repo, revision=revision)
        model_cls = AutoModelForMultimodalLM if model_key == "gemma12b" else Qwen3_5ForConditionalGeneration
        model = model_cls.from_pretrained(repo, revision=revision, dtype=torch.bfloat16,
                                        attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    pending = []; pack_index = len(list(root.glob("pack-*.json")))
    for row in rows:
        if row["id"] in completed:
            continue
        record = {"id": row["id"], "suite": row["suite"], "task": row["task"], "gold": row["gold"]}
        record["cluster"] = row["video"]["source_id"] if row["video"] else hashlib.sha256(json.dumps(sorted(a["sha256"] for a in row["images"])).encode()).hexdigest()
        start = time.perf_counter()
        try:
            frames, timestamps = frames_for(row)
            state = "" if not timeline or not timestamps else "One continuous clip; ordered frames at seconds: " + json.dumps(timestamps)
            if model_key in {"jev", "jev_all"}:
                torch.cuda.synchronize(); before = time.perf_counter()
                if native_video and row["video"]:
                    content = [{"type": "video", "video": row["video"]["path"]},
                               {"type": "text", "text": package._prompt(state, row["question"], row["options"])}]
                    inputs = classifier.processor.apply_chat_template([{"role":"user","content":content}],
                        tokenize=True, add_generation_prompt=True, enable_thinking=False,
                        return_dict=True, return_tensors="pt", processor_kwargs={"num_frames":16,"max_soft_tokens":70})
                    inputs = {k:v.to("cuda",dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda") for k,v in inputs.items()}
                    classifier._capture.clear()
                    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
                        classifier.model(**inputs,use_cache=False,**classifier._extra)
                    feature = classifier._capture["hidden"].detach().float()[0]
                    record["native_video_input_tokens"] = int(inputs["input_ids"].numel())
                else:
                    feature = _feature_content(torch, classifier, package, frames, state, row["question"], row["options"]).cuda()
                torch.cuda.synchronize(); forward_ms = (time.perf_counter()-before)*1000
                predictions = {}
                with torch.inference_mode():
                    for name, head in heads:
                        head_start = time.perf_counter()
                        logits = head(feature.unsqueeze(0), torch.tensor([len(row["options"])], device="cuda"))[0][:len(row["options"])]
                        scores = logits.float().cpu().tolist()
                        predictions[name] = {"selected": max(range(len(scores)), key=scores.__getitem__),
                                             "scores": scores, "backbone_and_preprocess_ms": forward_ms,
                                             "head_ms": (time.perf_counter()-head_start)*1000,
                                             "warm_total_ms": (time.perf_counter()-start)*1000}
                record["predictions"] = predictions
            elif model_key == "decider2b":
                example = Example(state or "Answer using the supplied image.",[Q(row["question"],row["options"],0)])
                inputs = decision_model.prepare([(frames[0],example)])
                torch.cuda.synchronize(); before = time.perf_counter()
                with torch.inference_mode():
                    logits = decision_model.slot_logits(inputs)[0,:len(row["options"])]
                    scores = logits.float().cpu().tolist()
                torch.cuda.synchronize()
                record["predictions"] = {model_key:{"selected":max(range(len(scores)),key=scores.__getitem__),
                    "scores":scores,"forward_ms":(time.perf_counter()-before)*1000,
                    "warm_total_ms":(time.perf_counter()-start)*1000,"input_tokens":int(inputs["input_ids"].numel())}}
            else:
                labels = [chr(65+i) for i in range(len(row["options"]))]
                ids = [processor.tokenizer.encode(s, add_special_tokens=False) for s in labels]
                if any(len(x) != 1 for x in ids):
                    raise ValueError("Choice letters are not single tokens")
                content = ([{"type":"video","video":row["video"]["path"]}] if native_video and row["video"] else [{"type": "image", "image": im} for im in frames])
                prompt = row["prompt"]
                if row["suite"] == "tempcompass_yes_no":
                    prompt += "\n(A) yes\n(B) no"
                prompt += "\nPlease directly give the best option. Reply with its letter only."
                if state:
                    prompt = state + "\n" + prompt
                content.append({"type": "text", "text": prompt})
                video_kwargs = {"processor_kwargs":{"num_frames":16,"max_soft_tokens":70}} if native_video and row["video"] else {}
                inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors="pt", **video_kwargs)
                inputs = inputs.to("cuda")
                torch.cuda.synchronize(); before = time.perf_counter()
                with torch.inference_mode():
                    output = model(**inputs, use_cache=False, logits_to_keep=1)
                    logits = output.logits[0, -1, [x[0] for x in ids]].float()
                    scores = logits.cpu().tolist()
                torch.cuda.synchronize(); forward_ms = (time.perf_counter()-before)*1000
                record["predictions"] = {model_key: {"selected": max(range(len(scores)), key=scores.__getitem__),
                    "scores": scores, "forward_ms": forward_ms,
                    "warm_total_ms": (time.perf_counter()-start)*1000, "input_tokens": int(inputs["input_ids"].numel())}}
            record["sampled_timestamps"] = timestamps
            record["midpoint_timestamps_are_actual_model_input"] = not native_video
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"ERROR {row['id']}: {record['error']}", flush=True)
            # Stop a systemic model/processor failure; never burn a complete split on it.
            if len(records)+len(pending) < 3:
                raise RuntimeError(record["error"]) from None
        pending.append(record)
        if len(pending) == 32:
            (root / f"pack-{pack_index:05d}.json").write_text(json.dumps(pending))
            records.extend(pending); pending = []; pack_index += 1
            summary = report_for(records, len(rows)); (root / "report.json").write_text(json.dumps(summary, indent=2))
            results.commit(); print(f"{run}: {len(records)}/{len(rows)}", flush=True)
    if pending:
        (root / f"pack-{pack_index:05d}.json").write_text(json.dumps(pending)); records.extend(pending)
    summary = report_for(records, len(rows))
    summary["video_mode"] = "native_processor_16frames_70softtokens" if native_video else "images_with_timestamps" if timeline else "ordered_images"
    summary["evaluated_scope"] = "BLINK eligible single-image subset only" if model_key == "decider2b" else "full_BLINK_val_and_TempCompass_MC_and_yes_no"
    summary["runtime"] = {"torch":str(torch.__version__),"gpu":torch.cuda.get_device_name(),"fla_version":"0.5.2", "causal_conv1d_optimized":False}
    summary["shard"] = shard; summary["shards"] = shards
    if limit:
        summary["claim_status"] = "smoke_only"
    (root / "report.json").write_text(json.dumps(summary, indent=2)); results.commit(); model_cache.commit()
    return summary


@app.function(image=gpu_image, gpu="H100", volumes={**MOUNTS, "/general-runs":modal.Volume.from_name("vl-jev-general-v1-training")},
              secrets=[modal.Secret.from_name("huggingface-secret")],timeout=86400,max_containers=16)
def evaluate(model_key: str="jev",limit: int=0,timeline: bool=False,native_video: bool=False,shard: int=0,shards: int=1):
    try:
        return _evaluate(model_key,limit,timeline,native_video,shard,shards)
    except Exception as error:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"{type(error).__name__}: {error}") from None


@app.function(image=cpu, volumes={"/results": results,"/general-runs":modal.Volume.from_name("vl-jev-general-v1-training")}, timeout=300)
def status():
    from pathlib import Path
    reports = {}
    for p in Path("/results").glob("*/report.json"):
        r = json.loads(p.read_text())
        reports[p.parent.name] = {k:r[k] for k in ("expected_rows","processed_rows","failures","claim_status")}
        reports[p.parent.name]["scores"] = {k:{field:v[field] for field in ("n","correct","accuracy","task_macro_accuracy","latency_ms")} for k,v in r["scores"].items()}
    training = Path("/general-runs")/UNIFIED_RUN/"report.json"
    if training.exists():
        r = json.loads(training.read_text())
        reports["unified_training"] = {k:r[k] for k in ("train_rows","selected_epoch","status","checkpoint_sha256","dev_counts")}
        reports["unified_training"]["dev"] = {name:{"baseline":r["baseline_dev"][name]["overall"],"selected":r["selected_dev"][name]["overall"]} for name in ("general","gui","soccer_visual")}
        reports["unified_training"]["test"] = {name:{"baseline":r["baseline_test"][name]["overall"],"selected":r["selected_test"][name]["overall"]} for name in ("general","soccer_public_usable")}
    return reports


@app.function(image=cpu,volumes={"/results":results,"/bench":data},timeout=300)
def progress():
    from pathlib import Path
    expected=[json.loads(line) for line in Path("/bench/rows.jsonl").open()]
    output={}
    for prefix in ("gemma12b-sequence","qwen4b-sequence","qwen9b-sequence","decider2b-sequence","jev_all-sequence","jev_all-native","jev_all-timeline"):
        rows=[]
        for folder in Path("/results").glob(prefix+"-full-v2-shard*of04"):
            for pack in folder.glob("pack-*.json"):rows.extend(json.loads(pack.read_text()))
        eligible=[r for r in expected if r["suite"]=="blink_val" and len(r["images"])==1] if prefix.startswith("decider") else expected
        complete=len(rows)==len(eligible) and {r["id"] for r in rows}=={r["id"] for r in eligible}
        output[prefix]={"processed":len(rows),"expected":len(eligible),"complete":complete}
        if complete:
            report=report_for(rows,len(eligible))
            output[prefix]["scores"]={k:{field:v[field] for field in ("n","correct","accuracy","task_macro_accuracy","latency_ms")} for k,v in report["scores"].items()}
    return output


@app.function(image=cpu, volumes={"/results":results,"/bench":data,
              "/general-runs":modal.Volume.from_name("vl-jev-general-v1-training")}, timeout=86400)
def pipeline(shards: int = 4):
    import time
    from pathlib import Path
    if not 1 <= shards <= 8:
        raise ValueError("Invalid shard count")
    jobs = []
    # Comparator runs start while the candidate finishes development selection.
    for key in ("gemma12b","qwen4b","qwen9b","decider2b"):
        for shard in range(shards):
            jobs.append(evaluate.spawn(key,0,False,False,shard,shards))
    training = modal.Volume.from_name("vl-jev-general-v1-training")
    deadline = time.monotonic()+7200
    while not (Path("/general-runs")/UNIFIED_RUN/"report.json").exists():
        if time.monotonic()>deadline:
            raise TimeoutError("Unified development selection did not finish")
        time.sleep(30); training.reload()
    for native, timeline in ((False,False),(True,False),(False,True)):
        for shard in range(shards):
            jobs.append(evaluate.spawn("jev_all",0,timeline,native,shard,shards))
    for index,job in enumerate(jobs):
        result = job.get()
        if result["failures"]:
            raise ValueError("Benchmark shard has failures; publication remains blocked")
        print(f"benchmark shards completed {index+1}/{len(jobs)}",flush=True)
    results.reload()
    merged = {}
    for prefix in ("gemma12b-sequence","qwen4b-sequence","qwen9b-sequence","decider2b-sequence",
                   "jev_all-sequence","jev_all-native","jev_all-timeline"):
        records = []
        for shard in range(shards):
            suffix = f"-shard{shard:02d}of{shards:02d}" if shards>1 else ""
            folder = Path("/results") / (prefix+"-full-v2"+suffix)
            for pack in sorted(folder.glob("pack-*.json")):
                records.extend(json.loads(pack.read_text()))
        source_rows = [json.loads(line) for line in Path("/bench/rows.jsonl").open()]
        if prefix.startswith("decider"):
            source_rows = [r for r in source_rows if r["suite"]=="blink_val" and len(r["images"])==1]
        if {r["id"] for r in records}!={r["id"] for r in source_rows} or len(records)!=len(source_rows):
            raise ValueError("Aggregate coverage mismatch")
        merged[prefix] = report_for(records,len(source_rows))
        merged[prefix]["evaluated_scope"] = "eligible_single_image_subset" if prefix.startswith("decider") else "full_BLINK_val_TempCompass_MC_yes_no"
        merged[prefix]["video_mode"] = prefix.split("-",1)[1]
    (Path("/results")/"aggregate.json").write_text(json.dumps(merged,indent=2)); results.commit()
    return {key:{"failures":value["failures"],"processed":value["processed_rows"],
                 "scores":{k:{field:v[field] for field in ("n","correct","accuracy","task_macro_accuracy","latency_ms")} for k,v in value["scores"].items()}} for key,value in merged.items()}


@app.local_entrypoint()
def main(mode: str = "build", model: str = "jev", limit: int = 0, timeline: bool = False, native_video: bool = False):
    if mode == "build":
        answer = build.remote()
    elif mode == "pipeline":
        answer = pipeline.remote()
    elif mode == "analyze":
        answer = analyze.remote()
    elif mode == "status":
        answer = status.remote()
    elif mode == "progress":
        answer = progress.remote()
    else:
        answer = evaluate.remote(model, limit, timeline, native_video)
    print(json.dumps(answer, indent=2))


def analyze_saved_results():
    from pathlib import Path
    import numpy as np
    from collections import defaultdict
    def read(prefix):
        found={}
        for folder in Path("/results").glob(prefix+"-full-v2-shard*of04"):
            for path in folder.glob("pack-*.json"):
                for row in json.loads(path.read_text()):
                    if row["id"] in found:
                        raise ValueError("Repeated evaluation identity")
                    found[row["id"]]=row
        return found
    streams={}
    for prefix in ("jev_all-sequence","jev_all-native","jev_all-timeline","gemma12b-sequence","qwen4b-sequence","qwen9b-sequence","decider2b-sequence"):
        rows=read(prefix)
        for model in {key for row in rows.values() for key in row.get("predictions",{})}:
            streams[f"{model}/{prefix.split('-')[1]}"]={i:{**row,"prediction":row["predictions"].get(model)} for i,row in rows.items()}
    output={"protocol":"paired cluster bootstrap, 2000 resamples, seed 3407; cluster by source video ID or exact image set; no fitted calibration",
            "comparisons":{},"option_score_diagnostics":{}}
    def compare(before,after,scope=None):
        left,right=streams[before],streams[after];ids=set(left)&set(right)
        if scope:
            ids&=scope
        suites=sorted({right[i]["suite"] for i in ids})
        for suite in suites:
            selected=sorted(i for i in ids if right[i]["suite"]==suite)
            groups=defaultdict(lambda:[0,0,0])
            improved=regressed=0
            for i in selected:
                a,b=left[i],right[i]
                if a["gold"]!=b["gold"]:
                    raise ValueError("Paired gold mismatch")
                x=int(a["prediction"] is not None and a["prediction"]["selected"]==a["gold"])
                y=int(b["prediction"] is not None and b["prediction"]["selected"]==b["gold"])
                improved+=int(y>x);regressed+=int(x>y)
                values=groups[b["cluster"]];values[0]+=x;values[1]+=y;values[2]+=1
            array=np.asarray(list(groups.values()),dtype=float);rng=np.random.default_rng(3407);boot=[]
            for offset in range(0,2000,100):
                draws=rng.integers(0,len(array),size=(100,len(array)))
                totals=array[draws].sum(axis=1);boot.extend(((totals[:,1]-totals[:,0])/totals[:,2]).tolist())
            totals=array.sum(axis=0)
            output["comparisons"][f"{after} versus {before}/{suite}"+("/eligible_single_image" if scope else "")]={
                "n":len(selected),"clusters":len(groups),"before_correct":int(totals[0]),"after_correct":int(totals[1]),
                "improved_only":improved,"regressed_only":regressed,"delta_accuracy":float((totals[1]-totals[0])/totals[2]),
                "cluster_bootstrap_95pct_delta":np.percentile(boot,[2.5,97.5]).tolist()}
    target="jev_unified_v4/sequence"
    for baseline in ("jev_upstream/sequence","jev_general_v3/sequence","gemma12b/sequence","qwen4b/sequence","qwen9b/sequence"):
        compare(baseline,target)
    for alternative in ("jev_unified_v4/native","jev_unified_v4/timeline"):
        compare(target,alternative)
    eligible=set(streams["decider2b/sequence"])
    for key in (target,"jev_upstream/sequence","jev_general_v3/sequence","gemma12b/sequence","qwen4b/sequence","qwen9b/sequence"):
        compare("decider2b/sequence",key,eligible)
    for name,rows in streams.items():
        for suite in sorted({r["suite"] for r in rows.values()}):
            entries=[r for r in rows.values() if r["suite"]==suite]
            if any(r["prediction"] is None for r in entries):
                raise ValueError("Missing diagnostic score")
            losses=[];briers=[];confs=[];correct=[]
            for row in entries:
                logits=np.asarray(row["prediction"]["scores"],dtype=float)
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite scores")
                exp=np.exp(logits-logits.max());p=exp/exp.sum();gold=row["gold"]
                losses.append(float(-np.log(max(float(p[gold]),1e-12))))
                truth=np.zeros(len(p));truth[gold]=1;briers.append(float(((p-truth)**2).sum()))
                confs.append(float(p.max()));correct.append(int(p.argmax()==gold))
            confs=np.asarray(confs);correct=np.asarray(correct);ece=0
            for lower in np.linspace(0,0.9,10):
                mask=(confs>=lower)&(confs<(lower+0.1) if lower<0.9 else confs<=1)
                if mask.any():
                    ece+=float(mask.mean()*abs(correct[mask].mean()-confs[mask].mean()))
            output["option_score_diagnostics"][f"{name}/{suite}"]={"n":len(entries),"log_loss":float(np.mean(losses)),
                "brier":float(np.mean(briers)),"ece_10_equal_width_bins":ece,"status":"raw_uncalibrated_scores; descriptive test diagnostics"}
    Path("/results/comparison.json").write_text(json.dumps(output,indent=2))
    return output


@app.function(image=cpu,volumes={"/results":results},timeout=3600)
def analyze():
    result=analyze_saved_results();results.commit()
    return result
