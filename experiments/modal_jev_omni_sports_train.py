"""Modal-only sports fine-tuning job for the published Jev-Omni decision head.

Modes:
  inspect  Inventory available/reviewed sports data and write candidate manifest.
  smoke    Verify frozen-feature extraction, gradients, and checkpointing on
           two synthetic Modal clips. This is not a sports training result.
  data-smoke  Extract one feature per available pilot family without training.
  train    Fine-tune the classifier head on materialized pilot rows using an
           explicit reviewed or source-label policy; select on dev only.
           Calibration/test are untouched.

The multimodal backbone remains frozen in this first study. Weights, media,
features, and checkpoints stay in Modal Volumes.
"""

import json

import modal


app = modal.App("vl-jev-omni-sports-head-training")
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.10.0", "torchvision==0.25.0", "transformers==5.17.0",
        "accelerate==1.13.0", "huggingface_hub==1.33.0", "safetensors==0.8.0",
        "numpy==2.4.6", "pillow==12.1.1", "opencv-python-headless==5.0.0.93",
        "soundfile==0.14.0", "librosa==0.11.0",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
data_volume = modal.Volume.from_name("vl-jev-pilot-data")
media_volume = modal.Volume.from_name("vl-jev-pilot-media")
probe_media = modal.Volume.from_name("vl-jev-probe-media")
model_cache = modal.Volume.from_name("vl-jev-zero-shot-hf-cache")
training_volume = modal.Volume.from_name("vl-jev-omni-sports-training", create_if_missing=True)
general_data = modal.Volume.from_name("vl-jev-general-v1-data", create_if_missing=True)
general_media = modal.Volume.from_name("vl-jev-general-v1-media", create_if_missing=True)
general_training = modal.Volume.from_name("vl-jev-general-v1-training", create_if_missing=True)
ucf_media = modal.Volume.from_name("vl-jev-ucf101-full-media", create_if_missing=True)
MODEL_ID = "akhilaaa3/Jev-Omni"
MODEL_REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
FAMILIES = ("sport_event", "shot_result", "action_classification")


def _families(value):
    selected = tuple(name.strip() for name in value.split(",") if name.strip())
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(FAMILIES):
        raise ValueError(f"Choose one or more of {FAMILIES}")
    return selected


def _options(row):
    question = row["question"]
    kind = question["type"]
    if kind == "choice":
        labels = list(question["criteria"])
        descriptions = [question["criteria"][key] for key in labels]
    elif kind == "noul":
        labels = ["false", "true"]
        descriptions = [question["criteria"][key] for key in labels]
    elif kind == "score":
        labels = [str(index) for index in range(len(question["criteria"]))]
        descriptions = question["criteria"]
    else:
        raise ValueError(f"Unsupported question type {kind}")
    if not 2 <= len(labels) <= 10 or row["gold"]["key"] not in labels:
        raise ValueError(f"Invalid options/gold for {row['id']}")
    return labels, descriptions


def _load_rows(families):
    from pathlib import Path

    root = Path("/dataset/pilot_v1")
    result = {}
    for split in ("train", "dev"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result[split] = [row for row in rows if row["task_family"] in families]
    return result


def _reviewed_rows(families):
    import hashlib
    from collections import Counter, defaultdict
    from pathlib import Path

    rows_by_split = _load_rows(families)
    review_path = Path("/runs/sports_review.jsonl")
    reviews = {}
    if review_path.is_file():
        for line in review_path.read_text().splitlines():
            entry = json.loads(line)
            if entry["id"] in reviews:
                raise ValueError(f"Duplicate review for {entry['id']}")
            reviews[entry["id"]] = entry
    approved = {"train": [], "dev": []}
    counts = {}
    problems = []
    group_splits = defaultdict(set)
    for split, rows in rows_by_split.items():
        status = Counter()
        for row in rows:
            group_splits[(row["source"]["repo"], row["source"]["group"])].add(split)
            media_file = Path("/pilot") / split / row["id"] / "media.mp4"
            review = reviews.get(row["id"])
            if not media_file.is_file():
                status["missing_media"] += 1
                continue
            if not review or review.get("review_status") != "approved":
                status["unreviewed"] += 1
                continue
            if review.get("split") != split or not review.get("reviewer"):
                problems.append(f"{row['id']}: missing split/reviewer")
                continue
            if review.get("visible_in_sampled_frames") is not True:
                status["not_visible"] += 1
                continue
            if review.get("frame_policy") != "16_midpoint_uniform_v1":
                problems.append(f"{row['id']}: unexpected frame policy")
                continue
            expected_hash = review.get("media_sha256")
            actual_hash = hashlib.sha256(media_file.read_bytes()).hexdigest()
            if expected_hash != actual_hash:
                problems.append(f"{row['id']}: media hash changed")
                continue
            gold = review.get("gold_key")
            labels, _ = _options(row)
            if gold not in labels:
                problems.append(f"{row['id']}: gold key is outside options")
                continue
            row = {**row, "gold": {"key": gold}}
            approved[split].append(row)
            status["approved"] += 1
        counts[split] = {"total": len(rows), **dict(status),
                         "approved_by_family": dict(Counter(row["task_family"]
                                                            for row in approved[split]))}
    leaking = [key for key, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        problems.append(f"{len(leaking)} source groups cross train/dev")
    ready = (not problems and len(approved["train"]) >= 20 and
             len(approved["dev"]) >= 5 and
             all(any(row["task_family"] == family for row in approved["train"])
                 and any(row["task_family"] == family for row in approved["dev"])
                 for family in families))
    return {"ready": ready, "families": families, "counts": counts,
            "problems": problems, "approved": approved,
            "review_path": str(review_path)}


def _source_label_rows(families):
    import hashlib
    from collections import Counter, defaultdict
    from pathlib import Path

    rows_by_split = _load_rows(families)
    selected = {"train": [], "dev": []}
    counts = {}
    problems = []
    group_splits = defaultdict(set)
    for split, rows in rows_by_split.items():
        status = Counter()
        for row in rows:
            group_splits[(row["source"]["repo"], row["source"]["group"])].add(split)
            folder = Path("/pilot") / split / row["id"]
            media_file, record_file = folder / "media.mp4", folder / "row.json"
            if not media_file.is_file() or not record_file.is_file():
                status["missing_media"] += 1
                continue
            record = json.loads(record_file.read_text())
            if record["id"] != row["id"] or record["source"] != row["source"]:
                problems.append(f"{row['id']}: materialized source does not match manifest")
                continue
            if media_file.stat().st_size == 0:
                problems.append(f"{row['id']}: empty media")
                continue
            if row.get("usage") != "research_only":
                problems.append(f"{row['id']}: unexpected usage policy")
                continue
            _options(row)
            selected[split].append(row)
            status["materialized_source_label"] += 1
        counts[split] = {"total": len(rows), **dict(status),
                         "selected_by_family": dict(Counter(row["task_family"]
                                                        for row in selected[split]))}
    leaking = [key for key, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        problems.append(f"{len(leaking)} source groups cross train/dev")
    ready = (not problems and len(selected["train"]) >= 20 and
             len(selected["dev"]) >= 5 and
             all(any(row["task_family"] == family for row in selected["train"])
                 and any(row["task_family"] == family for row in selected["dev"])
                 for family in families))
    fingerprint = hashlib.sha256(json.dumps(
        [{"split": split, "id": row["id"], "source": row["source"],
          "gold": row["gold"], "label_quality": row["label_quality"]}
         for split in ("train", "dev") for row in selected[split]],
        sort_keys=True).encode()).hexdigest()
    return {"ready": ready, "families": families, "counts": counts,
            "problems": problems, "approved": selected,
            "data_sha256": fingerprint}


@app.function(image=base_image, volumes={"/dataset": data_volume,
                                          "/pilot": media_volume,
                                          "/runs": training_volume}, timeout=600)
def inspect(families):
    import hashlib
    from pathlib import Path

    audit = _reviewed_rows(families)
    root = Path("/runs")
    root.mkdir(exist_ok=True)
    candidates = []
    for split, rows in _load_rows(families).items():
        for row in rows:
            folder = Path("/pilot") / split / row["id"]
            path = folder / "media.mp4"
            materialized = (folder / "row.json").is_file() and path.is_file()
            candidates.append({
                "id": row["id"], "split": split,
                "task_family": row["task_family"],
                "question_type": row["question"]["type"],
                "source": row["source"], "source_gold_key": row["gold"]["key"],
                "options": row["question"]["criteria"],
                "label_quality": row["label_quality"], "usage": row["usage"],
                "materialized": materialized,
                "media_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                if materialized else None,
                "review_status": "unreviewed",
            })
    output = root / "sports_candidates_train_dev.jsonl"
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in candidates) + "\n")
    training_volume.commit()
    return {"candidate_rows": len(candidates), "candidate_manifest": str(output),
            "ready": audit["ready"], "counts": audit["counts"],
            "problems": audit["problems"], "review_path": audit["review_path"],
            "required_review_fields": ["id", "split", "review_status", "reviewer",
                                       "visible_in_sampled_frames", "frame_policy",
                                       "media_sha256", "gold_key"]}


def _load_classifier():
    import sys
    from pathlib import Path

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    patterns = ["config.json", "generation_config.json", "model*.safetensors*",
                "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "decision_config.json", "head.pt"]
    checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                   allow_patterns=patterns)
    source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(source).parent))
    import jev_omni

    jev_omni.snapshot_download = lambda *_args, **_kwargs: checkpoint
    classifier = jev_omni.load_jev_omni()
    classifier.model.eval().requires_grad_(False)
    return torch, classifier, jev_omni


def _feature(torch, classifier, package, path, state, question, options):
    from PIL import Image

    # Match the published loader's 16-frame preprocessing and prompt exactly.
    frames = package._video_frames(path, 16)
    images = [Image.fromarray(frame).convert("RGB") if not isinstance(frame, Image.Image)
              else frame for frame in frames]
    return _feature_content(torch, classifier, package, images, state, question, options)


def _feature_content(torch, classifier, package, images, state, question, options):
    content = [{"type": "image", "image": frame} for frame in images]
    content.append({"type": "text", "text": package._prompt(state, question, options)})
    inputs = classifier.processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    inputs = {key: value.to("cuda", dtype=torch.bfloat16)
              if torch.is_floating_point(value) else value.to("cuda")
              for key, value in inputs.items()}
    classifier._capture.clear()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        classifier.model(**inputs, use_cache=False, **classifier._extra)
    # Clone outside inference_mode so the head can save this input for its
    # weight-gradient calculation during training.
    return classifier._capture["hidden"].detach().float().cpu()[0].clone()


def _general_feature(torch, classifier, package, row):
    from pathlib import Path
    from PIL import Image

    labels, options = _options(row)
    modality = row["modality"]
    media = row.get("media") or {}
    if modality == "text":
        images = []
    elif media.get("kind") == "modal_media_sequence":
        assets = media["assets"]
        if not assets:
            raise ValueError(f"Empty media sequence for {row['id']}")
        if len(assets) > 16:
            assets = [assets[round(index * (len(assets) - 1) / 15)]
                      for index in range(16)]
        images = []
        frame_budget = max(1, 16 // len(assets))
        for asset in assets:
            path = Path(asset["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if asset["kind"] == "image":
                images.append(Image.open(path).convert("RGB"))
            elif asset["kind"] == "video":
                images.extend(Image.fromarray(frame).convert("RGB")
                              if not isinstance(frame, Image.Image) else frame
                              for frame in package._video_frames(str(path), frame_budget))
            else:
                raise ValueError(f"Unsupported media asset: {asset['kind']}")
        images = images[:16]
    elif media.get("kind") == "modal_image":
        path = Path(media["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        images = [Image.open(path).convert("RGB")]
    elif modality == "video" and media.get("kind") == "modal_video":
        path = Path(media["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return _feature(torch, classifier, package, str(path), row["state"],
                        row["question"]["instructions"], options)
    else:
        folder = Path("/pilot") / row["split"] / row["id"]
        if modality == "video":
            path = folder / "media.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            return _feature(torch, classifier, package, str(path), row["state"],
                            row["question"]["instructions"], options)
        if modality not in {"image", "image_sequence"}:
            raise ValueError(f"Unsupported modality for {row['id']}: {modality}")
        paths = sorted(folder.glob("image_*.jpg"))
        if modality == "image":
            paths = paths[:1]
        if not paths:
            raise FileNotFoundError(folder / "image_0.jpg")
        images = [Image.open(path).convert("RGB") for path in paths[:16]]
    return _feature_content(torch, classifier, package, images, row["state"],
                            row["question"]["instructions"], options)


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/dataset": data_volume, "/pilot": media_volume,
                       "/general-media": general_media, "/model-cache": model_cache})
def general_smoke():
    from pathlib import Path

    rows = [json.loads(line) for line in
            (Path("/dataset/pilot_v1") / "train.jsonl").read_text().splitlines()]
    selected = {}
    for row in rows:
        modality = row["modality"]
        if modality in selected:
            continue
        if modality != "text" and not (Path("/pilot/train") / row["id"] / "row.json").is_file():
            continue
        try:
            _options(row)
        except ValueError:
            continue
        selected[modality] = row
    torch, classifier, package = _load_classifier()
    report = []
    for modality, row in selected.items():
        feature = _general_feature(torch, classifier, package, row)
        labels, _ = _options(row)
        with torch.no_grad():
            logits = classifier.head(
                feature.to("cuda").unsqueeze(0),
                torch.tensor([len(labels)], device="cuda"))[0, :len(labels)]
        report.append({"modality": modality, "id": row["id"],
                       "family": row["task_family"], "options": len(labels),
                       "feature_size": feature.numel(),
                       "finite": bool(torch.isfinite(feature).all()),
                       "gold_slot": labels.index(row["gold"]["key"]),
                       "predicted_slot": int(logits.argmax())})
    return {"model_revision": MODEL_REVISION, "cases": report}


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/general-data": general_data, "/general-media": general_media,
                       "/model-cache": model_cache})
def general_source_smoke():
    from pathlib import Path

    selected = []
    for name in ("aokvqa-train.jsonl", "scienceqa-train.jsonl",
                 "webintosh-dev.jsonl"):
        path = Path("/general-data/v1/parts") / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                if row["modality"] == "image":
                    selected.append(row)
                    break
    torch, classifier, package = _load_classifier()
    results = []
    for row in selected:
        feature = _general_feature(torch, classifier, package, row)
        labels, _ = _options(row)
        with torch.no_grad():
            logits = classifier.head(
                feature.to("cuda").unsqueeze(0),
                torch.tensor([len(labels)], device="cuda"))[0, :len(labels)]
        results.append({"repo": row["source"]["repo"],
                        "family": row["task_family"],
                        "options": len(labels), "feature_size": feature.numel(),
                        "finite": bool(torch.isfinite(feature).all()),
                        "gold_slot": labels.index(row["gold"]["key"]),
                        "predicted_slot": int(logits.argmax())})
    return {"model_revision": MODEL_REVISION, "cases": results}


def _general_rows(split, dataset_version="v1"):
    from pathlib import Path

    if dataset_version not in {"v1", "v2"}:
        raise ValueError("Unsupported dataset version")
    path = Path("/general-data") / dataset_version / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines()]


def _feature_shard(row_id, shards):
    import hashlib

    return int.from_bytes(hashlib.sha256(row_id.encode()).digest()[:8], "big") % shards


PACKED_FEATURE_ROWS = 64


def _packed_expected_count(split, shards, dataset_version):
    counts = [0] * shards
    for row in _general_rows(split, dataset_version):
        counts[_feature_shard(row["id"], shards)] += 1
    return sum((count + PACKED_FEATURE_ROWS - 1) // PACKED_FEATURE_ROWS
               for count in counts)


def _decision_hash(row):
    import hashlib

    payload = {"state": row["state"], "question": row["question"],
               "media": row.get("media"), "gold": row["gold"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@app.function(image=base_image, gpu="H100", timeout=86400,
              volumes={"/general-data": general_data, "/general-media": general_media,
                       "/pilot": media_volume, "/model-cache": model_cache,
                       "/ucf-media": ucf_media,
                       "/general-runs": general_training})
def extract_general_shard(run_name, shard, shards, splits=("train", "dev"),
                          dataset_version="v1"):
    from pathlib import Path
    import torch

    if not run_name or "/" in run_name or not 0 <= shard < shards <= 16:
        raise ValueError("Invalid general run/shard")
    root = Path("/general-runs") / run_name / "features"
    selected = {split: [row for row in _general_rows(split, dataset_version)
                        if _feature_shard(row["id"], shards) == shard]
                for split in splits}
    counts = {}
    torch, classifier, package = _load_classifier()
    for split, rows in selected.items():
        status = {"rows": len(rows), "created": 0, "reused": 0}
        if dataset_version == "v2":
            # V2 writes 64 decision records per file. Modal Volumes slow down
            # when tens of thousands of individual feature files accumulate.
            for offset in range(0, len(rows), PACKED_FEATURE_ROWS):
                chunk = rows[offset:offset + PACKED_FEATURE_ROWS]
                pack_index = offset // PACKED_FEATURE_ROWS
                destination = root / split / f"pack-{shard:02d}-{pack_index:05d}.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file():
                    pack = torch.load(destination, map_location="cpu", weights_only=True)
                    saved = pack.get("rows", [])
                    if (pack.get("dataset_version") != dataset_version or
                            pack.get("shard") != shard or
                            pack.get("shards") != shards or len(saved) != len(chunk) or
                            any(item["id"] != row["id"] or
                                item["decision_hash"] != _decision_hash(row)
                                for item, row in zip(saved, chunk))):
                        raise ValueError(f"Packed feature cache mismatch: {destination}")
                    status["reused"] += len(chunk)
                else:
                    saved = []
                    for row in chunk:
                        labels, _ = _options(row)
                        feature = _general_feature(torch, classifier, package, row)
                        if not bool(torch.isfinite(feature).all()):
                            raise ValueError(f"Nonfinite feature: {row['id']}")
                        saved.append({"id": row["id"], "feature": feature.half(),
                                      "target": labels.index(row["gold"]["key"]),
                                      "count": len(labels), "modality": row["modality"],
                                      "family": row["task_family"],
                                      "decision_hash": _decision_hash(row),
                                      "source_repo": row["source"]["repo"]})
                    torch.save({"dataset_version": dataset_version, "shard": shard,
                                "shards": shards, "rows": saved}, destination)
                    status["created"] += len(chunk)
                general_training.commit()
                print(json.dumps({"shard": shard, "split": split,
                                  "packs_done": pack_index + 1,
                                  "packs_total": (len(rows) + PACKED_FEATURE_ROWS - 1) //
                                                 PACKED_FEATURE_ROWS,
                                  **status}), flush=True)
            counts[split] = status
            continue
        for row in rows:
            labels, _ = _options(row)
            target = labels.index(row["gold"]["key"])
            destination = root / split / f"{row['id']}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                record = torch.load(destination, map_location="cpu", weights_only=True)
                if (record["id"] != row["id"] or record["target"] != target or
                        record["count"] != len(labels) or
                        record["modality"] != row["modality"] or
                        record["family"] != row["task_family"] or
                        record["decision_hash"] != _decision_hash(row)):
                    raise ValueError(f"Feature cache mismatch: {destination}")
                status["reused"] += 1
            else:
                feature = _general_feature(torch, classifier, package, row)
                if not bool(torch.isfinite(feature).all()):
                    raise ValueError(f"Nonfinite feature: {row['id']}")
                torch.save({"id": row["id"], "feature": feature.half(),
                            "target": target, "count": len(labels),
                            "modality": row["modality"],
                            "family": row["task_family"],
                            "decision_hash": _decision_hash(row),
                            "source_repo": row["source"]["repo"]}, destination)
                status["created"] += 1
            if (status["created"] + status["reused"]) % 100 == 0:
                general_training.commit()
                print(json.dumps({"shard": shard, "split": split, **status}), flush=True)
        general_training.commit()
        counts[split] = status
    return {"run": run_name, "dataset_version": dataset_version,
            "shard": shard, "shards": shards,
            "model_revision": MODEL_REVISION, "counts": counts}


def _load_general_features(torch, root, split, dataset_version="v1"):
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from pathlib import Path

    expected = {row["id"]: row for row in _general_rows(split, dataset_version)}
    found = {}
    pattern = "pack-*.pt" if dataset_version == "v2" else "*.pt"
    paths = list((Path(root) / "features" / split).glob(pattern))

    def load_item(path):
        return path, torch.load(path, map_location="cpu", weights_only=True)

    # Keep only a bounded number of outstanding Volume reads. Executor.map()
    # eagerly schedules every input on Python 3.11, which can retain thousands
    # of loaded tensors while one slow remote file blocks ordered delivery.
    with ThreadPoolExecutor(max_workers=8) as pool:
        remaining = iter(paths)
        pending = set()
        for _ in range(min(32, len(paths))):
            pending.add(pool.submit(load_item, next(remaining)))
        index = 0
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                path, payload = future.result()
                index += 1
                next_path = next(remaining, None)
                if next_path is not None:
                    pending.add(pool.submit(load_item, next_path))
                if dataset_version == "v2" and payload.get("dataset_version") != "v2":
                    raise ValueError(f"Invalid feature pack: {path}")
                records = payload["rows"] if dataset_version == "v2" else [payload]
                for item in records:
                    row = expected.get(item["id"])
                    if row is None or item["id"] in found:
                        raise ValueError(f"Unexpected or duplicated feature: {path}")
                    labels, _ = _options(row)
                    if (item["target"] != labels.index(row["gold"]["key"]) or
                            item["count"] != len(labels) or
                            item["modality"] != row["modality"] or
                            item["family"] != row["task_family"] or
                            item["decision_hash"] != _decision_hash(row) or
                            item["source_repo"] != row["source"]["repo"]):
                        raise ValueError(f"Feature metadata mismatch: {path}")
                    found[item["id"]] = item
                if index % (10 if dataset_version == "v2" else 1000) == 0:
                    print(json.dumps({"phase": "loading_features", "split": split,
                                      "files_loaded": index, "rows_loaded": len(found),
                                      "expected_rows": len(expected)}), flush=True)
    if set(found) != set(expected):
        missing = list(set(expected) - set(found))[:10]
        raise ValueError(f"Incomplete {split} features: {len(found)}/{len(expected)}; missing={missing}")
    return [found[row_id] for row_id in expected]


@app.function(image=base_image, timeout=600,
              volumes={"/general-data": general_data,
                       "/general-runs": general_training})
def general_feature_footprint(run_name, dataset_version="v1"):
    """Measure cached feature size in Modal without transferring feature data."""
    from pathlib import Path
    import torch

    root = Path("/general-runs") / run_name / "features"
    results = {}
    for split in ("train", "dev"):
        sample = next((root / split).glob("*.pt"), None)
        if sample is None:
            results[split] = {"status": "missing"}
            continue
        payload = torch.load(sample, map_location="cpu", weights_only=True)
        item = payload["rows"][0] if "rows" in payload else payload
        expected = len(_general_rows(split, dataset_version))
        tensor = item["feature"]
        results[split] = {"expected_rows": expected,
                          "feature_shape": list(tensor.shape),
                          "feature_dtype": str(tensor.dtype),
                          "feature_bytes_per_row": tensor.numel() * tensor.element_size(),
                          "all_feature_tensor_mib": round(
                              expected * tensor.numel() * tensor.element_size() / 1048576, 2),
                          "sample_file_bytes": sample.stat().st_size}
    return results


def _general_metrics(torch, head, items):
    from collections import defaultdict

    groups = defaultdict(lambda: {"n": 0, "correct": 0, "loss": 0.0, "brier": 0.0})
    head.eval()
    with torch.no_grad():
        for start in range(0, len(items), 128):
            batch = items[start:start + 128]
            features = torch.stack([item["feature"] for item in batch]).to("cuda").float()
            counts = torch.tensor([item["count"] for item in batch], device="cuda")
            logits = head(features, counts)
            for index, item in enumerate(batch):
                count, target = item["count"], item["target"]
                probs = logits[index, :count].softmax(-1)
                loss = -probs[target].clamp_min(1e-9).log().item()
                brier = sum((probs[j].item() - (j == target)) ** 2
                            for j in range(count))
                correct = int(probs.argmax().item() == target)
                broad = "text" if item["modality"] == "text" else "vision"
                for group in ("overall", broad, f"modality:{item['modality']}",
                              f"family:{item['family']}",
                              f"family_modality:{item['family']}:{item['modality']}",
                              f"source:{item['source_repo']}"):
                    values = groups[group]
                    values["n"] += 1
                    values["correct"] += correct
                    values["loss"] += loss
                    values["brier"] += brier
    result = {key: {"n": value["n"], "correct": value["correct"],
                  "accuracy": value["correct"] / value["n"],
                  "log_loss": value["loss"] / value["n"],
                  "brier": value["brier"] / value["n"]}
            for key, value in groups.items()}
    visual_families = [value["accuracy"] for key, value in result.items()
                       if key.startswith("family_modality:") and
                       not key.endswith(":text")]
    result["vision_macro"] = {
        "families": len(visual_families),
        "accuracy": sum(visual_families) / len(visual_families)
        if visual_families else None}
    return result


@app.function(image=base_image, gpu="H100", timeout=86400,
              volumes={"/general-data": general_data, "/general-runs": general_training,
                       "/model-cache": model_cache})
def train_general_head(run_name: str, epochs: int = 3,
                       weighting: str = "uniform", dataset_version: str = "v1"):
    import copy
    import hashlib
    import random
    from collections import Counter
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download

    if not run_name or "/" in run_name or not 1 <= epochs <= 8:
        raise ValueError("Invalid run or epoch count")
    if weighting not in {"uniform", "sqrt_inverse_source"}:
        raise ValueError("Invalid weighting")
    root = Path("/general-runs") / run_name
    dataset_summary = json.loads((Path("/general-data") / dataset_version / "summary.json").read_text())
    for split in ("train", "dev"):
        actual = hashlib.sha256((Path("/general-data") / dataset_version /
                                 f"{split}.jsonl").read_bytes()).hexdigest()
        if actual != dataset_summary["split_sha256"][split]:
            raise ValueError(f"Dataset split hash changed: {split}")
    train = _load_general_features(torch, root, "train", dataset_version)
    dev = _load_general_features(torch, root, "dev", dataset_version)
    import sys
    source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(source).parent))
    import jev_omni
    head = jev_omni._Head256(train[0]["feature"].numel()).to("cuda")
    published = hf_hub_download(MODEL_ID, "head.pt", revision=MODEL_REVISION)
    head.load_state_dict(torch.load(published, map_location="cuda", weights_only=True))
    baseline = _general_metrics(torch, head, dev)
    source_counts = Counter(item["source_repo"] for item in train)
    source_weights = {source: (len(train) / (len(source_counts) * count)) ** 0.5
                      for source, count in source_counts.items()}
    optimum = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=0.01)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        head.train()
        order = list(range(len(train)))
        random.Random(3407 + epoch).shuffle(order)
        for start in range(0, len(order), 128):
            batch = [train[index] for index in order[start:start + 128]]
            features = torch.stack([item["feature"] for item in batch]).to("cuda").float()
            counts = torch.tensor([item["count"] for item in batch], device="cuda")
            targets = torch.tensor([item["target"] for item in batch], device="cuda")
            logits = head(features, counts)
            losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            if weighting == "sqrt_inverse_source":
                weights = torch.tensor([source_weights[item["source_repo"]] for item in batch],
                                       device="cuda")
                loss = (losses * weights).sum() / weights.sum()
            else:
                loss = losses.mean()
            optimum.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimum.step()
        metrics = _general_metrics(torch, head, dev)
        history.append({"epoch": epoch + 1, "dev": metrics})
        value = metrics["overall"]["log_loss"]
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(head.state_dict())
        print(json.dumps({"epoch": epoch + 1, "weighting": weighting,
                          "dev_overall": metrics["overall"]}), flush=True)
    head.load_state_dict(best_state)
    checkpoint = root / f"head-{weighting}.pt"
    torch.save(best_state, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    chosen = _general_metrics(torch, head, dev)
    report = {"run": run_name, "dataset_version": dataset_version,
              "dataset_split_sha256": dataset_summary["split_sha256"],
              "method": "frozen_backbone_head",
              "weighting": weighting, "epochs": epochs,
              "model": MODEL_ID, "revision": MODEL_REVISION,
              "train_rows": len(train), "dev_rows": len(dev),
              "train_by_source": dict(source_counts),
              "baseline_dev": baseline, "history": history,
              "selected_dev": chosen, "checkpoint": str(checkpoint),
              "checkpoint_sha256": digest,
              "untouched_splits": ["calibration", "test"],
              "usage": "research_only"}
    (root / f"report-{weighting}.json").write_text(json.dumps(report, indent=2))
    general_training.commit()
    return report


@app.function(image=base_image, gpu="H100", timeout=86400,
              volumes={"/general-data": general_data, "/general-runs": general_training,
                       "/model-cache": model_cache})
def evaluate_general_head(run_name, weighting, split, dataset_version="v1"):
    import hashlib
    from pathlib import Path
    import sys
    import torch
    from huggingface_hub import hf_hub_download

    if split not in {"calibration", "test"}:
        raise ValueError("Evaluation split must be calibration or test")
    root = Path("/general-runs") / run_name
    report = json.loads((root / f"report-{weighting}.json").read_text())
    if report["dataset_version"] != dataset_version:
        raise ValueError("Checkpoint dataset version mismatch")
    actual = hashlib.sha256((Path("/general-data") / dataset_version /
                             f"{split}.jsonl").read_bytes()).hexdigest()
    if actual != report["dataset_split_sha256"][split]:
        raise ValueError(f"Held-out dataset split hash changed: {split}")
    checkpoint = Path(report["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash changed")
    items = _load_general_features(torch, root, split, dataset_version)
    source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
    sys.path.insert(0, str(Path(source).parent))
    import jev_omni
    head = jev_omni._Head256(items[0]["feature"].numel()).to("cuda")
    head.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    metrics = _general_metrics(torch, head, items)
    output = {"run": run_name, "dataset_version": dataset_version,
              "weighting": weighting, "split": split,
              "rows": len(items), "checkpoint_sha256": report["checkpoint_sha256"],
              "metrics": metrics}
    destination = root / f"evaluation-{split}-{weighting}.json"
    if destination.is_file():
        previous = json.loads(destination.read_text())
        if previous != output:
            raise ValueError("Immutable evaluation result changed")
    else:
        destination.write_text(json.dumps(output, indent=2))
        general_training.commit()
    return output


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              timeout=86400, volumes={"/general-data": general_data,
                                       "/general-runs": general_training})
def complete_general_head_pipeline(run_name: str, dataset_version: str = "v1",
                                   shards: int = 4, epochs: int = 3,
                                   poll_seconds: int = 60,
                                   wait_for_uniform: bool = False):
    """Wait for full features, train both heads, select on dev, score held-out."""
    import hashlib
    import time
    from pathlib import Path

    if not run_name or "/" in run_name or dataset_version not in {"v1", "v2"}:
        raise ValueError("Invalid run or dataset version")
    if not 1 <= shards <= 16 or not 1 <= epochs <= 8 or not 15 <= poll_seconds <= 300:
        raise ValueError("Invalid pipeline settings")
    data_root = Path("/general-data") / dataset_version
    summary = json.loads((data_root / "summary.json").read_text())
    run_root = Path("/general-runs") / run_name
    for split in ("train", "dev", "calibration", "test"):
        actual = hashlib.sha256((data_root / f"{split}.jsonl").read_bytes()).hexdigest()
        if actual != summary["split_sha256"][split]:
            raise ValueError(f"Unfrozen dataset split: {split}")
    expected = {split: (_packed_expected_count(split, shards, dataset_version)
                        if dataset_version == "v2" else summary["counts"][split])
                for split in ("train", "dev")}
    deadline = time.monotonic() + 20 * 3600
    while time.monotonic() < deadline:
        general_training.reload()
        pattern = "pack-*.pt" if dataset_version == "v2" else "*.pt"
        counts = {split: sum(1 for _ in (run_root / "features" / split).glob(pattern))
                  for split in expected}
        print(json.dumps({"phase": "waiting_for_features", "run": run_name,
                          "counts": counts, "expected": expected}), flush=True)
        if any(counts[split] > expected[split] for split in expected):
            raise ValueError("Unexpected excess feature files")
        if counts == expected:
            break
        time.sleep(poll_seconds)
    else:
        raise TimeoutError(f"Feature extraction incomplete after 20h: {counts}/{expected}")
    if wait_for_uniform:
        uniform_report = run_root / "report-uniform.json"
        deadline = time.monotonic() + 20 * 3600
        while time.monotonic() < deadline:
            general_training.reload()
            if uniform_report.is_file():
                break
            print(json.dumps({"phase": "waiting_for_uniform_report", "run": run_name}),
                  flush=True)
            time.sleep(poll_seconds)
        else:
            raise TimeoutError("Uniform training did not finish within 20h")
    reports = {}
    for weighting in ("uniform", "sqrt_inverse_source"):
        report_path = run_root / f"report-{weighting}.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            checkpoint = run_root / f"head-{weighting}.pt"
            if (report["run"] != run_name or
                    report["dataset_version"] != dataset_version or
                    report["epochs"] != epochs or
                    report["train_rows"] != summary["counts"]["train"] or
                    report["dev_rows"] != summary["counts"]["dev"] or
                    report["dataset_split_sha256"] != summary["split_sha256"] or
                    not checkpoint.is_file() or
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest() !=
                    report["checkpoint_sha256"]):
                raise ValueError(f"Saved training report is invalid: {report_path}")
            reports[weighting] = report
            print(json.dumps({"phase": "reused_training_report",
                              "weighting": weighting}), flush=True)
        else:
            reports[weighting] = train_general_head.remote(run_name, epochs, weighting,
                                                            dataset_version)
        general_training.reload()
    baseline_text = reports["uniform"]["baseline_dev"].get("text", {})
    base_text_accuracy = baseline_text.get("accuracy", 0)
    base_text_loss = baseline_text.get("log_loss", float("inf"))

    def selection_key(weighting):
        metrics = reports[weighting]["selected_dev"]
        vision = metrics.get("vision_macro", {}).get("accuracy") or 0
        text_metrics = metrics.get("text", {})
        text_ok = (text_metrics.get("accuracy", 0) >= base_text_accuracy - 0.02 and
                   text_metrics.get("log_loss", float("inf")) <= base_text_loss + 0.05)
        return (int(text_ok), vision, -metrics["overall"]["log_loss"])

    chosen = max(reports, key=selection_key)
    selection = {"run": run_name, "dataset_version": dataset_version,
                 "selected_weighting": chosen,
                 "selection_policy": "text accuracy <=2 percentage points below base and text log loss <=base+0.05; then dev equal-family vision accuracy, then overall dev log loss",
                 "baseline_text": baseline_text,
                 "dev": {name: report["selected_dev"] for name, report in reports.items()},
                 "text_guard_passed": bool(selection_key(chosen)[0]),
                 "checkpoint_sha256": reports[chosen]["checkpoint_sha256"]}
    (run_root / "selection.json").write_text(json.dumps(selection, indent=2))
    general_training.commit()
    heldout = tuple(("calibration", "test"))
    extracted = list(extract_general_shard.map(
        [run_name] * shards, range(shards), [shards] * shards,
        [heldout] * shards, [dataset_version] * shards))
    general_training.reload()
    calibration = evaluate_general_head.remote(run_name, chosen, "calibration",
                                               dataset_version)
    test = evaluate_general_head.remote(run_name, chosen, "test", dataset_version)
    output = {"run": run_name, "dataset_version": dataset_version,
              "selected_weighting": chosen, "selection": selection,
              "heldout_extraction": extracted,
              "calibration": calibration, "test": test}
    (run_root / "pipeline-result.json").write_text(json.dumps(output, indent=2))
    general_training.commit()
    return {"run": run_name, "dataset_version": dataset_version,
            "train_rows": summary["counts"]["train"],
            "dev_rows": summary["counts"]["dev"],
            "selected_weighting": chosen, "checkpoint_sha256": selection["checkpoint_sha256"],
            "test_vision": test["metrics"].get("vision"),
            "test_vision_macro": test["metrics"].get("vision_macro"),
            "test_text": test["metrics"].get("text"),
            "result_path": str(run_root / "pipeline-result.json")}


def _orders(row):
    labels, descriptions = _options(row)
    if row["question"]["type"] == "choice":
        # All three positions for football; two diverse positions for larger
        # action menus. Noul and Score retain their semantic order.
        rotations = range(len(labels)) if len(labels) == 3 else (0, len(labels) // 2)
    else:
        rotations = (0,)
    seen = set()
    for rotation in rotations:
        indices = tuple(list(range(len(labels)))[rotation:] +
                        list(range(len(labels)))[:rotation])
        if indices in seen:
            continue
        seen.add(indices)
        ordered_labels = [labels[index] for index in indices]
        ordered_options = [descriptions[index] for index in indices]
        yield rotation, ordered_options, ordered_labels.index(row["gold"]["key"]), len(labels)


def _metrics(torch, head, items):
    from collections import defaultdict

    sums = defaultdict(lambda: {"n": 0, "correct": 0, "loss": 0.0})
    head.eval()
    with torch.no_grad():
        for item in items:
            feature = item["feature"].to("cuda").float().unsqueeze(0)
            count = torch.tensor([item["count"]], device="cuda")
            logits = head(feature, count)[0, :item["count"]]
            target = torch.tensor([item["target"]], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
            record = sums[item["family"]]
            record["n"] += 1
            record["correct"] += int(logits.argmax().item() == item["target"])
            record["loss"] += loss.item()
    return {family: {"n": values["n"],
                     "accuracy": values["correct"] / values["n"],
                     "log_loss": values["loss"] / values["n"]}
            for family, values in sums.items()}


def _extract(torch, classifier, package, rows, split, feature_root, limit=0):
    from pathlib import Path

    items = []
    selected = rows[:limit] if limit else rows
    for row in selected:
        path = Path("/pilot") / split / row["id"] / "media.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        for rotation, options, target, count in _orders(row):
            destination = feature_root / split / f"{row['id']}-{rotation}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                data = torch.load(destination, map_location="cpu", weights_only=True)
                feature = data["feature"]
                if data["target"] != target or data["count"] != count:
                    raise ValueError(f"Cached feature label mismatch: {destination}")
            else:
                feature = _feature(torch, classifier, package, str(path), row["state"],
                                   row["question"]["instructions"], options)
                torch.save({"feature": feature.half(), "target": target,
                            "count": count}, destination)
            items.append({"feature": feature, "target": target, "count": count,
                          "family": row["task_family"], "id": row["id"]})
        if len(items) % 48 == 0:
            training_volume.commit()
    training_volume.commit()
    return items


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/model-cache": model_cache, "/probe-media": probe_media,
                       "/runs": training_volume})
def smoke():
    from pathlib import Path

    torch, classifier, package = _load_classifier()
    cases = [("goal", "omni_goal.mp4"), ("miss", "omni_miss.mp4")]
    features = []
    for expected, filename in cases:
        path = Path("/probe-media") / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        feature = _feature(torch, classifier, package, str(path), "Soccer scene.",
                           "Did the ball enter the goal?",
                           ["The ball enters the goal.", "The ball misses the goal."])
        features.append(feature)
    head = classifier.head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    vectors = torch.stack(features).to("cuda")
    targets = torch.tensor([0, 1], device="cuda")
    counts = torch.tensor([2, 2], device="cuda")
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        logits = head(vectors, counts)[:, :2]
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(round(loss.item(), 5))
    if not losses[-1] < losses[0]:
        raise RuntimeError("Sports head smoke did not reduce loss")
    destination = Path("/runs/smoke")
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), destination / "head.pt")
    training_volume.commit()
    return {"mode": "synthetic_smoke", "model_revision": MODEL_REVISION,
            "loss_first": losses[0], "loss_last": losses[-1],
            "checkpoint": str(destination / "head.pt")}


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/dataset": data_volume, "/pilot": media_volume,
                       "/model-cache": model_cache})
def data_smoke(families):
    from pathlib import Path

    torch, classifier, package = _load_classifier()
    results = []
    for family in families:
        available = [row for row in _load_rows((family,))["train"]
                     if (Path("/pilot/train") / row["id"] / "media.mp4").is_file()]
        if not available:
            results.append({"family": family, "status": "no_materialized_train_row"})
            continue
        seen_types = set()
        for row in available:
            question_type = row["question"]["type"]
            if question_type in seen_types:
                continue
            seen_types.add(question_type)
            _, options, target, count = next(_orders(row))
            path = Path("/pilot/train") / row["id"] / "media.mp4"
            feature = _feature(torch, classifier, package, str(path), row["state"],
                               row["question"]["instructions"], options)
            with torch.no_grad():
                logits = classifier.head(feature.to("cuda").unsqueeze(0),
                                         torch.tensor([count], device="cuda"))[0, :count]
                probs = logits.softmax(-1).float().cpu().tolist()
            results.append({"family": family, "question_type": question_type,
                            "status": "ok", "row_id": row["id"],
                            "feature_size": feature.numel(), "option_count": count,
                            "gold_slot": target, "gold_probability": round(probs[target], 5)})
    return {"mode": "pilot_data_smoke", "results": results}


@app.function(image=base_image, gpu="H100", timeout=86400,
              volumes={"/dataset": data_volume, "/pilot": media_volume,
                       "/model-cache": model_cache, "/runs": training_volume})
def train_head(families, run_name, epochs=3, label_policy="reviewed"):
    import hashlib
    import random
    from pathlib import Path

    if not run_name or "/" in run_name or not 1 <= epochs <= 5:
        raise ValueError("Use a simple run name and 1–5 epochs")
    if label_policy not in {"reviewed", "source_labels"}:
        raise ValueError("label_policy must be reviewed or source_labels")
    audit = (_reviewed_rows(families) if label_policy == "reviewed"
             else _source_label_rows(families))
    if not audit["ready"]:
        raise ValueError(f"Materialized data for {label_policy} are insufficient: {audit['counts']}; "
                         f"problems={audit['problems']}")
    torch, classifier, package = _load_classifier()
    torch.manual_seed(3407)
    data_hash = (hashlib.sha256(Path(audit["review_path"]).read_bytes()).hexdigest()
                 if label_policy == "reviewed" else audit["data_sha256"])
    run_dir = Path("/runs") / run_name
    metadata = {"model_revision": MODEL_REVISION, "data_sha256": data_hash,
                "label_policy": label_policy,
                "families": list(families), "epochs": epochs,
                "frame_policy": "16_midpoint_uniform_v1"}
    meta_path = run_dir / "run_meta.json"
    if run_dir.exists():
        if not meta_path.is_file() or json.loads(meta_path.read_text()) != metadata:
            raise ValueError(f"Existing run name has different inputs: {run_dir}")
        if (run_dir / "report.json").is_file():
            return json.loads((run_dir / "report.json").read_text())
    else:
        run_dir.mkdir(parents=True)
        meta_path.write_text(json.dumps(metadata, indent=2))
        training_volume.commit()
    feature_root = run_dir / "features"
    train = _extract(torch, classifier, package, audit["approved"]["train"],
                     "train", feature_root)
    dev = _extract(torch, classifier, package, audit["approved"]["dev"],
                   "dev", feature_root)
    head = classifier.head
    baseline = _metrics(torch, head, dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=0.01)
    best_loss = float("inf")
    history = []
    start_epoch = 0
    state_path = run_dir / "last_epoch.pt"
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        best_loss = state["best_loss"]
        history = state["history"]
        start_epoch = state["epoch"]
    for epoch in range(start_epoch, epochs):
        head.train()
        order = list(range(len(train)))
        random.Random(3407 + epoch).shuffle(order)
        for start in range(0, len(order), 32):
            batch = [train[index] for index in order[start:start + 32]]
            vectors = torch.stack([item["feature"] for item in batch]).to("cuda").float()
            counts = torch.tensor([item["count"] for item in batch], device="cuda")
            targets = torch.tensor([item["target"] for item in batch], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = head(vectors, counts)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        dev_metrics = _metrics(torch, head, dev)
        mean_loss = sum(value["log_loss"] * value["n"]
                        for value in dev_metrics.values()) / len(dev)
        history.append({"epoch": epoch + 1, "dev": dev_metrics,
                        "dev_log_loss": mean_loss})
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(head.state_dict(), run_dir / "head.pt")
        torch.save({"epoch": epoch + 1, "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_loss": best_loss,
                    "history": history}, state_path)
        training_volume.commit()
    report = {"model": MODEL_ID, "revision": MODEL_REVISION,
              "method": "frozen_backbone_classifier_head_finetune",
              "families": families, "data_sha256": data_hash,
              "label_policy": label_policy, "usage": "research_only",
              "label_limitations": ("source labels have not been visually reviewed for sampled-frame visibility"
                                    if label_policy == "source_labels" else None),
              "train_rows": len(audit["approved"]["train"]),
              "dev_rows": len(audit["approved"]["dev"]),
              "train_features": len(train), "dev_features": len(dev),
              "baseline_dev": baseline, "history": history,
              "best_dev_log_loss": best_loss,
              "checkpoint": str(run_dir / "head.pt"),
              "untouched_splits": ["calibration", "test"]}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    training_volume.commit()
    return report


@app.function(image=base_image, volumes={"/runs": training_volume}, timeout=600)
def verify_run(run_name):
    import hashlib
    from pathlib import Path
    import torch

    run_dir = Path("/runs") / run_name
    report = json.loads((run_dir / "report.json").read_text())
    metadata = json.loads((run_dir / "run_meta.json").read_text())
    checkpoint = run_dir / "head.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not state or not all(torch.is_tensor(value) and torch.isfinite(value).all()
                            for value in state.values()):
        raise ValueError("Checkpoint contains missing or nonfinite weights")
    if (report["checkpoint"] != str(checkpoint) or
            report["data_sha256"] != metadata["data_sha256"] or
            len(report["history"]) != metadata["epochs"]):
        raise ValueError("Report, metadata, and checkpoint do not agree")
    return {"checkpoint": str(checkpoint), "bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "tensor_count": len(state), "train_rows": report["train_rows"],
            "dev_rows": report["dev_rows"], "label_policy": report["label_policy"],
            "best_dev_log_loss": report["best_dev_log_loss"],
            "history": report["history"], "baseline_dev": report["baseline_dev"]}


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/dataset": data_volume, "/model-cache": model_cache,
                       "/runs": training_volume})
def verify_inference(run_name):
    from pathlib import Path

    run_dir = Path("/runs") / run_name
    report = json.loads((run_dir / "report.json").read_text())
    torch, classifier, _ = _load_classifier()
    classifier.head.load_state_dict(torch.load(run_dir / "head.pt", map_location="cpu",
                                               weights_only=True))
    dev = []
    for path in sorted((run_dir / "features" / "dev").glob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        row_id = path.stem.rsplit("-", 1)[0]
        dev.append({"feature": record["feature"], "target": record["target"],
                    "count": record["count"], "id": row_id})
    # Read family from the same data fingerprint inputs used by training.
    if not dev:
        raise ValueError("No cached dev features")
    family_by_id = {row["id"]: row["task_family"]
                    for rows in _load_rows(tuple(report["families"])).values()
                    for row in rows}
    for item in dev:
        item["family"] = family_by_id[item["id"]]
    metrics = _metrics(torch, classifier.head, dev)
    chosen = min(report["history"], key=lambda item: item["dev_log_loss"])
    for family, values in metrics.items():
        expected = chosen["dev"][family]
        if values["n"] != expected["n"] or abs(values["log_loss"] - expected["log_loss"]) > 1e-4:
            raise ValueError(f"Saved checkpoint predictions differ for {family}")
    return {"checkpoint": report["checkpoint"], "dev_features": len(dev),
            "saved_epoch": chosen["epoch"], "dev": metrics}


@app.local_entrypoint()
def main(mode: str = "inspect", families: str = "sport_event,shot_result,action_classification",
         run_name: str = "sports-head-v1", epochs: int = 3,
         label_policy: str = "reviewed", shards: int = 4, shard: int = 0,
         weighting: str = "uniform", splits: str = "train,dev",
         dataset_version: str = "v1"):
    selected = _families(families)
    if mode == "inspect":
        result = inspect.remote(selected)
    elif mode == "smoke":
        result = smoke.remote()
    elif mode == "data-smoke":
        result = data_smoke.remote(selected)
    elif mode == "train":
        result = train_head.remote(selected, run_name, epochs, label_policy)
    elif mode == "verify":
        result = verify_run.remote(run_name)
    elif mode == "verify-inference":
        result = verify_inference.remote(run_name)
    elif mode == "general-smoke":
        result = general_smoke.remote()
    elif mode == "general-source-smoke":
        result = general_source_smoke.remote()
    elif mode == "general-footprint":
        result = general_feature_footprint.remote(run_name, dataset_version)
    elif mode == "general-extract":
        selected_splits = tuple(name.strip() for name in splits.split(","))
        if not selected_splits or not set(selected_splits) <= {"train", "dev", "calibration", "test"}:
            raise ValueError("Invalid general extraction splits")
        result = list(extract_general_shard.map(
            [run_name] * shards, range(shards), [shards] * shards,
            [selected_splits] * shards, [dataset_version] * shards))
    elif mode == "general-extract-shard":
        selected_splits = tuple(name.strip() for name in splits.split(","))
        if (not selected_splits or
                not set(selected_splits) <= {"train", "dev", "calibration", "test"} or
                not 0 <= shard < shards <= 16):
            raise ValueError("Invalid general extraction shard/splits")
        result = extract_general_shard.remote(
            run_name, shard, shards, selected_splits, dataset_version)
    elif mode == "general-complete":
        result = complete_general_head_pipeline.remote(run_name, dataset_version,
                                                       shards, epochs)
    elif mode == "general-train":
        result = train_general_head.remote(run_name, epochs, weighting, dataset_version)
    elif mode == "general-evaluate":
        if splits not in {"calibration", "test"}:
            raise ValueError("Set --splits calibration or test")
        result = evaluate_general_head.remote(run_name, weighting, splits, dataset_version)
    else:
        raise ValueError("unsupported mode")
    print(json.dumps(result, ensure_ascii=False, indent=2))
