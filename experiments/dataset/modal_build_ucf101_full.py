"""Build and optionally materialize full UCF101 choice rows on Modal only."""

import json
import modal

app = modal.App("vl-jev-ucf101-full-expansion")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg")
         .pip_install("huggingface_hub==1.32.0", "requests==2.32.5"))
data_volume = modal.Volume.from_name("vl-jev-ucf101-full-data", create_if_missing=True)
media_volume = modal.Volume.from_name("vl-jev-ucf101-full-media", create_if_missing=True)

REPO = "guyuchao/UCF101"
REVISION = "057753e5d0709d3f5b8104a803b91a420a069103"
SHARDS = ["shard-00000.tar", "shard-00001.tar"]
DATA_PREFIX = "datasets/ucf101/"
ACTION_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "CleanAndJerk",
    "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen", "Diving",
    "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics", "FrisbeeCatch",
    "FrontCrawl", "GolfSwing", "Haircut", "Hammering", "HammerThrow", "HandstandPushups",
    "HandstandWalking", "HeadMassage", "HighJump", "HorseRace", "HorseRiding", "HulaHoop",
    "IceDancing", "JavelinThrow", "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking",
    "Knitting", "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf", "PlayingDhol",
    "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar", "PlayingTabla",
    "PlayingViolin", "PoleVault", "PommelHorse", "PullUps", "Punch", "PushUps", "Rafting",
    "RockClimbingIndoor", "RopeClimbing", "Rowing", "SalsaSpin", "ShavingBeard",
    "Shotput", "SkateBoarding", "Skiing", "Skijet", "SkyDiving", "SoccerJuggling",
    "SoccerPenalty", "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing", "UnevenBars",
    "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard", "YoYo",
    "BrushingTeeth",
]

# Curated close-label pools for all 101 UCF actions. Sports classes deliberately
# share same-sport labels wherever the taxonomy provides them; other actions use
# stable near-motion groups, then deterministic cross-class fillers. The builder
# derives the authoritative class inventory from the pinned split manifests.
NEAR_GROUPS = [
    {"Basketball", "BasketballDunk"},
    {"SoccerJuggling", "SoccerPenalty", "FieldHockeyPenalty"},
    {"BaseballPitch", "CricketBowling", "CricketShot", "Bowling"},
    {"TennisSwing", "TableTennisShot", "Billiards"},
    {"HorseRace", "HorseRiding", "Biking", "SkateBoarding"},
    {"BreastStroke", "FrontCrawl", "CliffDiving", "Diving", "Rafting", "Kayaking", "Surfing", "Skijet"},
    {"Skiing", "SkyDiving", "LongJump", "HighJump", "PoleVault", "JavelinThrow", "HammerThrow", "Shotput", "ThrowDiscus", "Archery"},
    {"VolleyballSpiking", "FrisbeeCatch", "Fencing", "Punch", "BoxingPunchingBag", "BoxingSpeedBag"},
    {"BalanceBeam", "FloorGymnastics", "ParallelBars", "PommelHorse", "StillRings", "UnevenBars", "HandstandPushups", "HandstandWalking", "TrampolineJumping", "CleanAndJerk", "BenchPress", "BodyWeightSquats", "JumpingJack", "JumpRope", "Lunges", "PullUps", "PushUps", "WallPushups", "RopeClimbing"},
    {"PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar", "PlayingTabla", "PlayingViolin", "Drumming"},
    {"ApplyEyeMakeup", "ApplyLipstick", "BlowDryHair", "Haircut", "HeadMassage", "ShavingBeard"},
    {"BabyCrawling", "WalkingWithDog", "MilitaryParade", "BandMarching", "TaiChi", "SumoWrestling", "SalsaSpin", "IceDancing", "HulaHoop", "Swing"},
    {"CuttingInKitchen", "Mixing", "MoppingFloor", "PizzaTossing", "Knitting", "Hammering", "Typing", "WritingOnBoard"},
    {"BlowingCandles", "JugglingBalls", "Nunchucks", "YoYo"},
]


def rank(value):
    import hashlib
    return hashlib.sha256(("vl-jev-ucf101-full-v1:" + value).encode()).hexdigest()


def display_label(label):
    import re
    overrides = {"Shotput": "Shot Put", "Skijet": "Ski Jet", "YoYo": "Yo-Yo",
                 "TaiChi": "Tai Chi", "Nunchucks": "Nunchucks"}
    if label in overrides:
        return overrides[label]
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)


@app.function(image=image, volumes={"/ucf-data": data_volume}, timeout=1800, memory=4096)
def inspect_source():
    from huggingface_hub import HfApi
    api = HfApi()
    current = api.dataset_info(REPO, files_metadata=False)
    if current.sha != REVISION:
        raise ValueError(f"source revision changed: expected {REVISION}, found {current.sha}")
    info = api.dataset_info(REPO, revision=REVISION, files_metadata=True)
    sizes = {item.rfilename: {"size": item.size, "blob_id": item.blob_id}
             for item in info.siblings if item.rfilename in ("train03.json", "test03.json", *SHARDS)}
    return {"repo": REPO, "revision": REVISION, "gated": info.gated,
            "license": info.card_data.to_dict().get("license") if info.card_data else None,
            "files": sizes}


@app.function(image=image, volumes={"/ucf-data": data_volume}, timeout=3600, memory=4096)
def build_manifest():
    import hashlib
    import json
    import re
    from collections import Counter, defaultdict
    from pathlib import Path
    import requests
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    current = api.dataset_info(REPO, files_metadata=False)
    if current.sha != REVISION:
        raise ValueError(f"source revision changed: expected {REVISION}, found {current.sha}")
    session = requests.Session()
    records = []
    for original, filename in (("train", "train03.json"), ("test", "test03.json")):
        path = hf_hub_download(REPO, filename, revision=REVISION,
                               repo_type="dataset", cache_dir="/ucf-data/hf-cache")
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, list):
            raise ValueError(f"unexpected {filename} format: {type(payload).__name__}")
        for index, item in enumerate(payload):
            video_path = item.get("video_path")
            if not isinstance(video_path, str) or not video_path:
                continue
            parts = video_path.replace("\\", "/").split("/")
            if len(parts) < 3 or not parts[-2]:
                continue
            label = parts[-2]
            filename_only = parts[-1]
            match = re.search(r"_g(\d+)_c\d+\.[^.]+$", filename_only)
            group = f"{label}:g{match.group(1)}" if match else f"{label}:{filename_only}"
            records.append({"original_split": original, "row_index": index,
                            "video_path": video_path, "label": label, "group": group,
                            "filename": filename_only})

    # Exclude every source group that appears in both official partitions.
    by_group = defaultdict(set)
    for row in records:
        by_group[row["group"]].add(row["original_split"])
    collision_groups = {group for group, splits in by_group.items() if len(splits) > 1}
    quarantined = [row for row in records if row["group"] in collision_groups]
    usable = [row for row in records if row["group"] not in collision_groups]
    class_seen = {row["label"] for row in usable}
    action_classes = sorted(class_seen)
    if len(action_classes) != 101:
        raise ValueError(f"unexpected UCF101 class inventory ({len(action_classes)}): {action_classes}")

    def family_for(label):
        return next((group for group in NEAR_GROUPS if label in group), {label})

    rows = []
    for source in usable:
        label = source["label"]
        family = family_for(label)
        # Start with exact semantic near-misses. If a small pool cannot fill the
        # 8-way question, widen deterministically across the same broad motion.
        preferred = sorted((candidate for candidate in family if candidate != label),
                           key=lambda value: rank(source["video_path"] + ":" + value))
        fillers = sorted((candidate for candidate in action_classes
                          if candidate != label and candidate not in family),
                         key=lambda value: rank(source["video_path"] + ":fill:" + value))
        options = [label] + (preferred + fillers)[:7]
        if len(options) != 8:
            raise ValueError(f"failed 8-way option construction for {label}")
        options.sort(key=lambda value: rank(source["video_path"] + ":option:" + value))
        correct_index = options.index(label)
        keys = "ABCDEFGH"
        display_options = [display_label(value) for value in options]
        if len({value.casefold() for value in display_options}) != 8:
            raise ValueError(f"rendered option collision for {label}: {display_options}")
        criteria = {key: value for key, value in zip(keys, display_options)}
        if source["original_split"] == "test":
            output_split = "test"
        else:
            bucket = int(rank("train-split:" + source["group"]), 16) % 100
            output_split = "train" if bucket < 70 else "dev" if bucket < 85 else "calibration"
        row_id = hashlib.sha256((REPO + ":" + source["original_split"] + ":" + source["video_path"]).encode()).hexdigest()[:24]
        base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
        row = {
            "id": row_id, "split": output_split, "modality": "video",
            "task_family": "ucf101_action_classification",
            "source": {"repo": REPO, "revision": REVISION,
                       "row": f"{source['original_split']}:{source['row_index']}",
                       "group": source["group"], "original_split": source["original_split"]},
            "media": {"kind": "hf_tar_member", "path": DATA_PREFIX + source["video_path"].removeprefix(DATA_PREFIX),
                      "shards": [base + shard for shard in SHARDS]},
            "state": "", "question": {"type": "choice",
                "instructions": "Which human activity is shown in this video?",
                "criteria": criteria},
            "gold": {"key": keys[correct_index]},
            "label_quality": "official_class_annotation", "usage": "research_only",
            "extra": {"dataset": "UCF101", "action_label": label,
                      "sport_or_action": label, "video_group_id": source["group"],
                      "original_filename": source["filename"],
                      "rights_note": "Research use only; verify upstream footage rights before redistribution."},
        }
        rows.append(row)

    root = Path("/ucf-data/ucf101_full_v1")
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "calibration", "test"):
        selected = [row for row in rows if row["split"] == split]
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    (root / "quarantine.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quarantined))
    counts = {split: sum(row["split"] == split for row in rows)
              for split in ("train", "dev", "calibration", "test")}
    by_label = {split: dict(Counter(row["extra"]["action_label"] for row in rows if row["split"] == split))
                for split in ("train", "dev", "calibration", "test")}
    gold_positions = {split: dict(Counter(
        "ABCDEFGH".index(row["gold"]["key"]) + 1 for row in rows if row["split"] == split))
        for split in ("train", "dev", "calibration", "test")}
    label_balance = {}
    for split in ("train", "dev", "calibration", "test"):
        label_counts = sorted(by_label[split].get(label, 0) for label in action_classes)
        label_balance[split] = {"classes": len(action_classes), "min": label_counts[0],
                                "median": label_counts[len(label_counts) // 2], "max": label_counts[-1],
                                "zero_count_classes": sum(value == 0 for value in label_counts)}
    summary = {"source": REPO, "revision": REVISION, "classes": len(class_seen),
               "class_names": action_classes,
               "total_usable_rows": len(rows), "split_counts": counts,
               "original_split_counts": dict(Counter(row["source"]["original_split"] for row in rows)),
               "quarantined_rows": len(quarantined), "quarantined_groups": sorted(collision_groups),
               "train_group_partition_counts": dict(Counter(row["split"] for row in rows
                                                  if row["source"]["original_split"] == "train")),
               "rows_by_label_by_split": by_label, "label_count_summary_by_split": label_balance,
               "gold_option_position_counts": gold_positions,
               "options_per_row": 8, "rights": "research_only"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    data_volume.commit()
    return summary


@app.function(image=image, volumes={"/ucf-data": data_volume, "/ucf-media": media_volume},
              timeout=3600, memory=4096)
def smoke_media():
    import hashlib
    import json
    import subprocess
    import tarfile
    from pathlib import Path
    import requests

    root = Path("/ucf-data/ucf101_full_v1")
    rows = []
    for split in ("train", "test"):
        path = root / f"{split}.jsonl"
        rows.append(json.loads(path.open().readline()))
    selected = {row["media"]["path"].removeprefix(DATA_PREFIX): row for row in rows}
    results = []
    for url in rows[0]["media"]["shards"]:
        with requests.get(url, stream=True, timeout=3600) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode="r|") as archive:
                for member in archive:
                    key = member.name.lstrip("./")
                    if key not in selected or not member.isfile():
                        continue
                    row = selected.pop(key)
                    target = Path("/ucf-media/smoke") / f"{row['split']}-{row['id']}.avi"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open("wb") as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
                    duration = float(subprocess.check_output(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(target)], text=True).strip())
                    if duration <= 0:
                        raise ValueError(f"invalid clip duration: {target}")
                    results.append({"id": row["id"], "split": row["split"],
                                    "bytes": target.stat().st_size, "duration_seconds": duration,
                                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
                    if not selected:
                        break
        if not selected:
            break
    media_volume.commit()
    return {"smoke_media_verified": len(results), "results": results,
            "missing": sorted(selected)}


@app.function(image=image, volumes={"/ucf-data": data_volume, "/ucf-media": media_volume},
              timeout=86400, memory=8192)
def materialize_all():
    """Stream each archive once and extract every manifest member to the media Volume."""
    import hashlib
    import json
    import subprocess
    import tarfile
    from collections import Counter
    from pathlib import Path
    import requests

    root = Path("/ucf-data/ucf101_full_v1")
    if not (root / "summary.json").is_file():
        raise FileNotFoundError("run manifest mode first")
    pending = {}
    skipped_existing = 0
    for split in ("train", "dev", "calibration", "test"):
        split_dir = Path("/ucf-media") / split
        existing_ids = {path.name for path in split_dir.iterdir() if path.is_dir()} if split_dir.exists() else set()
        print(f"stage=inventory split={split} existing_dirs={len(existing_ids)}", flush=True)
        for line in (root / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = row["media"]["path"].removeprefix(DATA_PREFIX).lstrip("./")
            if row["id"] in existing_ids:
                # Prior snapshots were committed only after full row + media writes.
                # Avoid per-row FUSE stat/read calls here; integrity is checked in status mode.
                skipped_existing += 1
                continue
            pending[key] = row
    print(f"stage=pending ready_existing={skipped_existing} pending={len(pending)}", flush=True)
    stats = Counter({"skipped_existing": skipped_existing})
    media_failures = []
    session = requests.Session()
    summary = json.loads((root / "summary.json").read_text())
    shards = [f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}" for name in SHARDS]
    for shard_url in shards:
        print(f"stage=stream_shard url={shard_url.rsplit('/', 1)[-1]} pending={len(pending)}", flush=True)
        with session.get(shard_url, stream=True, timeout=3600) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode="r|") as archive:
                scanned = 0
                for member in archive:
                    scanned += 1
                    if scanned % 5000 == 0:
                        print(f"stage=scanning shard={shard_url.rsplit('/', 1)[-1]} members={scanned} pending={len(pending)} extracted={stats['materialized']}", flush=True)
                    key = member.name.lstrip("./")
                    row = pending.get(key)
                    if row is None or not member.isfile():
                        continue
                    split = row["split"]
                    target_dir = Path("/ucf-media") / split / row["id"]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / "media.avi"
                    with archive.extractfile(member) as src, target.open("wb") as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
                    try:
                        duration = float(subprocess.check_output(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(target)], text=True,
                        timeout=60).strip())
                        if duration <= 0:
                            raise ValueError(f"nonpositive duration {duration}")
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as error:
                        target.unlink(missing_ok=True)
                        stats["invalid_media"] += 1
                        failed = {"id": row["id"], "path": key, "split": split,
                                  "reason": f"{type(error).__name__}: {error}"}
                        stats["failed_media"] += 1
                        media_failures.append(failed)
                    else:
                        meta = {**row, "materialized_media": [{"path": "media.avi",
                                "duration_seconds": duration, "bytes": target.stat().st_size,
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}]}
                        (target_dir / "row.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                        stats["materialized"] += 1
                    pending.pop(key)
                    if stats["materialized"] and stats["materialized"] % 100 == 0:
                        media_volume.commit()
                        print(f"stage=commit extracted={stats['materialized']} pending={len(pending)}", flush=True)
                    elif stats["materialized"] % 25 == 0:
                        print(f"stage=progress extracted={stats['materialized']} pending={len(pending)}", flush=True)
                media_volume.commit()
        if not pending:
            break
    print(f"stage=archive_scan_complete extracted={stats['materialized']} pending={len(pending)}", flush=True)
    stats["missing_members"] = len(pending)
    missing_path = root / "missing_members.json"
    missing_path.write_text(json.dumps([
        {"path": key, "id": row["id"], "split": row["split"],
         "source_row": row["source"]["row"], "group": row["source"]["group"]}
        for key, row in sorted(pending.items())
    ] + media_failures, indent=2, ensure_ascii=False))
    stats["expected_rows"] = summary["total_usable_rows"]
    stats["quarantined_rows"] = summary["quarantined_rows"]
    (root / "materialization_summary.json").write_text(json.dumps(dict(stats), indent=2))
    data_volume.commit()
    media_volume.commit()
    return dict(stats)


@app.function(image=image, volumes={"/ucf-data": data_volume, "/ucf-media": media_volume},
              timeout=1800)
def repair_orphaned_records():
    """Validate videos left by an interrupted extraction and restore metadata."""
    import hashlib
    import subprocess
    from pathlib import Path

    root = Path("/ucf-data/ucf101_full_v1")
    repaired = []
    for split in ("train", "dev", "calibration", "test"):
        for line in (root / f"{split}.jsonl").open():
            row = json.loads(line)
            folder = Path("/ucf-media") / split / row["id"]
            target = folder / "media.avi"
            record = folder / "row.json"
            if record.is_file() or not target.is_file():
                continue
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(target)],
                check=True, capture_output=True, text=True, timeout=60)
            duration = float(probe.stdout.strip())
            if duration <= 0:
                raise ValueError(f"Invalid duration for {row['id']}")
            subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(target),
                            "-f", "null", "-"], check=True, capture_output=True,
                           text=True, timeout=180)
            digest = hashlib.sha256()
            with target.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            meta = {**row, "materialized_media": [{
                "path": "media.avi", "duration_seconds": duration,
                "bytes": target.stat().st_size, "sha256": digest.hexdigest()}]}
            record.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            repaired.append({"id": row["id"], "split": split,
                             "bytes": target.stat().st_size,
                             "sha256": digest.hexdigest()})
            media_volume.commit()
    return {"repaired": repaired, "count": len(repaired)}


@app.function(image=image, volumes={"/ucf-data": data_volume, "/ucf-media": media_volume}, timeout=1800)
def status():
    import hashlib
    import json
    from pathlib import Path
    root = Path("/ucf-data/ucf101_full_v1")
    if not (root / "summary.json").is_file():
        return {"manifest": "missing"}
    summary = json.loads((root / "summary.json").read_text())
    materialized = {}
    missing_ids = []
    hashes = {}
    for split in ("train", "dev", "calibration", "test"):
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        ready = 0
        for row in rows:
            media_dir = Path("/ucf-media") / split / row["id"]
            record = media_dir / "row.json"
            target = media_dir / "media.avi"
            if record.is_file() and target.is_file():
                saved = json.loads(record.read_text())
                info = (saved.get("materialized_media") or [{}])[0]
                digest = hashlib.sha256()
                with target.open("rb") as source:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                if (saved.get("id") == row["id"] and saved.get("source") == row["source"]
                        and saved.get("media") == row["media"]
                        and info.get("bytes") == target.stat().st_size
                        and float(info.get("duration_seconds", 0)) > 0
                        and len(info.get("sha256", "")) == 64
                        and digest.hexdigest() == info.get("sha256")):
                    ready += 1
                    continue
            missing_ids.append(row["id"])
        materialized[split] = ready
        hashes[f"{split}.jsonl"] = hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
    hashes["summary.json"] = hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
    quarantine_file = root / "quarantine.jsonl"
    hashes["quarantine.jsonl"] = hashlib.sha256(quarantine_file.read_bytes()).hexdigest()
    return {"source": summary["source"], "revision": summary["revision"],
            "classes": summary["classes"], "total_rows": summary["total_usable_rows"],
            "split_counts": summary["split_counts"],
            "quarantined_rows": summary["quarantined_rows"],
            "materialized_by_split": materialized, "missing_media_rows": len(missing_ids),
            "missing_media_sample_ids": missing_ids[:20], "sha256": hashes}


@app.local_entrypoint()
def main(mode: str = "inspect"):
    if mode == "inspect":
        print(json.dumps(inspect_source.remote(), indent=2, ensure_ascii=False))
    elif mode == "build":
        print(json.dumps(build_manifest.remote(), indent=2, ensure_ascii=False))
    elif mode == "smoke":
        print(json.dumps(smoke_media.remote(), indent=2, ensure_ascii=False))
    elif mode == "materialize":
        print(json.dumps(materialize_all.remote(), indent=2, ensure_ascii=False))
    elif mode == "repair-orphaned-records":
        print(json.dumps(repair_orphaned_records.remote(), indent=2, ensure_ascii=False))
    elif mode == "status":
        print(json.dumps(status.remote(), indent=2, ensure_ascii=False))
    else:
        raise ValueError("mode must be inspect, build, smoke, materialize, repair-orphaned-records, or status")
