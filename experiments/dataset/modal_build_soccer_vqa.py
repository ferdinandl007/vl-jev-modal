"""Pin and materialize SoccerNet SN-VQA-2026 on Modal Volumes.

The full source manifest is retained. `materialize` can prepare a balanced
pilot or the complete archive; neither operation downloads media locally.
"""

import json

import modal

app = modal.App("vl-jev-soccer-vqa-builder")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg")
         .pip_install("huggingface_hub==1.33.0", "requests==2.32.5", "pillow==12.1.1"))
data_volume = modal.Volume.from_name("vl-jev-general-v1-data")
media_volume = modal.Volume.from_name("vl-jev-general-v1-media")
REPO = "SoccerNet/SN-VQA-2026"
REVISION = "e8312fd21ece3c11ad42a6639cca3149669c8fce"
ROOT = "/general-data/soccer_vqa_2026"
SPLITS = ("train", "valid", "test")
SOURCE_COUNTS = {"train": 8000, "valid": 2000, "test": 500}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class RangeFile:
    """Seekable remote ZIP reader. Each media member is fetched by byte range."""

    def __init__(self, url, size):
        import requests

        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.cache = {}
        self.block_size = 2 * 1024 * 1024

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if not 0 <= self.pos <= self.size:
            raise ValueError("Seek outside archive")
        return self.pos

    def read(self, length=-1):
        if length < 0:
            length = self.size - self.pos
        parts = []
        while length and self.pos < self.size:
            start = self.pos // self.block_size * self.block_size
            if start not in self.cache:
                end = min(start + self.block_size, self.size) - 1
                response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != end - start + 1:
                    raise IOError("Source did not return the requested archive range")
                if len(self.cache) >= 8:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[start] = response.content
            chunk = self.cache[start][self.pos - start:self.pos - start + length]
            if not chunk:
                raise IOError("Empty archive range")
            parts.append(chunk)
            self.pos += len(chunk)
            length -= len(chunk)
        return b"".join(parts)


def _archive(split):
    import zipfile
    from huggingface_hub import HfApi

    if split not in SPLITS:
        raise ValueError(split)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("SoccerNet source revision changed")
    size = next(file.size for file in info.siblings if file.rfilename == f"{split}.zip")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{split}.zip"
    return zipfile.ZipFile(RangeFile(url, size))


def _spread(items, limit):
    """Keep temporal coverage of a frame folder within the 16-frame budget."""
    if len(items) <= limit:
        return items
    return [items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)]


def _members(names, name_set, sizes, split, materials):
    from bisect import bisect_left
    from pathlib import PurePosixPath

    ordered = []
    for original in materials or []:
        path = PurePosixPath(original)
        if path.is_absolute() or ".." in path.parts or not original.startswith("materials/"):
            raise ValueError(f"Unsafe source path: {original}")
        candidate = f"{split}/{original.rstrip('/')}"
        if (candidate in name_set and sizes[candidate] > 0 and
                PurePosixPath(candidate).suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES):
            matches = [candidate]
        else:
            prefix = candidate + "/"
            matches = []
            offset = bisect_left(names, prefix)
            while offset < len(names) and names[offset].startswith(prefix):
                if (sizes[names[offset]] > 0 and
                        PurePosixPath(names[offset]).suffix.lower() in IMAGE_SUFFIXES):
                    matches.append(names[offset])
                offset += 1
            if matches:
                def frame_order(name):
                    stem = PurePosixPath(name).stem
                    tail = stem.rsplit("_", 1)[-1]
                    return (int(tail), name) if tail.isdigit() else (0, name)

                matches.sort(key=frame_order)
                matches = _spread(matches, 16)
        if not matches:
            raise FileNotFoundError(f"Unresolved SoccerNet material: {candidate}")
        ordered.extend(matches)
    return list(dict.fromkeys(ordered))


def _row(raw, split, index, members):
    import hashlib

    options = {f"O{number}": str(raw[f"O{number}"]) for number in range(1, 11)
               if f"O{number}" in raw}
    if not 2 <= len(options) <= 10 or any(not value.strip() or value == "None"
                                           for value in options.values()):
        raise ValueError(f"Blank option in {split}:{index}")
    if raw["closeA"] not in options or not isinstance(raw["Q"], str):
        raise ValueError(f"Invalid label/question in {split}:{index}")
    kinds = {"text" if member.lower().endswith(tuple(IMAGE_SUFFIXES | VIDEO_SUFFIXES)) is False
             else "video" if member.lower().endswith(tuple(VIDEO_SUFFIXES)) else "image"
             for member in members}
    if "text" in kinds:
        raise ValueError(f"Unrecognized media extension in {split}:{index}")
    modality = ("text" if not members else "video" if "video" in kinds else
                "image" if len(members) == 1 else "image_sequence")
    family = str(raw.get("task type") or "unlabeled_test")
    source_id = raw.get("id", index + 1)
    identity = f"{split}:{source_id}:{index}"
    return {
        "id": "soccer-vqa-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
        "split": {"train": "train", "valid": "dev", "test": "test"}[split],
        "modality": modality,
        "task_family": "soccer_vqa_" + family.lower().replace(" ", "_").replace("-", "_"),
        "source": {"repo": REPO, "revision": REVISION, "row": source_id,
                   "group": members[0].rsplit("/", 1)[0] if members else identity,
                   "original_split": split, "index": index, "task_type": family},
        "media": None,
        "source_materials": raw.get("materials") or [],
        "source_members": members,
        "state": "Soccer match context. Use the supplied image or video evidence when present.",
        "question": {"type": "choice", "instructions": raw["Q"], "criteria": options},
        "gold": {"key": raw["closeA"], "provenance": "source_annotation"},
        "quality": "source_annotation", "usage": "research_only",
    }


@app.function(image=image, volumes={"/general-data": data_volume}, timeout=86400, memory=8192)
def build_manifest():
    import hashlib
    from collections import Counter
    from pathlib import Path

    root = Path(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    rejected = []
    for split in SPLITS:
        with _archive(split) as archive:
            raw_rows = json.loads(archive.read(f"{split}/{split}.json"))
            names = sorted(archive.namelist())
            name_set = set(names)
            sizes = {item.filename: item.file_size for item in archive.infolist()}
            if len(raw_rows) != SOURCE_COUNTS[split]:
                raise ValueError(f"Unexpected {split} source count")
            rows = []
            for index, raw in enumerate(raw_rows):
                try:
                    members = _members(names, name_set, sizes, split, raw.get("materials"))
                    rows.append(_row(raw, split, index, members))
                except (ValueError, KeyError, FileNotFoundError) as error:
                    rejected.append({"split": split, "index": index, "source_id": raw.get("id"),
                                     "reason": f"{type(error).__name__}: {error}"})
            all_rows[split] = rows
        print(json.dumps({"phase": "indexed", "split": split, "eligible": len(rows),
                          "rejected": len(raw_rows) - len(rows)}), flush=True)

    # Official test remains untouched. Keep all source rows and mark any exact
    # question/media overlap, allowing downstream selectors to exclude it.
    seen = {}
    duplicates = []
    for split in ("test", "valid", "train"):
        for row in all_rows[split]:
            key = hashlib.sha256(json.dumps([row["question"], row["source_members"]],
                                            sort_keys=True).encode()).hexdigest()
            if key in seen:
                duplicates.append({"id": row["id"], "other": seen[key], "split": split})
            else:
                seen[key] = row["id"]
    hashes = {}
    for split, rows in all_rows.items():
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                for row in rows))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"repo": REPO, "revision": REVISION, "source_counts": SOURCE_COUNTS,
               "counts": {split: len(rows) for split, rows in all_rows.items()},
               "task_types": {split: dict(Counter(row["source"]["task_type"] for row in rows))
                              for split, rows in all_rows.items()},
               "modalities": {split: dict(Counter(row["modality"] for row in rows))
                              for split, rows in all_rows.items()},
               "split_sha256": hashes, "exact_overlaps": duplicates,
               "rejected": rejected,
               "license_card": "cc-by-sa-4.0", "usage": "research_only",
               "status": "manifest_complete"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    data_volume.commit()
    return {"counts": summary["counts"], "source_counts": SOURCE_COUNTS,
            "rejected_count": len(rejected), "modalities": summary["modalities"],
            "exact_overlap_count": len(duplicates), "summary": str(root / "summary.json")}


def _selected(rows, split, mode, per_type):
    import hashlib
    from collections import defaultdict

    if mode == "all" or split == "test":
        return rows
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["source"]["task_type"]].append(row)
    chosen = []
    for task, candidates in buckets.items():
        candidates.sort(key=lambda row: hashlib.sha256(row["id"].encode()).digest())
        chosen.extend(candidates[:per_type])
    return sorted(chosen, key=lambda row: row["source"]["index"])


@app.function(image=image, volumes={"/general-data": data_volume,
                                    "/general-media": media_volume},
              timeout=86400, memory=8192)
def materialize(split: str, mode: str = "pilot", per_type: int = 32,
                part: int = 0, parts: int = 1):
    import hashlib
    import subprocess
    from PIL import Image
    from pathlib import Path

    if (split not in SPLITS or mode not in {"pilot", "all"} or
            not 1 <= per_type <= 10000 or not 0 <= part < parts <= 32):
        raise ValueError("Invalid materialization request")
    root = Path(ROOT)
    summary = json.loads((root / "summary.json").read_text())
    source = root / f"{split}.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != summary["split_sha256"][split]:
        raise ValueError("Source manifest changed")
    rows = [json.loads(line) for line in source.open()]
    selected = _selected(rows, split, mode, per_type)
    if parts > 1:
        selected = [row for index, row in enumerate(selected) if index % parts == part]
    output = []
    rejected_media = []
    with _archive(split) as archive:
        for index, row in enumerate(selected, start=1):
            assets = []
            folder = Path("/general-media/soccer_vqa_2026") / split / row["id"]
            folder.mkdir(parents=True, exist_ok=True)
            for member_index, member in enumerate(row["source_members"]):
                suffix = Path(member).suffix.lower()
                target = folder / f"asset_{member_index:02d}{suffix}"
                if target.is_file():
                    digest = hashlib.sha256(target.read_bytes()).hexdigest()
                else:
                    blob = archive.read(member)
                    digest = hashlib.sha256(blob).hexdigest()
                    target.write_bytes(blob)
                try:
                    if suffix in IMAGE_SUFFIXES:
                        with Image.open(target) as picture:
                            picture.verify()
                    else:
                        probe = subprocess.run(
                            ["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(target)],
                            capture_output=True, text=True, timeout=60)
                        if probe.returncode or "video" not in probe.stdout:
                            raise ValueError(probe.stderr.strip() or "No decodable video stream")
                except (OSError, ValueError) as error:
                    rejected_media.append({"id": row["id"], "member": member,
                                           "reason": f"{type(error).__name__}: {error}"})
                    assets = []
                    break
                assets.append({"path": str(target), "sha256": digest,
                               "source_member": member,
                               "kind": "video" if suffix in VIDEO_SUFFIXES else "image"})
            if len(assets) == len(row["source_members"]):
                row["media"] = {"kind": "modal_media_sequence", "assets": assets} if assets else None
                output.append(row)
            if index % 25 == 0:
                media_volume.commit()
                print(json.dumps({"phase": "materialized", "split": split,
                                  "part": part, "parts": parts,
                                  "rows": index, "total": len(selected)}), flush=True)
    media_volume.commit()
    stem = f"{mode}-{per_type}-{split}"
    if parts > 1:
        stem += f"-part-{part:02d}-of-{parts:02d}"
    target = root / f"{stem}.jsonl"
    target.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                              for row in output))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (root / f"{stem}.json").write_text(json.dumps(
        {"rows": len(output), "split": split, "mode": mode, "per_type": per_type,
         "part": part, "parts": parts, "sha256": digest,
         "path": str(target), "rejected_media": rejected_media}, indent=2))
    data_volume.commit()
    return {"split": split, "part": part, "parts": parts,
            "rows": len(output), "sha256": digest,
            "rejected_media": rejected_media,
            "media_assets": sum(len(row["media"]["assets"]) if row["media"] else 0
                                for row in output)}


@app.function(image=image, volumes={"/general-data": data_volume}, timeout=3600)
def finalize_parts(split: str, per_type: int, parts: int):
    import hashlib
    from pathlib import Path

    if split not in SPLITS or not 2 <= parts <= 32:
        raise ValueError("Invalid finalization request")
    root = Path(ROOT)
    manifest = json.loads((root / "summary.json").read_text())
    source = root / f"{split}.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["split_sha256"][split]:
        raise ValueError("Source manifest changed")
    selected = _selected([json.loads(line) for line in source.open()], split, "all", per_type)
    expected_ids = {row["id"] for row in selected}
    output, rejected = [], []
    for part in range(parts):
        stem = f"all-{per_type}-{split}-part-{part:02d}-of-{parts:02d}"
        descriptor = json.loads((root / f"{stem}.json").read_text())
        path = root / f"{stem}.jsonl"
        if (descriptor["part"] != part or descriptor["parts"] != parts or
                hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]):
            raise ValueError(f"Invalid materialized part: {stem}")
        output.extend(json.loads(line) for line in path.open())
        rejected.extend(descriptor["rejected_media"])
    ids = [row["id"] for row in output]
    rejected_ids = [row["id"] for row in rejected]
    if (len(ids) != len(set(ids)) or len(rejected_ids) != len(set(rejected_ids)) or
            set(ids) & set(rejected_ids) or set(ids) | set(rejected_ids) != expected_ids):
        raise ValueError("Materialized parts do not cover the selected source rows")
    output.sort(key=lambda row: row["source"]["index"])
    target = root / f"all-{per_type}-{split}.jsonl"
    target.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                              for row in output))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (root / f"all-{per_type}-{split}.json").write_text(json.dumps(
        {"rows": len(output), "split": split, "mode": "all", "per_type": per_type,
         "parts": parts, "sha256": digest, "path": str(target),
         "rejected_media": rejected}, indent=2))
    data_volume.commit()
    return {"split": split, "rows": len(output), "rejected_media": len(rejected),
            "sha256": digest}


@app.function(image=image, timeout=86400)
def parallel_all():
    jobs = [("train", 32, part, 8) for part in range(8)] + [
        ("valid", 8, part, 4) for part in range(4)]
    materialized = list(materialize.map(
        [job[0] for job in jobs], ["all"] * len(jobs),
        [job[1] for job in jobs], [job[2] for job in jobs],
        [job[3] for job in jobs]))
    train = finalize_parts.remote("train", 32, 8)
    valid = finalize_parts.remote("valid", 8, 4)
    return {"parts": materialized, "train": train, "valid": valid}


@app.function(image=image, volumes={"/general-data": data_volume}, timeout=300)
def status():
    from collections import Counter
    from pathlib import Path

    root = Path(ROOT)
    summary = root / "summary.json"
    manifest = json.loads(summary.read_text()) if summary.is_file() else None
    if manifest:
        manifest["rejected_by_reason"] = dict(Counter(
            item["reason"].split(":", 1)[0] + ":" + item["reason"].split(":", 1)[-1].split(" in ")[0]
            for item in manifest["rejected"]))
        manifest["rejected_examples"] = manifest["rejected"][:10]
        manifest["rejected_test"] = [item for item in manifest["rejected"] if item["split"] == "test"]
        del manifest["rejected"]
    return {"manifest": manifest,
            "materialized": [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))
                             if path.name != "summary.json"]}


@app.function(image=image, volumes={"/general-media": media_volume}, timeout=300)
def inspect_asset(split: str, row_id: str, asset_name: str = "asset_00.jpg"):
    from pathlib import Path

    if split not in SPLITS or not row_id.startswith("soccer-vqa-") or "/" in asset_name:
        raise ValueError("Invalid asset path")
    path = Path("/general-media/soccer_vqa_2026") / split / row_id / asset_name
    with path.open("rb") as source:
        prefix = source.read(32)
    return {"size": path.stat().st_size, "first_32_bytes_hex": prefix.hex()}


@app.function(image=image, volumes={"/general-data": data_volume}, timeout=300)
def audit_overlap():
    import hashlib
    from collections import Counter, defaultdict
    from pathlib import Path

    root = Path(ROOT)
    by_question = defaultdict(set)
    by_material = defaultdict(set)
    by_group = defaultdict(set)
    for split in SPLITS:
        for line in (root / f"{split}.jsonl").open():
            row = json.loads(line)
            question = hashlib.sha256(json.dumps(row["question"], sort_keys=True).encode()).hexdigest()
            by_question[question].add(split)
            for material in row["source_materials"]:
                by_material[material].add(split)
            by_group[row["source"]["group"].split("/", 1)[-1]].add(split)
    return {"cross_split_questions": dict(Counter(
                "+".join(sorted(splits)) for splits in by_question.values() if len(splits) > 1)),
            "cross_split_materials": dict(Counter(
                "+".join(sorted(splits)) for splits in by_material.values() if len(splits) > 1)),
            "cross_split_groups": dict(Counter(
                "+".join(sorted(splits)) for splits in by_group.values() if len(splits) > 1))}


@app.local_entrypoint()
def main(mode: str = "status", split: str = "train", per_type: int = 32,
         row_id: str = "", parts: int = 1):
    if mode == "manifest":
        result = build_manifest.remote()
    elif mode in {"pilot", "all"}:
        result = materialize.remote(split, mode, per_type)
    elif mode == "parallel-all":
        result = parallel_all.remote()
    elif mode == "finalize":
        result = finalize_parts.remote(split, per_type, parts)
    elif mode == "status":
        result = status.remote()
    elif mode == "inspect":
        result = inspect_asset.remote(split, row_id)
    elif mode == "audit":
        result = audit_overlap.remote()
    else:
        raise ValueError("mode must be manifest, pilot, all, parallel-all, finalize, status, inspect, or audit")
    if mode == "status" and result["manifest"]:
        result["manifest"].pop("exact_overlaps", None)
    print(json.dumps(result, indent=2))
