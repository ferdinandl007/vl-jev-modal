"""Materialize selected pilot media on Modal, leaving the local disk untouched.

Run a small source smoke check with:
    modal run experiments/dataset/modal_materialize_pilot.py --smoke

UCF101 tar shards total about 3.5 GB and require --include-shards explicitly.
"""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal


app = modal.App("vl-jev-pilot-materializer")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("requests==2.32.5", "pillow==12.1.1"))
volume = modal.Volume.from_name("vl-jev-pilot-media", create_if_missing=True)
data_volume = modal.Volume.from_name("vl-jev-pilot-data", create_if_missing=True)


@app.function(image=image, volumes={"/pilot": volume, "/dataset": data_volume}, timeout=300)
def media_status():
    from collections import Counter

    status = {}
    for split in ("train", "dev", "calibration", "test"):
        rows = [json.loads(line) for line in
                (Path("/dataset/pilot_v1") / f"{split}.jsonl").read_text().splitlines()]
        visual = [row for row in rows if row["media"]]
        by_family = Counter(row["task_family"] for row in visual)
        ready = Counter()
        missing_sample = {}
        for row in visual:
            folder = Path("/pilot") / split / row["id"]
            required = "image_0.jpg" if row["modality"] == "image" else "media.mp4"
            if (folder / "row.json").is_file() and (folder / required).is_file():
                ready[row["task_family"]] += 1
            else:
                missing_sample.setdefault(row["task_family"], row["id"])
        status[split] = {"visual_expected": len(visual), "visual_ready": sum(ready.values()),
                         "expected_by_family": dict(by_family), "ready_by_family": dict(ready),
                         "missing_sample_by_family": missing_sample}
    return status


@app.function(image=image, volumes={"/dataset": data_volume}, timeout=3600)
def inspect_ucf_shards():
    import requests

    source = Path("/dataset/pilot_v1/train.jsonl")
    row = next(json.loads(line) for line in source.read_text().splitlines()
               if '"hf_tar_member"' in line)
    result = []
    for url in row["media"]["shards"]:
        first, horses = [], []
        with requests.get(url, stream=True, timeout=3600) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode="r|") as archive:
                for member in archive:
                    if len(first) < 8:
                        first.append(member.name)
                    if "HorseRace" in member.name and len(horses) < 8:
                        horses.append(member.name)
                    if len(first) == 8 and len(horses) == 8:
                        break
        result.append({"shard": url.rsplit("/", 1)[-1], "first": first, "horse_examples": horses})
    return result


@app.function(image=image, volumes={"/pilot": volume, "/dataset": data_volume}, timeout=3600, memory=4096)
def materialize_batch(split: str, limit: int, smoke: bool, row_id: str,
                      include_shards: bool = False, families: str = "",
                      per_family: bool = False, only_missing: bool = False):
    import requests
    from PIL import Image
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    source_file = Path("/dataset/pilot_v1") / f"{split}.jsonl"
    rows = [json.loads(line) for line in source_file.read_text().splitlines()]
    rows = [row for row in rows if row["media"]]
    if families:
        wanted = set(families.split(","))
        rows = [row for row in rows if row["task_family"] in wanted]
    if row_id:
        selected = [row for row in rows if row["id"] == row_id]
        if len(selected) != 1:
            raise ValueError(f"row id {row_id} not found in {split}")
    elif smoke:
        selected = []
        seen = set()
        for row in rows:
            kind = row["media"]["kind"]
            if kind != "hf_tar_member" and kind not in seen:
                selected.append(row)
                seen.add(kind)
    else:
        if per_family and limit:
            selected = []
            counts = {}
            for row in rows:
                family = row["task_family"]
                if counts.get(family, 0) < limit:
                    selected.append(row)
                    counts[family] = counts.get(family, 0) + 1
        else:
            selected = rows if limit == 0 else rows[:limit]

    if only_missing:
        selected = [row for row in selected if not (
            (Path("/pilot") / split / row["id"] / "row.json").is_file() and
            (Path("/pilot") / split / row["id"] /
             ("image_0.jpg" if row["modality"] == "image" else "media.mp4")).is_file())]

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=8, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True)))
    source_sha = {}
    results = []
    work = Path(tempfile.mkdtemp(prefix="vl-jev-materialize-"))

    def check_source(row):
        repo = row["source"]["repo"]
        if repo.startswith("github:"):
            return
        if repo not in source_sha:
            response = session.get(f"https://huggingface.co/api/datasets/{repo}", timeout=30)
            response.raise_for_status()
            source_sha[repo] = response.json()["sha"]
        if source_sha[repo] != row["source"]["revision"]:
            raise ValueError(f"source revision changed for {repo}")

    def fetch(url, path, maximum=200_000_000):
        total = 0
        digest = hashlib.sha256()
        with session.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with path.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > maximum:
                        raise ValueError(f"media exceeds {maximum} bytes: {url}")
                    digest.update(chunk)
                    out.write(chunk)
        return {"sha256_downloaded_bytes": digest.hexdigest(), "bytes": total}

    def viewer_row(row):
        source, media = row["source"], row["media"]
        params = {"dataset": source["repo"], "config": media["config"],
                  "split": media["split"], "offset": media["row_index"], "length": 1}
        response = session.get("https://datasets-server.huggingface.co/rows", params=params, timeout=90)
        response.raise_for_status()
        item = response.json()["rows"][0]
        if item["row_idx"] != media["row_index"]:
            raise ValueError(f"viewer row changed for {row['id']}")
        return item["row"]

    def save_image(url, path):
        downloaded = fetch(url, path.with_suffix(".raw"), maximum=30_000_000)
        with Image.open(path.with_suffix(".raw")) as source:
            rgb = source.convert("RGB")
            rgb.save(path, format="JPEG", quality=95)
            dimensions = [rgb.width, rgb.height]
        path.with_suffix(".raw").unlink()
        return {"path": path.name, "dimensions": dimensions,
                "sha256_downloaded_bytes": downloaded["sha256_downloaded_bytes"]}

    def verify_video(path):
        command = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
        output = subprocess.check_output(command, text=True, timeout=60).strip()
        duration = float(output)
        if duration <= 0:
            raise ValueError("nonpositive video duration")
        return duration

    def process(row):
        check_source(row)
        media = row["media"]
        destination = Path("/pilot") / row["split"] / row["id"]
        record = destination / "row.json"
        target = destination / "media.mp4"
        required = destination / ("image_0.jpg" if row["modality"] == "image" else "media.mp4")
        if record.is_file() and required.is_file():
            prior = json.loads(record.read_text())
            if (prior["id"] == row["id"] and prior["source"] == row["source"]
                    and prior["media"] == row["media"] and
                    (row["modality"] == "image" or verify_video(target) > 0)):
                return {"id": row["id"], "kind": media["kind"], "skipped_existing": True}
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        kind = media["kind"]
        files = []
        if kind in {"hf_file", "external_video", "external_video_segment"}:
            source_path = work / f"{row['id']}-source.mp4"
            downloaded = fetch(media["url"], source_path)
            target = destination / "media.mp4"
            if kind == "external_video_segment":
                length = media["end_seconds"] - media["start_seconds"]
                if not 0 < length <= 15:
                    raise ValueError("invalid segment interval")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(media["start_seconds"]),
                                "-i", str(source_path), "-t", str(length), "-an", "-c:v", "libx264",
                                "-preset", "veryfast", "-pix_fmt", "yuv420p", str(target)],
                               check=True, timeout=120)
            else:
                shutil.copyfile(source_path, target)
            files.append({"path": "media.mp4", "duration_seconds": verify_video(target),
                          "source_bytes": downloaded["bytes"],
                          "source_sha256_downloaded_bytes": downloaded["sha256_downloaded_bytes"]})
            source_path.unlink()
        elif kind == "external_image":
            files.append(save_image(media["url"], destination / "image_0.jpg"))
        elif kind == "hf_dataset_row_images":
            source = viewer_row(row)
            image_urls = []
            for column in media["columns"]:
                cell = source[column]
                cells = cell if isinstance(cell, list) else [cell]
                image_urls.extend(x["src"] for x in cells if isinstance(x, dict) and x.get("src"))
            if media.get("max_images") is not None:
                image_urls = image_urls[:media["max_images"]]
            if not image_urls:
                raise ValueError("dataset viewer returned no images")
            for i, url in enumerate(image_urls):
                files.append(save_image(url, destination / f"image_{i}.jpg"))
            if row["modality"] == "image_sequence":
                pattern = str(destination / "image_%d.jpg")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "1", "-i", pattern,
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination / "media.mp4")],
                               check=True, timeout=120)
                files.append({"path": "media.mp4", "duration_seconds": verify_video(destination / "media.mp4")})
        elif kind == "hf_dataset_row_video":
            source = viewer_row(row)
            cell = source[media["column"]]
            downloaded = fetch(cell["src"], destination / "media.mp4")
            files.append({"path": "media.mp4", "duration_seconds": verify_video(destination / "media.mp4"),
                          **downloaded})
        else:
            raise ValueError(f"unsupported media kind {kind}")
        materialized = {**row, "materialized_media": files}
        (destination / "row.json").write_text(json.dumps(materialized, ensure_ascii=False, indent=2))
        return {"id": row["id"], "kind": kind, "files": files}

    ordinary = [x for x in selected if x["media"] and x["media"]["kind"] != "hf_tar_member"]
    shard_rows = [x for x in selected if x["media"] and x["media"]["kind"] == "hf_tar_member"]
    try:
        for row in ordinary:
            try:
                results.append({"ok": True, **process(row)})
            except Exception as error:
                results.append({"ok": False, "id": row["id"], "error": str(error)[:500]})
            if len(results) % 50 == 0:
                volume.commit()
                print(json.dumps({"split": split, "processed": len(results),
                                  "selected": len(selected),
                                  "ok": sum(item["ok"] for item in results)}), flush=True)
        if shard_rows and not include_shards:
            for row in shard_rows:
                results.append({"ok": False, "id": row["id"], "error": "UCF101 shards require --include-shards"})
        elif shard_rows:
            # Stream each uncompressed tar once, extracting only requested members.
            pending = {}
            for row in shard_rows:
                destination = Path("/pilot") / row["split"] / row["id"]
                record = destination / "row.json"
                target = destination / "media.mp4"
                if record.is_file() and target.is_file():
                    prior = json.loads(record.read_text())
                    if (prior["id"] == row["id"] and prior["source"] == row["source"]
                            and prior["media"] == row["media"] and verify_video(target) > 0):
                        results.append({"ok": True, "id": row["id"], "kind": "hf_tar_member",
                                        "skipped_existing": True})
                        continue
                pending[row["media"]["path"].removeprefix("datasets/ucf101/").lstrip("./")] = row
            for url in shard_rows[0]["media"]["shards"]:
                if not pending:
                    break
                with session.get(url, stream=True, timeout=3600) as response:
                    response.raise_for_status()
                    response.raw.decode_content = True
                    with tarfile.open(fileobj=response.raw, mode="r|") as archive:
                        for member in archive:
                            key = member.name.lstrip("./")
                            if key not in pending or not member.isfile():
                                continue
                            row = pending.pop(key)
                            check_source(row)
                            destination = Path("/pilot") / row["split"] / row["id"]
                            destination.mkdir(parents=True, exist_ok=True)
                            target = destination / "media.mp4"
                            with archive.extractfile(member) as source, target.open("wb") as out:
                                shutil.copyfileobj(source, out)
                            files = [{"path": "media.mp4", "duration_seconds": verify_video(target),
                                      "sha256_downloaded_bytes": hashlib.sha256(target.read_bytes()).hexdigest()}]
                            (destination / "row.json").write_text(json.dumps({**row, "materialized_media": files},
                                                                             ensure_ascii=False, indent=2))
                            results.append({"ok": True, "id": row["id"], "kind": "hf_tar_member", "files": files})
            for row in pending.values():
                results.append({"ok": False, "id": row["id"], "error": "member not found in pinned UCF101 shards"})
        volume.commit()
        return {"selected": len(selected), "ok": sum(x["ok"] for x in results), "results": results}
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.local_entrypoint()
def main(split: str = "train", limit: int = 8, smoke: bool = False,
         include_shards: bool = False, row_id: str = "", inspect_ucf: bool = False,
         families: str = "", per_family: bool = False, status: bool = False,
         only_missing: bool = False):
    if status:
        print(json.dumps(media_status.remote(), ensure_ascii=False, indent=2))
        return
    if inspect_ucf:
        print(json.dumps(inspect_ucf_shards.remote(), ensure_ascii=False, indent=2))
        return
    valid_families = {"sport_event", "shot_result", "action_classification",
                      "horse_race_result"}
    if split not in {"train", "dev", "calibration", "test"} or limit < 0:
        raise ValueError("choose a valid split and a nonnegative limit")
    if families and not set(families.split(",")) <= valid_families:
        raise ValueError("families must name sports task families")
    result = materialize_batch.remote(split, limit, smoke, row_id, include_shards, families,
                                      per_family, only_missing)
    print(json.dumps({"selected": result["selected"], "ok": result["ok"],
                      "failures": [item for item in result["results"] if not item["ok"]]},
                     ensure_ascii=False, indent=2))
