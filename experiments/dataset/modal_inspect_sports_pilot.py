"""Read-only sports inventory for the existing pilot dataset, entirely on Modal."""

import json

import modal


app = modal.App("vl-jev-sports-pilot-inspection")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub==1.32.0")
data = modal.Volume.from_name("vl-jev-pilot-data")
media = modal.Volume.from_name("vl-jev-pilot-media")


@app.function(image=image, volumes={"/dataset": data, "/pilot": media}, timeout=300)
def inspect():
    from collections import Counter, defaultdict
    from pathlib import Path

    root = Path("/dataset/pilot_v1")
    if not root.is_dir():
        return {"dataset_present": False}
    result = {"dataset_present": True, "splits": {}, "sports": {},
              "cross_split_groups": [], "examples": []}
    groups = defaultdict(set)
    sports_rows = []
    for split in ("train", "dev", "calibration", "test"):
        path = root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result["splits"][split] = len(rows)
        for row in rows:
            sport = row.get("extra", {}).get("sport")
            if sport is None and row.get("source", {}).get("repo") not in (
                "guyuchao/UCF101", "Reubencf/Adaption-video-qa-diverse-topics"):
                continue
            sports_rows.append(row)
            groups[(row["source"]["repo"], row["source"]["group"])].add(split)
    for (repo, group), splits in groups.items():
        if len(splits) > 1:
            result["cross_split_groups"].append({"repo": repo, "group": group,
                                                  "splits": sorted(splits)})
    by_family = defaultdict(list)
    for row in sports_rows:
        by_family[row["task_family"]].append(row)
    for family, rows in sorted(by_family.items()):
        materialized = []
        for row in rows:
            folder = Path("/pilot") / row["split"] / row["id"]
            if (folder / "row.json").is_file():
                materialized.append(row["id"])
        result["sports"][family] = {
            "total": len(rows),
            "splits": dict(Counter(row["split"] for row in rows)),
            "sources": dict(Counter(row["source"]["repo"] for row in rows)),
            "question_types": dict(Counter(row["question"]["type"] for row in rows)),
            "gold_keys": dict(Counter(row["gold"]["key"] for row in rows)),
            "events": dict(Counter(row.get("extra", {}).get("event") for row in rows)),
            "events_by_split": {
                split: dict(Counter(row.get("extra", {}).get("event")
                                    for row in rows if row["split"] == split))
                for split in ("train", "dev", "calibration", "test")
            },
            "label_quality": dict(Counter(row["label_quality"] for row in rows)),
            "usage": dict(Counter(row["usage"] for row in rows)),
            "materialized_count": len(materialized),
            "unique_groups": len(set((row["source"]["repo"], row["source"]["group"])
                                     for row in rows)),
        }
        for row in rows[:2]:
            result["examples"].append({"id": row["id"], "split": row["split"],
                                       "family": family,
                                       "question": row["question"], "gold": row["gold"],
                                       "media_kind": row["media"]["kind"] if row["media"] else None,
                                       "group": row["source"]["group"],
                                       "materialized": row["id"] in materialized})
    stats = root / "stats.json"
    if stats.is_file():
        result["declared_total"] = json.loads(stats.read_text()).get("total")
    # The football source manifest supplies clip lengths without fetching video.
    if "sport_event" in by_family:
        import csv
        from huggingface_hub import hf_hub_download

        football = by_family["sport_event"]
        revision = football[0]["source"]["revision"]
        metadata = hf_hub_download("infactory-ai/soccer-events", "metadata.csv",
                                   repo_type="dataset", revision=revision,
                                   cache_dir="/tmp/sports-inspection")
        with open(metadata, newline="") as source:
            lengths = {item["asset_id"]: float(item["duration_seconds"])
                       for item in csv.DictReader(source)}
        result["sports"]["sport_event"]["duration_seconds_by_split"] = {}
        for split in ("train", "dev", "calibration", "test"):
            durations = sorted(lengths[row["source"]["row"]]
                               for row in football if row["split"] == split)
            result["sports"]["sport_event"]["duration_seconds_by_split"][split] = {
                "min": durations[0], "median": durations[len(durations) // 2],
                "max": durations[-1],
                "at_most_10_seconds": sum(value <= 10 for value in durations),
            }
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), ensure_ascii=False, indent=2))
