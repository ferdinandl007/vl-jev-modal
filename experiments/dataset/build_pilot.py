#!/usr/bin/env python3
"""Build a small, source-pinned Jev-style multimodal decision dataset.

Metadata only: video/image bytes are deliberately left at their original source.
Requires Python 3.10+ and internet access, but no third-party packages.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ENV = os.environ.get("VL_JEV_DATASET_OUT", "")
if not OUTPUT_ENV.startswith("/dataset/"):
    raise RuntimeError("The pilot builder must run inside the Modal function with /dataset mounted")
OUT = Path(OUTPUT_ENV)
SEED = "vl-jev-pilot-v1-2026-09-23"
SOURCES = {
    "soccer": ("infactory-ai/soccer-events", "d8ba9a491eea5c573525a37d42fd55144c12dd37"),
    "basketball": ("GabrieleGiudici/BARD", "d98a075d42c46518cf5b3c7255cb4c6d207adeef"),
    "driving": ("ibarcelo/Automingo_dataset", "8b5dda7070621ae59fa68649522e383177efff88"),
    "counting": ("allenai/pixmo-count", "ebf51cf70e45d12374e64d475862e4f8d21d31d0"),
    "nli": ("stanfordnlp/snli", "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"),
    "intent": ("github:PolyAI-LDN/task-specific-datasets", "57ec275d8078af65b7731c2a98be812d844a6d6b"),
    "tool": ("5551z/VC-Tooler-SFT", "258dd7c42f4076133ace60cc294ef403dcee94af"),
    "action": ("guyuchao/UCF101", "057753e5d0709d3f5b8104a803b91a420a069103"),
    "archive": ("davanstrien/prelinger-moments", "2505953d378cf004a9d8235c723f80a9ae43e7d7"),
    "horse_qa": ("Reubencf/Adaption-video-qa-diverse-topics", "1874e12bd430562c22e969ca3bcd35f8cc9f84ac"),
    "sentiment": ("SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3"),
}


def get_json(url: str):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vl-jev-pilot-builder/1"}), timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if attempt == 4:
                raise
            time.sleep(15 * (2 ** attempt) if error.code == 429 else 2 ** attempt)
        except (TimeoutError, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def get_text(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vl-jev-pilot-builder/1"}), timeout=90) as response:
        return response.read().decode("utf-8-sig")


def hf_file(source: str, path: str):
    repo, revision = SOURCES[source]
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def github_file(source: str, path: str):
    repo, revision = SOURCES[source]
    assert repo.startswith("github:")
    return f"https://raw.githubusercontent.com/{repo.removeprefix('github:')}/{revision}/{path}"


def rows(source: str, split: str, offsets: list[int], config: str = "default", length: int = 100):
    repo, _ = SOURCES[source]
    seen = set()
    for offset in offsets:
        params = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split, "offset": offset, "length": length})
        for item in get_json("https://datasets-server.huggingface.co/rows?" + params)["rows"]:
            if item["row_idx"] not in seen:
                seen.add(item["row_idx"])
                yield item["row_idx"], item["row"]


def spread(n: int, blocks: int, block_size: int = 100):
    return sorted(set(round(i * max(0, n - block_size) / max(1, blocks - 1)) for i in range(blocks)))


def rank(value: str):
    return int(hashlib.sha256((SEED + ":" + value).encode()).hexdigest(), 16)


def holdout(group: str, official: str | None = None):
    if official == "test":
        return "test"
    if official == "validation":
        return "calibration"
    p = rank("split:" + group) % 100
    return "train" if p < 60 else "dev" if p < 70 else "calibration" if p < 85 else "test"


def choice(prompt: str, values: list[str], correct: str):
    assert correct in values and len(values) == len(set(values)) and 2 <= len(values) <= 8
    keys = "ABCDEFGH"[: len(values)]
    return {"type": "choice", "instructions": prompt, "criteria": dict(zip(keys, values))}, keys[values.index(correct)]


def noul(prompt: str, positive: bool):
    return {"type": "noul", "instructions": prompt,
            "criteria": {"true": "Yes, this is true", "false": "No, this is false"}}, "true" if positive else "false"


def score(prompt: str, levels: list[str], correct_index: int):
    assert 2 <= len(levels) <= 10 and 0 <= correct_index < len(levels)
    return {"type": "score", "instructions": prompt, "criteria": levels}, str(correct_index)


def make(source: str, key: str, modality: str, family: str, group: str, question: dict,
         gold: str, media: dict | None, *, state: str = "", official: str | None = None,
         quality: str = "source_annotation", usage: str = "research_only", extra: dict | None = None):
    repo, revision = SOURCES[source]
    return {
        "id": hashlib.sha256(f"{source}:{key}".encode()).hexdigest()[:20],
        "split": holdout(f"{source}:{group}", official),
        "modality": modality,
        "task_family": family,
        "source": {"repo": repo, "revision": revision, "row": key, "group": group, "original_split": official},
        "media": media,
        "state": state,
        "question": question,
        "gold": {"key": gold},
        "label_quality": quality,
        "usage": usage,
        "extra": extra or {},
    }


def soccer():
    text = get_text(hf_file("soccer", "metadata.csv"))
    all_rows = list(csv.DictReader(io.StringIO(text)))
    result = []
    for item in all_rows:
        label = item["event_category"]
        if label not in {"goal", "yellow_card", "red_card"}:
            continue
        values = ["Goal", "Yellow card", "Red card"]
        question, gold = choice("Which event occurs in this football clip?", values,
                                {"goal": "Goal", "yellow_card": "Yellow card", "red_card": "Red card"}[label])
        group = ":".join([item["game_date"], item["home_team"], item["away_team"]])
        path = "data/" + item["mp4_file"]
        result.append(make("soccer", item["asset_id"], "video", "sport_event", group, question, gold,
                           {"kind": "hf_file", "url": hf_file("soccer", path), "path": path},
                           quality="publisher_event_label", extra={"sport": "football", "event": label}))
    return result


SHOT = re.compile(r"made a (2PT Shot|3PT Shot|Free Throw) which result was (miss|made)\b", re.I)


def basketball():
    data = get_json(hf_file("basketball", "captions/caption.json"))
    result = []
    for item in data:
        caption = "\n".join(x.get("value", "") for x in item["conversations"] if x.get("from") == "gpt")
        found = SHOT.findall(caption)
        if len(found) != 1:
            continue
        path = item["video"]
        shot_type, outcome = found[0]
        if rank(path) % 3 == 0:
            points = 0 if outcome.lower() == "miss" else {"2pt shot": 2, "3pt shot": 3, "free throw": 1}[shot_type.lower()]
            question, gold = score("How many points result from the shot attempt in this clip?",
                                   ["0 points", "1 point", "2 points", "3 points"], points)
        else:
            question, gold = noul("Does the shot attempt in this basketball clip go in?", outcome.lower() == "made")
        result.append(make("basketball", path, "video", "shot_result", path.split("/")[0], question, gold,
                           {"kind": "hf_file", "url": hf_file("basketball", path), "path": path},
                           quality="caption_derived_weak", extra={"sport": "basketball"}))
    quotas = {("noul", "true"): 120, ("noul", "false"): 110,
              ("score", "0"): 50, ("score", "1"): 20,
              ("score", "2"): 35, ("score", "3"): 25}
    selected = []
    for (kind, gold), quota in quotas.items():
        pool = [x for x in result if x["question"]["type"] == kind and x["gold"]["key"] == gold]
        selected.extend(sorted(pool, key=lambda x: rank(x["id"]))[:quota])
    return selected


def driving():
    result = []
    for original, n, blocks in [("train", 3256, 7), ("validation", 1055, 3)]:
        for index, item in rows("driving", original, spread(n, blocks)):
            answer = item["ground_truth_answer"].strip().lower()
            if answer not in {"yes", "no"} or not item["question"].strip():
                continue
            if rank(f"driving:{original}:{index}") % 3:
                question, gold = noul(item["question"].strip(), answer == "yes")
            else:
                question, gold = choice(item["question"].strip(), ["Yes", "No"], answer.title())
            result.append(make("driving", f"{original}:{index}", "image_sequence", "driving_vqa",
                               item["scene_id"], question, gold,
                               {"kind": "hf_dataset_row_images", "config": "default", "split": original,
                                "row_index": index, "columns": [f"image_{i}" for i in range(1, 6)]},
                               official="test" if original == "validation" else original,
                               extra={"scene_id": item["scene_id"], "situation": item["situation"],
                                                         "time_span": item["time_span"]}))
    return sorted(result, key=lambda x: rank(x["id"]))[:330]


def counting():
    result = []
    for original, n, blocks in [("train", 36916, 5), ("validation", 540, 1), ("test", 540, 1)]:
        for index, item in rows("counting", original, spread(n, blocks)):
            count = item["count"]
            if not isinstance(count, int) or count < 0 or count > 15 or not item["image_url"]:
                continue
            options = sorted(set([count, max(0, count - 1), count + 1, count + 2]))
            if len(options) < 4:
                options.append(max(options) + 1)
            if count <= 9 and rank(f"counting:{original}:{index}") % 2:
                question, gold = score(f"How many {item['label']} are visible in the image?",
                                       [str(x) for x in range(10)], count)
            else:
                question, gold = choice(f"How many {item['label']} are visible in the image?",
                                        [str(x) for x in options], str(count))
            result.append(make("counting", f"{original}:{index}", "image", "object_count",
                               item["image_sha256"], question, gold,
                               {"kind": "external_image", "url": item["image_url"],
                                "sha256": item["image_sha256"]}, official=original,
                               extra={"object": item["label"]}))
    return sorted(result, key=lambda x: rank(x["id"]))[:230]


def nli():
    labels = ["Entails", "Neither", "Contradicts"]
    result = []
    for original, n, blocks in [("train", 550152, 3), ("validation", 10000, 1), ("test", 10000, 1)]:
        for index, item in rows("nli", original, spread(n, blocks), config="plain_text"):
            if item["label"] not in range(3):
                continue
            if rank(f"nli:{original}:{index}") % 3 == 0:
                question, gold = noul("Does the hypothesis follow from the premise?", item["label"] == 0)
            else:
                question, gold = choice("What is the relation between the premise and hypothesis?", labels,
                                        labels[item["label"]])
            state = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}"
            result.append(make("nli", f"{original}:{index}", "text", "natural_language_inference",
                               item["premise"], question, gold, None, state=state, official=original))
    return sorted(result, key=lambda x: rank(x["id"]))[:280]


def intent():
    raw = []
    for original in ["train", "test"]:
        text = get_text(github_file("intent", f"banking_data/{original}.csv"))
        raw += [(original, index, item) for index, item in enumerate(csv.DictReader(io.StringIO(text)))]
    labels = sorted({item["category"] for _, _, item in raw})
    assert len(labels) == 77
    result = []
    for original, index, item in raw:
        correct = item["category"]
        # Same-prefix alternatives are intentionally harder than random labels.
        prefix = correct.split("_")[0].lower()
        related = [x for x in labels if x != correct and (x.lower().startswith(prefix) or prefix in x.lower())]
        others = sorted((x for x in labels if x != correct and x not in related), key=lambda x: rank(correct + x))
        width = 8 if rank(f"intent-width:{original}:{index}") % 5 == 0 else 4
        close_count = min(len(related), width // 2)
        options = [correct] + related[:close_count] + others[: width - 1 - close_count]
        options = sorted(options, key=lambda x: rank(f"{index}:{x}"))
        question, gold = choice("Which support intent best matches the message?", options, correct)
        result.append(make("intent", f"{original}:{index}", "text", "intent_classification",
                           item["text"], question, gold, None, state=item["text"], official=original))
    by_label = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in result:
        answer = row["question"]["criteria"][row["gold"]["key"]]
        by_label[answer][row["source"]["original_split"]].append(row)
    selected = []
    for label in sorted(by_label):
        for original, quota in [("train", 6), ("test", 1)]:
            selected.extend(sorted(by_label[label][original], key=lambda x: rank(x["id"]))[:quota])
    return selected


TOOL_NAME = re.compile(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]")


def visual_tool():
    result = []
    for index, item in rows("tool", "train", spread(49164, 8, 10), length=10):
        messages = item.get("messages") or {}
        roles, contents = messages.get("role", []), messages.get("content", [])
        tools = item.get("tools") or []
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                continue
        names = list(dict.fromkeys(t.get("name") for t in tools if isinstance(t, dict) and t.get("name")))
        if len(set(names)) < 2 or "user" not in roles or "function_call" not in roles:
            continue
        raw_user = contents[roles.index("user")]
        if "<image>" not in raw_user or not item.get("images"):
            continue
        user = raw_user.replace("<image>", "").strip()
        call = contents[roles.index("function_call")]
        match = TOOL_NAME.search(call)
        if not match or match.group(1) not in names or not user or len(names) > 8:
            continue
        selected = match.group(1)
        question, gold = choice("Which visual tool should be used next?", names, selected)
        result.append(make("tool", f"train:{index}", "image", "visual_tool_selection",
                           str(item["id"]), question, gold,
                           {"kind": "hf_dataset_row_images", "config": "default", "split": "train",
                            "row_index": index, "columns": ["images"], "max_images": 1},
                           state=user, quality="synthetic_trajectory_weak", extra={"tools": tools}))
    return sorted(result, key=lambda x: rank(x["id"]))[:120]


ACTION_CLASSES = ["HorseRace", "HorseRiding", "BaseballPitch", "BasketballDunk", "SoccerPenalty",
                  "VolleyballSpiking", "TennisSwing", "CricketShot", "FieldHockeyPenalty", "Skiing",
                  "Surfing", "Biking", "BoxingPunchingBag", "GolfSwing", "Archery", "Rowing"]


def actions():
    result = []
    for original, filename in [("train", "train03.json"), ("test", "test03.json")]:
        data = get_json(hf_file("action", filename))
        for index, item in enumerate(data):
            path = item["video_path"]
            action = path.split("/")[-2]
            if action not in ACTION_CLASSES:
                continue
            width = 8 if rank(f"action-width:{path}") % 5 == 0 else 4
            if action.startswith("Horse"):
                base = ["HorseRace", "HorseRiding"]
                others = sorted((x for x in ACTION_CLASSES if x not in base), key=lambda x: rank(path + x))
                alternatives = base + others[:width - 2]
            else:
                alternatives = [action] + sorted((x for x in ACTION_CLASSES if x != action),
                                                  key=lambda x: rank(path + x))[:width - 1]
            question, gold = choice("Which activity is shown in this video?", alternatives, action)
            filename_only = path.rsplit("/", 1)[-1]
            group_match = re.search(r"_g(\d+)_c\d+", filename_only)
            group = action + ":" + (group_match.group(1) if group_match else filename_only)
            result.append(make("action", f"{original}:{index}", "video", "action_classification",
                               group, question, gold,
                               {"kind": "hf_tar_member", "path": path,
                                "shards": [hf_file("action", "shard-00000.tar"), hf_file("action", "shard-00001.tar")]},
                               official=original, extra={"sport_or_action": action}))
    horses = [x for action in ["HorseRace", "HorseRiding"]
              for x in sorted((y for y in result if y["extra"]["sport_or_action"] == action),
                              key=lambda y: rank(y["id"]))[:100]]
    others = sorted((x for x in result if not x["extra"]["sport_or_action"].startswith("Horse")),
                    key=lambda x: rank(x["id"]))[:160]
    return horses + others


def archive():
    raw = []
    for index, item in rows("archive", "train", spread(23148, 5)):
        license_url = item.get("licenseurl") or ""
        events = item.get("events") or []
        if isinstance(events, str):
            try:
                events = json.loads(events)
            except json.JSONDecodeError:
                continue
        if "publicdomain" not in license_url.lower() or not item.get("video_url") or not events:
            continue
        event = next((x for x in events if isinstance(x, dict) and x.get("start") is not None and x.get("end") is not None
                      and item["chunk_start"] <= x["start"] < x["end"] <= item["chunk_end"]
                      and 1 <= x["end"] - x["start"] <= 12 and 12 <= len(x.get("text", "")) <= 130), None)
        if event:
            raw.append((index, item, event))
    result = []
    for index, item, event in raw:
        wrong = [x[2]["text"].strip() for x in sorted(raw, key=lambda x: rank(str(index) + str(x[0])))
                 if x[1]["identifier"] != item["identifier"] and x[2]["text"].strip() != event["text"].strip()]
        options = [event["text"].strip()] + list(dict.fromkeys(wrong))[:3]
        if len(options) != 4:
            continue
        options = sorted(options, key=lambda x: rank(f"{index}:{x}"))
        question, gold = choice("Which event happens in this short video segment?", options, event["text"].strip())
        result.append(make("archive", str(index), "video", "caption_to_choice",
                           item["identifier"], question, gold,
                           {"kind": "external_video_segment", "url": item["video_url"],
                            "start_seconds": event["start"], "end_seconds": event["end"]},
                           quality="generated_caption_weak", usage="public_domain_marked",
                           extra={"license_url": item["licenseurl"], "archive_url": item["ia_url"],
                                  "source_chunk_start": item["chunk_start"], "source_chunk_end": item["chunk_end"]}))
    return sorted(result, key=lambda x: rank(x["id"]))[:100]


def horse_qa():
    result = []
    for index, item in rows("horse_qa", "train", [0], length=86):
        if index != 53 or item.get("license") != "public-domain":
            continue
        question, gold = choice(item["question"], ["Horse number 7", "Horse number 5"], "Horse number 7")
        result.append(make("horse_qa", str(index), "video", "horse_race_result", item["url"], question, gold,
                           {"kind": "hf_dataset_row_video", "config": "default", "split": "train",
                            "row_index": index, "column": "video", "original_url": item["url"]},
                           quality="publisher_qa",
                           usage="public_domain_marked", extra={"license": item["license"], "title": item["title"]}))
    return result


def sentiment():
    levels = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
    result = []
    for original, n, blocks in [("train", 8544, 2), ("validation", 1101, 1), ("test", 2210, 1)]:
        for index, item in rows("sentiment", original, spread(n, blocks)):
            label = item.get("label")
            if label not in range(5):
                continue
            question, gold = score("Rate the sentiment expressed by this text.", levels, label)
            result.append(make("sentiment", f"{original}:{index}", "text", "sentiment_score",
                               item["text"], question, gold, None, state=item["text"], official=original))
    return result


def validate(data: list[dict]):
    ids, groups = set(), {}
    for item in data:
        assert item["id"] not in ids, item["id"]
        ids.add(item["id"])
        question, gold = item["question"], item["gold"]["key"]
        if question["type"] == "choice":
            assert gold in question["criteria"]
        elif question["type"] == "noul":
            assert gold in {"false", "true"}
        elif question["type"] == "score":
            assert 0 <= int(gold) < len(question["criteria"])
        else:
            raise AssertionError(question["type"])
        assert item["source"]["revision"] == SOURCES[next(k for k, v in SOURCES.items() if v[0] == item["source"]["repo"])][1]
        assert (item["media"] is None) == (item["modality"] == "text")
        group = (item["source"]["repo"], item["source"]["group"])
        if group in groups:
            assert groups[group] == item["split"], f"split leakage: {group} {groups[group]} {item['split']}"
        groups[group] = item["split"]
    return {
        "total": len(data),
        "split": dict(collections.Counter(x["split"] for x in data)),
        "modality": dict(collections.Counter(x["modality"] for x in data)),
        "task_family": dict(collections.Counter(x["task_family"] for x in data)),
        "task_family_by_split": {family: dict(collections.Counter(x["split"] for x in data if x["task_family"] == family))
                                 for family in sorted({x["task_family"] for x in data})},
        "question_type": dict(collections.Counter(x["question"]["type"] for x in data)),
        "question_type_by_split": {split: dict(collections.Counter(x["question"]["type"] for x in data if x["split"] == split))
                                   for split in ["train", "dev", "calibration", "test"]},
        "modality_by_type": {modality: dict(collections.Counter(x["question"]["type"] for x in data if x["modality"] == modality))
                             for modality in sorted({x["modality"] for x in data})},
        "source": dict(collections.Counter(x["source"]["repo"] for x in data)),
        "label_quality": dict(collections.Counter(x["label_quality"] for x in data)),
        "media_kind": dict(collections.Counter(x["media"]["kind"] if x["media"] else "none" for x in data)),
    }


def finalize_row(item: dict):
    source = item["source"]
    item["split"] = ("train" if item["task_family"] == "horse_race_result" else
                     holdout(f"{next(k for k, v in SOURCES.items() if v[0] == source['repo'])}:{source['group']}",
                             source["original_split"]))
    question = item["question"]
    if question["type"] == "choice":
        answer = question["criteria"][item["gold"]["key"]]
        values = sorted(question["criteria"].values(),
                        key=lambda value: rank(f"option:{source['repo']}:{source['row']}:{value}"))
        keys = "ABCDEFGH"[:len(values)]
        question["criteria"] = dict(zip(keys, values))
        item["gold"]["key"] = keys[values.index(answer)]
    return item


def main():
    for name, (repo, revision) in SOURCES.items():
        if repo.startswith("github:"):
            current = get_json(f"https://api.github.com/repos/{repo.removeprefix('github:')}/commits/master")["sha"]
        else:
            current = get_json(f"https://huggingface.co/api/datasets/{repo}")["sha"]
        if current != revision:
            raise RuntimeError(f"{name} changed from locked revision {revision} to {current}; audit before rebuilding")
    builders = [soccer, basketball, driving, counting, nli, intent, visual_tool, actions, archive, horse_qa, sentiment]
    data = []
    for builder in builders:
        cache_path = OUT / ".build_cache" / (builder.__name__ + ".json")
        if cache_path.exists():
            produced = json.loads(cache_path.read_text())
        else:
            produced = builder()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(produced, ensure_ascii=False))
        print(f"{builder.__name__}: {len(produced)} rows", flush=True)
        data.extend(finalize_row(item) for item in produced)
    # Prefer the official evaluation row if a source repeats an image/scene across its splits.
    priority = {"train": 0, "dev": 1, "calibration": 2, "test": 3}
    best = {}
    for item in data:
        group = (item["source"]["repo"], item["source"]["group"])
        best[group] = max(priority[item["split"]], best.get(group, -1))
    data = [x for x in data if priority[x["split"]] == best[(x["source"]["repo"], x["source"]["group"])] ]
    # Stratify the rare red-card event across the four splits at match level.
    red_train_groups = sorted({x["source"]["group"] for x in data
                               if x["task_family"] == "sport_event" and x["extra"]["event"] == "red_card"
                               and x["split"] == "train"}, key=rank)
    if not any(x["task_family"] == "sport_event" and x["extra"]["event"] == "red_card"
               and x["split"] == "calibration" for x in data) and red_train_groups:
        for item in data:
            if item["task_family"] == "sport_event" and item["source"]["group"] == red_train_groups[0]:
                item["split"] = "calibration"
    data.sort(key=lambda x: (x["split"], x["task_family"], x["id"]))
    stats = validate(data)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "calibration", "test"]:
        with (OUT / f"{split}.jsonl").open("w") as f:
            for item in data:
                if item["split"] == split:
                    f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    (OUT / "sources.lock.json").write_text(json.dumps({k: {"repo": v[0], "revision": v[1]} for k, v in SOURCES.items()},
                                                indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
