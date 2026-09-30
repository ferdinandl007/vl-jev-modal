"""Convert BrowserGym AX-tree actions to labeled general-head decisions on Modal."""

import json
import modal

app = modal.App("vl-jev-ax-actions-general-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==0.36.0", "pyarrow==20.0.0")
data = modal.Volume.from_name("vl-jev-gui-general-data", create_if_missing=True)
REPO = "saital/browser-agent-phase1-sft-action-only"
REVISION = "dd7aa1bdf86f0aed911e33c6f6faae26d67b1482"
SHARD = "data/train-00000-of-00001.parquet"
ROOT = "/gui-data/ax_actions_general_v1"


def _convert(source):
    import hashlib
    import re

    messages = source["messages"]
    metadata = source["metadata"]
    if metadata.get("teacher_used_fallback"):
        return None, "teacher_fallback"
    if len(messages) != 3 or [item["role"] for item in messages] != ["system", "user", "assistant"]:
        return None, "invalid_messages"
    prompt, answer = messages[1]["content"], messages[2]["content"].strip()
    match = re.fullmatch(r"(click|dblclick|hover|focus|fill|clear|select_option)\(['\"](\d+)['\"](?:,.*)?\)", answer)
    if not match:
        return None, "unsupported_action"
    op, bid = match.groups()
    observations = prompt.split("Current observation:\n", 1)
    if len(observations) != 2:
        return None, "missing_observation"
    tree = observations[1]
    nodes = []
    for line in tree.splitlines():
        node = re.search(r"\[(\d+)\]\s+([A-Za-z][\w-]*)\s+'([^']*)'", line)
        if node:
            nodes.append({"id": node.group(1), "role": node.group(2),
                          "name": " ".join(node.group(3).split())[:100]})
    target = next((node for node in nodes if node["id"] == bid), None)
    if target is None or not target["name"]:
        return None, "target_missing_or_unnamed"
    seen = {(target["role"], target["name"])}
    negatives = []
    for node in nodes:
        signature = (node["role"], node["name"])
        if node["id"] == bid or not node["name"] or signature in seen:
            continue
        negatives.append(node)
        seen.add(signature)
    negatives.sort(key=lambda node: (node["role"] != target["role"],
                                     abs(nodes.index(node) - nodes.index(target)),
                                     node["id"]))
    if len(negatives) < 2:
        return None, "too_few_named_negatives"
    alternate = "fill" if op == "click" else "click"
    pairs = [(op, target), (op, negatives[0]), (op, negatives[1]),
             (alternate, target)]
    raw_id = f"{metadata['episode_id']}:{metadata['step_idx']}"
    shift = int(hashlib.sha256(raw_id.encode()).hexdigest()[:8], 16) % 4
    pairs = pairs[shift:] + pairs[:shift]
    gold_index = next(index for index, (action, node) in enumerate(pairs)
                      if action == op and node["id"] == bid)
    task_name = metadata["task_name"]
    split = "dev" if int(hashlib.sha256(task_name.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
    row = {"id": "ax-" + hashlib.sha256(raw_id.encode()).hexdigest()[:24],
           "split": split, "modality": "text",
           "task_family": "gui_ax_action_choice",
           "source": {"repo": REPO, "revision": REVISION, "row": raw_id,
                      "group": task_name, "shard": SHARD},
           "state": observations[0].strip() + "\nCurrent accessibility tree:\n" + tree.strip(),
           "question": {"type": "choice", "instructions": "Which action and labeled control should be used next?",
                        "criteria": {label: f"{action} [{node['id']}] {node['role']}: {node['name']}"
                                     for label, (action, node) in zip("ABCD", pairs)}},
           "gold": {"key": "ABCD"[gold_index]},
           "label_quality": "teacher_action_unverified", "usage": "research_only",
           "extra": {"operation": op, "target_bid": bid,
                     "teacher_model": metadata["teacher_model"]}}
    return row, None


@app.function(image=image, timeout=1200, volumes={"/gui-data": data})
def build():
    import hashlib
    from collections import Counter
    from pathlib import Path
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    root = Path(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(REPO, SHARD, repo_type="dataset", revision=REVISION)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    files = {split: (root / f"{split}.jsonl").open("w") for split in ("train", "dev")}
    counts = Counter()
    groups = {"train": set(), "dev": set()}
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for source in batch.to_pylist():
                counts["source_rows"] += 1
                row, reason = _convert(source)
                if reason:
                    counts[reason] += 1
                    continue
                files[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["split"]] += 1
                counts[f"operation:{row['extra']['operation']}"] += 1
                groups[row["split"]].add(row["source"]["group"])
    finally:
        for handle in files.values():
            handle.close()
    if groups["train"] & groups["dev"]:
        raise ValueError("Task leakage")
    summary = {"repo": REPO, "revision": REVISION, "shard": SHARD,
               "shard_sha256": digest, "counts": dict(counts),
               "task_groups": {key: len(value) for key, value in groups.items()},
               "split_sha256": {split: hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
                                for split in ("train", "dev")},
               "source_validation_split_read": False,
               "label_quality": "teacher_action_unverified"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    data.commit()
    return summary


@app.local_entrypoint()
def main():
    print(json.dumps(build.remote(), indent=2))
