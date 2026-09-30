"""Read-only dataset/model inventory and official benchmark discovery on Modal."""
import json
import modal

app = modal.App("vl-jev-benchmark-audit-20260930")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==1.33.0", "pyarrow==21.0.0", "requests==2.32.5")
volumes = {
    "/general-data": modal.Volume.from_name("vl-jev-general-v1-data"),
    "/general-runs": modal.Volume.from_name("vl-jev-general-v1-training"),
    "/pilot-data": modal.Volume.from_name("vl-jev-pilot-data"),
    "/gui-data": modal.Volume.from_name("vl-jev-gui-general-data"),
    "/gui-test": modal.Volume.from_name("vl-jev-gui-test-task-data"),
    "/ucf-data": modal.Volume.from_name("vl-jev-ucf101-full-data"),
    "/audit": modal.Volume.from_name("vl-jev-benchmark-audit", create_if_missing=True),
    "/bench": modal.Volume.from_name("vl-jev-official-benchmarks", create_if_missing=True),
}


@app.function(image=image, volumes=volumes, timeout=1800)
def audit():
    import hashlib
    from collections import Counter, defaultdict
    from pathlib import Path
    from huggingface_hub import HfApi

    roots = {"pilot": "/pilot-data/pilot_v1", "general_v1": "/general-data/v1",
             "general_v2": "/general-data/v2", "human_actions": "/general-data/v1/human_actions_v1",
             "soccer": "/general-data/soccer_vqa_2026", "ucf101": "/ucf-data/ucf101_full_v1",
             "mind2web": "/gui-data/mind2web_general_v1", "ax_actions": "/gui-data/ax_actions_general_v1",
             "mind2web_test": "/gui-test/mind2web_test_task_v1",
             "webintosh": "/general-data/webintosh_actions_v1"}
    inventory = {}
    for name, folder in roots.items():
        root = Path(folder)
        summaries = {p.name: json.loads(p.read_text()) for p in root.glob("*.json")
                     if p.name in {"summary.json", "stats.json", "sources.lock.json"}}
        splits = {}
        source_groups = defaultdict(set)
        media_digests = defaultdict(set)
        for split in ("train", "dev", "valid", "calibration", "test"):
            file = root / f"{split}.jsonl"
            if not file.is_file():
                continue
            sources, modalities, families, qualities, types = (Counter() for _ in range(5))
            for line in file.open():
                row = json.loads(line)
                source = row.get("source", {})
                sources[source.get("repo", "unknown")] += 1
                modalities[row.get("modality", "unknown")] += 1
                families[row.get("task_family", "unknown")] += 1
                qualities[row.get("label_quality", "unknown")] += 1
                types[row.get("question", {}).get("type", "unknown")] += 1
                if source.get("group") is not None:
                    source_groups[(source.get("repo"), str(source["group"]))].add(split)
                media = row.get("media") or {}
                if media.get("sha256"):
                    media_digests[media["sha256"]].add(split)
            splits[split] = {"rows": sum(sources.values()), "by_source": dict(sources),
                             "by_modality": dict(modalities), "by_family": dict(families),
                             "by_label_quality": dict(qualities), "by_type": dict(types),
                             "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        inventory[name] = {"path": folder, "present": root.exists(), "splits": splits,
                           "summaries": summaries,
                           "cross_split_source_groups": sum(len(s) > 1 for s in source_groups.values()),
                           "cross_split_media_hashes": sum(len(s) > 1 for s in media_digests.values())}
    reports = {}
    for run in ("general-head-v2", "general-head-v3-gui", "soccer-vqa-posttrain-full-v1",
                "soccer-vqa-posttrain-full-v1-direct-visual"):
        root = Path("/general-runs") / run
        reports[run] = {p.name: json.loads(p.read_text()) for p in root.glob("*.json")
                       if p.name in {"report.json", "report-uniform.json", "selection.json",
                                     "official-test-task.json", "retention-test.json",
                                     "soccer-test-evaluation.json", "pipeline-result.json", "test.json"}}
    api = HfApi(token=False)
    datasets = {}
    for repo in ("BLINK-Benchmark/BLINK", "lmms-lab/TempCompass", "OpenGVLab/MVBench",
                 "lmms-lab/Video-MME", "lmms-lab/MMAU", "lmms-lab/MVBench"):
        try:
            info = api.dataset_info(repo, files_metadata=True)
            datasets[repo] = {"revision": info.sha,
                              "files": [{"name": f.rfilename, "bytes": f.size} for f in info.siblings],
                              "card": info.card_data.to_dict() if info.card_data else {}}
        except Exception as error:
            datasets[repo] = {"error": f"{type(error).__name__}: {error}"}
    models = {}
    for repo in ("akhilaaa3/Jev-Omni", "ferdinandl007/jev-omni-general-head-v3",
                 "google/gemma-4-12B-it", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"):
        info = api.model_info(repo, files_metadata=True)
        models[repo] = {"revision": info.sha, "files": [f.rfilename for f in info.siblings],
                        "card": info.card_data.to_dict() if info.card_data else {}}
    result = {"inventory": inventory, "reports": reports, "benchmarks": datasets, "models": models}
    Path("/audit/inventory.json").write_text(json.dumps(result, indent=2))
    volumes["/audit"].commit()
    return {"datasets": {key: {"present": value["present"],
                               "splits": {s: {"rows": v["rows"], "by_source": v["by_source"]}
                                          for s, v in value["splits"].items()},
                               "cross_split_source_groups": value["cross_split_source_groups"],
                               "cross_split_media_hashes": value["cross_split_media_hashes"]}
                          for key, value in inventory.items()},
            "reports": {key: sorted(value) for key, value in reports.items()},
            "benchmarks": {key: {"revision": value.get("revision"),
                                 "files": value.get("files", [])[:45], "error": value.get("error")}
                           for key, value in datasets.items()},
            "models": {key: value["revision"] for key, value in models.items()},
            "saved": "/audit/inventory.json"}


@app.function(image=image, volumes=volumes, timeout=1800)
def probe():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    from collections import defaultdict
    import requests
    from huggingface_hub import HfApi
    result = {"decider_model_revision": HfApi(token=False).model_info("Mapika/decider-2b-vision").sha}
    response = requests.get("https://api.github.com/repos/Mapika/decider/commits/main", timeout=60)
    response.raise_for_status()
    result["decider_code_revision"] = response.json()["sha"]
    for repo, revision, filename in (
        ("BLINK-Benchmark/BLINK", "a3666eb249237ba3d5eca8db21176cc47967e040", "Spatial_Relation/val-00000-of-00001.parquet"),
        ("lmms-lab/TempCompass", "c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722", "multi-choice/test-00000-of-00001.parquet"),
        ("lmms-lab/TempCompass", "c8a67d88a3bc6fd4b2f9ae2f9e112668fbe05722", "yes_no/test-00000-of-00001.parquet")):
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir="/audit/hf-cache")
        table = pq.read_table(path)
        sample = table.slice(0, 1).to_pylist()[0]
        result[filename] = {"rows": len(table), "schema": str(table.schema),
                            "sample": {k: ({"path": v.get("path"), "bytes": len(v.get("bytes") or b"")} if isinstance(v, dict) and "bytes" in v else v) for k, v in sample.items()}}
    by_hash = defaultdict(list)
    for split in ("train", "dev"):
        for line in (Path("/gui-data/mind2web_general_v1") / f"{split}.jsonl").open():
            row = json.loads(line)
            if row.get("media", {}).get("sha256"):
                by_hash[row["media"]["sha256"]].append({"split": split, "id": row["id"], "question": row["question"], "gold": row["gold"]})
    result["gui_shared_screenshots"] = [v for v in by_hash.values() if len({r["split"] for r in v}) > 1]
    volumes["/audit"].commit()
    return result


@app.local_entrypoint()
def main(mode: str = "audit"):
    print(json.dumps(jevbench_probe.remote() if mode == "jevbench" else overlap.remote() if mode == "overlap" else probe.remote() if mode == "probe" else audit.remote(), indent=2))


@app.function(image=image, volumes=volumes, timeout=1800)
def overlap():
    import hashlib
    from collections import defaultdict
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    training = defaultdict(list)
    for file in ("/general-data/v2/train.jsonl", "/gui-data/mind2web_general_v1/train.jsonl", "/gui-data/ax_actions_general_v1/train.jsonl"):
        for line in Path(file).open():
            row=json.loads(line); media=row.get("media") or {}
            assets=media.get("assets",[]) if media.get("kind")=="modal_media_sequence" else [media]
            for asset in assets:
                if asset.get("sha256"):
                    training[asset["sha256"]].append(row["id"])
    overlaps=[]
    for line in Path("/bench/rows.jsonl").open():
        row=json.loads(line)
        for asset in row["images"]+([row["video"]] if row["video"] else []):
            if asset["sha256"] in training:
                overlaps.append({"benchmark_id":row["id"],"training_ids":training[asset["sha256"]]})
    gui_train={json.loads(line)["media"]["sha256"] for line in Path("/gui-data/mind2web_general_v1/train.jsonl").open()}
    gui_test=[json.loads(line) for line in Path("/gui-test/mind2web_test_task_v1/test_task.jsonl").open()]
    result={"known_general_gui_train_unique_media_hashes":len(training),"benchmark_exact_byte_overlaps":overlaps,
            "mind2web_test_shared_training_screenshots":[r["id"] for r in gui_test if r["media"]["sha256"] in gui_train],
            "limitations":"Exact file bytes only; transcoding, crops, near duplicates and upstream backbone pretraining remain unaudited. Soccer materials were filtered by source identity separately."}
    source=Path(hf_hub_download("akhilaaa3/Jev-Omni","jev_omni.py",revision="5addda86ddee081a68fb067477ea100c221b8917",cache_dir="/audit/hf-cache"))
    # Loader and checkpoint contract, for the consolidated release.
    lines=source.read_text().splitlines(); result["upstream_loader_source"]="\n".join(lines[-70:])
    config=Path(hf_hub_download("akhilaaa3/Jev-Omni","decision_config.json",revision="5addda86ddee081a68fb067477ea100c221b8917",cache_dir="/audit/hf-cache"))
    result["decision_config"]=json.loads(config.read_text())
    (Path("/audit")/"benchmark-overlap.json").write_text(json.dumps(result,indent=2)); volumes["/audit"].commit()
    return result


@app.function(image=image,volumes={"/audit":volumes["/audit"]},timeout=600)
def jevbench_probe():
    import requests
    from pathlib import Path
    response=requests.get("https://api.github.com/repos/fstandhartinger/jevbench/commits/main",timeout=60)
    response.raise_for_status();revision=response.json()["sha"]
    output={"revision":revision,"files":{}}
    root=Path("/audit/jevbench")/revision;root.mkdir(parents=True,exist_ok=True)
    for name in ("datasets/public/original.jsonl","datasets/public/easy.jsonl","datasets/public/hard.jsonl","jevbench/scoring.py"):
        response=requests.get(f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{revision}/{name}",timeout=60)
        response.raise_for_status();(root/name.replace("/","__")).write_text(response.text)
        output["files"][name]={"sample":json.loads(response.text.splitlines()[0]) if name.endswith(".jsonl") else response.text[:15000],"lines":len(response.text.splitlines())}
    volumes["/audit"].commit();return output
