"""Build and validate the complete candidate dataset inside a Modal function.

The local entrypoint only dispatches the function. Dataset JSONL and source
cache live in the Modal Volume vl-jev-pilot-data.
"""

from pathlib import Path

import modal


app = modal.App("vl-jev-pilot-builder")
builder = Path(__file__).with_name("build_pilot.py")
image = modal.Image.debian_slim(python_version="3.11").add_local_file(str(builder), "/root/build_pilot.py")
data_volume = modal.Volume.from_name("vl-jev-pilot-data", create_if_missing=True)
BUILDERS = ["soccer", "basketball", "driving", "counting", "nli", "intent",
            "visual_tool", "actions", "archive", "horse_qa", "sentiment"]


@app.function(image=image, volumes={"/dataset": data_volume}, timeout=3600, memory=4096)
def build_source(name: str):
    import json
    import os
    import runpy
    from pathlib import Path

    if name not in BUILDERS:
        raise ValueError(f"unknown source builder: {name}")
    output = Path("/dataset/pilot_v1")
    os.environ["VL_JEV_DATASET_OUT"] = str(output)
    cache = output / ".build_cache" / f"{name}.json"
    if cache.exists():
        return {"source": name, "rows": len(json.loads(cache.read_text())), "cached": True}
    module = runpy.run_path("/root/build_pilot.py", run_name="vl_jev_builder")
    try:
        rows = module[name]()
    except Exception as error:
        # urllib HTTPError holds an open response that Modal cannot serialize.
        raise RuntimeError(f"{name}: {error}") from None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False))
    data_volume.commit()
    return {"source": name, "rows": len(rows), "cached": False}


@app.function(image=image, volumes={"/dataset": data_volume}, timeout=3600, memory=4096)
def build_and_validate():
    import hashlib
    import json
    import os
    import runpy
    from pathlib import Path

    output = Path("/dataset/pilot_v1")
    os.environ["VL_JEV_DATASET_OUT"] = str(output)
    runpy.run_path("/root/build_pilot.py", run_name="__main__")
    stats = json.loads((output / "stats.json").read_text())
    checksums = {}
    for name in ["train.jsonl", "dev.jsonl", "calibration.jsonl", "test.jsonl",
                 "stats.json", "sources.lock.json"]:
        data = (output / name).read_bytes()
        checksums[name] = hashlib.sha256(data).hexdigest()
    ids = set()
    for split in ["train", "dev", "calibration", "test"]:
        for line in (output / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["id"] in ids or row["split"] != split or row["gold"]["key"] is None:
                raise ValueError("duplicate, wrong-split, or null-gold row")
            ids.add(row["id"])
    if len(ids) != stats["total"]:
        raise ValueError("row count mismatch")
    data_volume.commit()
    return {"volume": "vl-jev-pilot-data", "directory": str(output),
            "stats": stats, "sha256": checksums}


@app.local_entrypoint()
def main():
    import json

    for name in BUILDERS:
        print(json.dumps(build_source.remote(name)), flush=True)
    result = build_and_validate.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
