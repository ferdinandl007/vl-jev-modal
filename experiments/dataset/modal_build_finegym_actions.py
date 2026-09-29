"""Inspect/build a source-pinned FineGym Gym99 choice dataset on Modal only."""

import json
import modal

app = modal.App("vl-jev-finegym-gym99-adapter")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("huggingface_hub==1.32.0", "requests==2.32.5", "pillow==12.1.1", "numpy>=2,<3"))
data = modal.Volume.from_name("vl-jev-finegym-gym99-data", create_if_missing=True)
media = modal.Volume.from_name("vl-jev-finegym-gym99-media", create_if_missing=True)

REPO = "Lozumi/FineGym-skeleton"
REVISION = "8bb9b431c4213eb8f29438e41e878464c58cbcdd"
PICKLE_PATH = "Gym99-skeleton-V2.pkl"


@app.function(image=image, volumes={"/finegym-data": data}, timeout=1800, memory=4096)
def inspect_source():
    from huggingface_hub import HfApi, hf_hub_download
    from pathlib import Path
    import hashlib
    import pickle

    api = HfApi()
    info = api.dataset_info(REPO, revision=REVISION, files_metadata=True)
    current = api.dataset_info(REPO, files_metadata=False)
    if current.sha != REVISION:
        raise ValueError(f"HF source moved: expected {REVISION}, found {current.sha}")
    siblings = {item.rfilename: {"size": item.size, "blob_id": item.blob_id}
                for item in info.siblings if item.rfilename == PICKLE_PATH}
    if PICKLE_PATH not in siblings:
        raise FileNotFoundError(PICKLE_PATH)
    try:
        local = hf_hub_download(REPO, PICKLE_PATH, revision=REVISION,
                                cache_dir="/finegym-data/hf-cache")
    except Exception as exc:
        # Keep the inspection result actionable when repo metadata is public
        # but its large Xet object cannot be fetched anonymously.
        data.commit()
        return {"repo": REPO, "revision": REVISION,
                "file": siblings[PICKLE_PATH], "gated_metadata": info.gated,
                "row_data_accessible": False,
                "blocker": f"{type(exc).__name__}: {str(exc)[:300]}",
                "rows_built": 0, "media_smoke": "not run; no row manifest available"}

    # The source pickle contains only numpy arrays and builtin containers.
    # Restrict global resolution to prevent arbitrary pickle imports.
    class RestrictedUnpickler(pickle.Unpickler):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy", "ndarray"), ("numpy", "dtype"),
        }

        def find_class(self, module, name):
            if (module, name) in self.allowed:
                import numpy as np
                if name == "_reconstruct":
                    return np.core.multiarray._reconstruct
                if name == "scalar":
                    return np.core.multiarray.scalar
                return getattr(np, name)
            if module == "builtins" and name in {"dict", "list", "tuple", "set", "slice"}:
                return getattr(__import__(module), name)
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    with open(local, "rb") as handle:
        payload = RestrictedUnpickler(handle).load()
    annotations = payload["annotations"]
    return {
        "repo": REPO, "revision": REVISION,
        "file": siblings[PICKLE_PATH], "sha256": hashlib.sha256(Path(local).read_bytes()).hexdigest(),
        "top_keys": list(payload), "split_counts": {key: len(value) for key, value in payload["split"].items()},
        "annotation_count": len(annotations),
        "annotation_keys": list(annotations[0]),
        "annotation_example": {key: str(value)[:180] for key, value in annotations[0].items()
                                if key not in {"keypoint", "keypoint_score"}},
        "available_video_part_dirs": [x for x in api.list_repo_files(REPO, repo_type="dataset", revision=REVISION)
                                       if x.startswith("FineGym-RGB-subactions/")][:3],
    }


@app.local_entrypoint()
def main(mode: str = "inspect"):
    if mode != "inspect":
        raise ValueError("only --mode inspect is available until the upstream manifest is accessible")
    print(json.dumps(inspect_source.remote(), indent=2, ensure_ascii=False))
