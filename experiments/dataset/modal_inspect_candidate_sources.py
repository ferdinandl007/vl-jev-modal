"""Inspect public multimodal source schemas on Modal, without moving media locally."""

import json
import modal

app = modal.App("vl-jev-candidate-source-inspection")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("datasets>=3,<5", "huggingface_hub>=0.33", "pillow>=11"))


@app.function(image=image, timeout=1800)
def inspect_source(repo):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo)
    features = {}
    splits = {}
    for split in ("train", "validation", "val", "test"):
        try:
            stream = load_dataset(repo, split=split, streaming=True)
            row = next(iter(stream))
        except (ValueError, KeyError, StopIteration) as error:
            splits[split] = {"available": False, "error": type(error).__name__}
            continue
        splits[split] = {"available": True,
                         "fields": {key: ("image" if hasattr(value, "size") else
                                           "bytes" if isinstance(value, bytes) else
                                           type(value).__name__)
                                    for key, value in row.items()},
                         "choice_count": len(row.get("choices", []))
                         if isinstance(row.get("choices"), list) else None,
                         "image_present": bool(row.get("image")),
                         "sample_question": str(row.get("question", ""))[:160]}
    card = info.card_data.to_dict() if info.card_data else {}
    return {"repo": repo, "sha": info.sha, "gated": info.gated,
            "card_data": {key: value for key, value in card.items()
                          if key in {"license", "size_categories", "task_categories"}},
            "splits": splits}


@app.local_entrypoint()
def main():
    repos = ("HuggingFaceM4/A-OKVQA", "Gisiyuan/ScienceQA")
    print(json.dumps(list(inspect_source.map(repos)), ensure_ascii=False, indent=2))
