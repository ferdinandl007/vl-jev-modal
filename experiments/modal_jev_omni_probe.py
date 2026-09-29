"""Pinned Jev-Omni choice probe on the existing VL Jev media cases.

Weights and media stay in Modal. This tests the published classifier directly,
not an implementation of the TypeSafe or VL Jev wire contract.
"""

import json

import modal


app = modal.App("vl-jev-omni-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.10.0", "torchvision==0.25.0", "transformers==5.17.0",
        "accelerate==1.13.0", "huggingface_hub", "safetensors", "numpy",
        "pillow==12.1.1", "opencv-python-headless", "soundfile", "librosa",
        "av==16.1.0",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
cache = modal.Volume.from_name("vl-jev-zero-shot-hf-cache", create_if_missing=True)
probe_media = modal.Volume.from_name("vl-jev-probe-media", create_if_missing=True)
MODEL_ID = "akhilaaa3/Jev-Omni"
MODEL_REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
SPORTS_REVISION = "d8ba9a491eea5c573525a37d42fd55144c12dd37"


@app.function(image=image, volumes={"/probe-media": probe_media}, timeout=1800)
def prepare_cases(include_sports=True):
    import csv
    import shutil
    from pathlib import Path

    import av
    from PIL import Image, ImageDraw
    from huggingface_hub import hf_hub_download

    root = Path("/probe-media")
    root.mkdir(exist_ok=True)
    cases = []
    positions = {
        "goal": [18 + i * 24 for i in range(8)],
        "miss": [18 + i * 24 for i in range(8)],
        "cross_and_return": [20, 50, 80, 120, 150, 180, 130, 100],
        "approach_and_return": [20, 50, 80, 120, 150, 150, 130, 100],
    }
    for name, xs in positions.items():
        path = root / f"omni_{name}.mp4"
        still = root / f"omni_{name}.png"
        if not path.exists() or not still.exists():
            output = av.open(str(path), mode="w")
            stream = output.add_stream("libx264", rate=8)
            stream.width = stream.height = 224
            stream.pix_fmt = "yuv420p"
            for x in xs:
                frame = Image.new("RGB", (224, 224), "#218b47")
                draw = ImageDraw.Draw(frame)
                draw.line((160, 20, 160, 204), fill="white", width=3)
                draw.rectangle((160, 54, 215, 150), outline="white", width=5)
                y = 178 if name == "miss" else 105
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="white", outline="black", width=2)
                for packet in stream.encode(av.VideoFrame.from_image(frame)):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
            output.close()
            frame.save(still)
        expected = "miss" if name in ("miss", "approach_and_return") else "goal"
        cases.append({"id": name, "kind": "synthetic_video", "path": str(path),
                      "expected": expected})
        if name in ("goal", "miss"):
            cases.append({"id": name, "kind": "synthetic_image", "path": str(still),
                          "expected": expected})
    if include_sports:
        repo = "infactory-ai/soccer-events"
        metadata = hf_hub_download(repo, "metadata.csv", repo_type="dataset",
                                   revision=SPORTS_REVISION,
                                   cache_dir="/probe-media/omni-dataset-cache")
        with open(metadata, newline="") as source:
            rows = list(csv.DictReader(source))
        for category in ("goal", "yellow_card", "red_card"):
            for row in [r for r in rows if r["event_category"] == category][:3]:
                destination = root / "public-sports" / row["mp4_file"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    source = hf_hub_download(repo, f"data/{row['mp4_file']}",
                                             repo_type="dataset", revision=SPORTS_REVISION,
                                             cache_dir="/probe-media/omni-dataset-cache")
                    shutil.copyfile(source, destination)
                cases.append({"id": row["asset_id"], "kind": "public_football",
                              "path": str(destination), "expected": category,
                              "duration_seconds": float(row["duration_seconds"])})
    probe_media.commit()
    return cases


@app.cls(image=image, gpu="H100", volumes={"/model-cache": cache,
                                          "/probe-media": probe_media}, timeout=3600)
class Classifier:
    @modal.enter()
    def load(self):
        import sys

        import torch
        from huggingface_hub import hf_hub_download, snapshot_download

        torch.set_grad_enabled(False)
        allow = ["config.json", "generation_config.json", "model*.safetensors*",
                 "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "decision_config.json", "head.pt"]
        checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                       allow_patterns=allow)
        source = hf_hub_download(MODEL_ID, "jev_omni.py", revision=MODEL_REVISION)
        sys.path.insert(0, str(__import__("pathlib").Path(source).parent))
        import jev_omni

        # Use the published loader, but force its otherwise unpinned snapshot
        # lookup to the exact repository revision inspected for this run.
        jev_omni.snapshot_download = lambda *_args, **_kwargs: checkpoint
        self.classifier = jev_omni.load_jev_omni()
        self.torch = torch
        cache.commit()

    @modal.method()
    def evaluate(self, cases, reverse=False, rotation=0):
        import time

        football = {
            "goal": "A goal is scored: the ball goes into the goal.",
            "yellow_card": "The referee shows a player a yellow card.",
            "red_card": "The referee shows a player a red card.",
        }
        goal = {
            "goal": "The ball enters the goal between the posts.",
            "miss": "The ball passes outside the goal posts.",
        }
        results = []
        for case in cases:
            kind = case["kind"]
            candidates = football if kind == "public_football" else goal
            labels = list(candidates)
            if reverse:
                labels.reverse()
            if rotation:
                labels = labels[rotation:] + labels[:rotation]
            options = [candidates[label] for label in labels]
            modality = "image" if kind == "synthetic_image" else "video"
            question = ("Which main soccer event is shown in this clip?"
                        if kind == "public_football" else
                        "Where is the ball relative to the goal?" if modality == "image"
                        else "At any point in this clip, did the ball enter the goal?")
            self.torch.cuda.synchronize()
            start = time.perf_counter()
            answer = self.classifier.predict(
                state="Soccer scene.", question=question, options=options,
                media=case["path"], modality=modality,
            )
            self.torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            probabilities = {label: answer["probabilities"][candidates[label]]
                             for label in labels}
            results.append({"id": case["id"], "kind": kind,
                            "expected": case["expected"],
                            "selected": labels[answer["prediction_index"]],
                            "probabilities": probabilities,
                            "confidence": answer["confidence"],
                            "elapsed_ms": round(elapsed_ms, 2),
                            **({"duration_seconds": case["duration_seconds"]}
                               if "duration_seconds" in case else {})})
        return {"model": MODEL_ID, "revision": MODEL_REVISION,
                "gpu": self.torch.cuda.get_device_name(), "reversed": reverse,
                "rotation": rotation,
                "cases": results}

    @modal.method()
    def text_smoke(self):
        result = self.classifier.predict(
            state="The meeting starts at 10 AM. It is now 9 AM.",
            question="Has the meeting started?", options=["Yes", "No"])
        return result

    @modal.method()
    def text_order_control(self):
        labels = ["red", "blue", "green"]
        results = []
        for rotation in range(3):
            ordered = labels[rotation:] + labels[:rotation]
            result = self.classifier.predict(
                state="The object is a red cube.",
                question="What color is the object?", options=ordered)
            results.append({"options": ordered,
                            "selected": result["prediction"],
                            "probabilities": result["probabilities"]})
        return results


@app.local_entrypoint()
def main(real_sports: bool = False):
    cases = prepare_cases.remote(include_sports=real_sports)
    model = Classifier()
    print(json.dumps({"text": model.text_smoke.remote()}, indent=2), flush=True)
    print(json.dumps({"text_order_control": model.text_order_control.remote()},
                     indent=2), flush=True)
    print(json.dumps(model.evaluate.remote(cases), indent=2), flush=True)
    # Repeat every case with reversed option slots. The head scores slot
    # positions, so order stability is part of the decision test.
    print(json.dumps(model.evaluate.remote(cases, reverse=True), indent=2), flush=True)
    if real_sports:
        print(json.dumps(model.evaluate.remote(
            [case for case in cases if case["kind"] == "public_football"],
            rotation=2), indent=2), flush=True)
