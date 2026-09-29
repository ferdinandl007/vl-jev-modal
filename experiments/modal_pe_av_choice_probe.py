"""Modal-only PE-AV closed-choice probe for the VL Jev visual contract.

This measures similarity ranking, not calibrated answer probabilities. No model
weights or media are fetched by the local entrypoint.
"""

import json

import modal


app = modal.App("vl-jev-pe-av-choice-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.17.0",
        "timm==1.0.20",
        "av==16.1.0",
        "pillow==12.1.1",
        "torchcodec==0.10.0",
        "omegaconf==2.3.0",
        "sentencepiece==0.2.0",
        "tiktoken==0.9.0",
        "blobfile==3.0.0",
        "iopath==0.1.10",
        "einops==0.8.1",
        "ftfy==6.3.1",
    )
    .apt_install("git")
    .run_commands(
        "pip install --no-deps git+https://github.com/facebookresearch/perception_models.git@3e352cca660658d4b5c90f42a7808b11469e4c66"
    )
    .pip_install("xformers==0.0.35")
    .apt_install("ffmpeg")
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
cache = modal.Volume.from_name("vl-jev-zero-shot-hf-cache", create_if_missing=True)
probe_media = modal.Volume.from_name("vl-jev-probe-media", create_if_missing=True)
MODEL_NAME = "facebook/pe-av-small-16-frame"
# Native perception_models-format checkpoint, not the Transformers-format main branch.
MODEL_REVISION = "6f1fd6a42d00b4f6b76b11c6f329261d14e20df8"
SPORTS_REVISION = "d8ba9a491eea5c573525a37d42fd55144c12dd37"


@app.function(image=image, volumes={"/probe-media": probe_media}, timeout=120)
def create_synthetic_cases():
    from pathlib import Path

    import av
    from PIL import Image, ImageDraw

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
        path = root / f"pe_av_{name}.mp4"
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
        cases.append({"id": name, "path": str(path),
                      "expected": "no_goal" if name in ("miss", "approach_and_return") else "goal"})
    probe_media.commit()
    return cases


@app.function(image=image, volumes={"/probe-media": probe_media}, timeout=1800)
def prepare_sports_cases():
    import csv
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    repo = "infactory-ai/soccer-events"
    metadata = hf_hub_download(repo, "metadata.csv", repo_type="dataset",
                               revision=SPORTS_REVISION, cache_dir="/probe-media/pe-av-dataset-cache")
    with open(metadata, newline="") as source:
        rows = list(csv.DictReader(source))
    cases = []
    for category in ("goal", "yellow_card", "red_card"):
        for row in [r for r in rows if r["event_category"] == category][:3]:
            filename = f"data/{row['mp4_file']}"
            source = hf_hub_download(repo, filename, repo_type="dataset",
                                     revision=SPORTS_REVISION, cache_dir="/probe-media/pe-av-dataset-cache")
            destination = Path("/probe-media/public-sports") / row["mp4_file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
            cases.append({"id": row["asset_id"], "path": str(destination),
                          "expected": category, "duration_seconds": float(row["duration_seconds"])})
    probe_media.commit()
    return cases


@app.cls(image=image, gpu="L40S", volumes={"/model-cache": cache, "/probe-media": probe_media}, timeout=1800)
class Encoder:
    @modal.enter()
    def load(self):
        import torch
        from core.audio_visual_encoder import PEAudioVisual, PEAudioVisualTransform
        from huggingface_hub import snapshot_download

        self.torch = torch
        torch.set_grad_enabled(False)
        checkpoint = snapshot_download(repo_id=MODEL_NAME, revision=MODEL_REVISION)
        self.processor = PEAudioVisualTransform.from_config(checkpoint)
        self.model = PEAudioVisual.from_config(checkpoint, pretrained=True).to("cuda").eval()
        cache.commit()

    @modal.method()
    def score(self, cases, descriptions):
        import time

        torch = self.torch
        torch.cuda.synchronize()
        results = []
        for case in cases:
            started = time.perf_counter()
            inputs = self.processor(videos=[case["path"]], text=descriptions)
            preprocess_ms = (time.perf_counter() - started) * 1000
            inputs = inputs.to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                video_embeds = self.model.encode_video(
                    inputs["pixel_values_videos"],
                    padding_mask_videos=inputs.get("padding_mask_videos"),
                )
                text_embeds = self.model.encode_video_text(
                    inputs["input_ids"], attention_mask=inputs.get("attention_mask")
                )
            torch.cuda.synchronize()
            model_ms = (time.perf_counter() - started) * 1000
            scores = (video_embeds @ text_embeds.T)[0].float().cpu().tolist()
            results.append({**case, "selected": max(range(len(scores)), key=scores.__getitem__),
                            "similarities": [round(x, 5) for x in scores],
                            "preprocess_ms": round(preprocess_ms, 2), "model_ms": round(model_ms, 2)})
        return {"model": MODEL_NAME, "revision": MODEL_REVISION,
                "descriptions": descriptions, "gpu": torch.cuda.get_device_name(), "cases": results}


@app.local_entrypoint()
def main(real_sports: bool = False, balanced: bool = False):
    # All media creation/download and all inference stay inside Modal containers.
    score = Encoder().score
    cases = prepare_sports_cases.remote() if real_sports else create_synthetic_cases.remote()
    if real_sports:
        descriptions = ([
            "In this soccer clip, the main event is a goal.",
            "In this soccer clip, the main event is a yellow card.",
            "In this soccer clip, the main event is a red card.",
        ] if balanced else [
            "A soccer goal is scored and the ball goes into the net.",
            "A referee shows a player a yellow card.",
            "A referee shows a player a red card.",
        ])
    else:
        descriptions = ([
            "The soccer ball crosses the goal line between the posts.",
            "The soccer ball passes outside the goal posts.",
        ] if balanced else [
            "The soccer ball crosses the goal line between the posts.",
            "The soccer ball does not cross the goal line.",
        ])
    print(json.dumps(score.remote(cases, descriptions), indent=2))
