"""Training-free video decisions on Modal. All media and weights stay remote."""

import json
import time

import modal


app = modal.App("vl-jev-zero-shot-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        "torchvision",
        "transformers==5.10.2",
        "accelerate==1.13.0",
        "av==16.1.0",
        "pillow==12.1.1",
        "numpy==2.4.2",
        "requests==2.32.5",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
cache = modal.Volume.from_name("vl-jev-zero-shot-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="L4", volumes={"/model-cache": cache}, timeout=1800)
def run_probe():
    from pathlib import Path

    import av
    import numpy as np
    import requests
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, XCLIPModel

    torch.set_grad_enabled(False)
    device = "cuda"
    cases = [
        {
            "id": "reading",
            "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/sample_demo_1.mp4",
            "choices": ["a child reading a book", "a person playing tennis", "a person eating spaghetti"],
            "correct": 0,
        },
        {
            "id": "tennis",
            "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
            "choices": ["a child reading a book", "a person playing tennis", "a person eating spaghetti"],
            "correct": 1,
        },
        {
            "id": "spaghetti",
            "url": "https://huggingface.co/datasets/nielsr/video-demo/resolve/main/eating_spaghetti.mp4",
            "choices": ["a child reading a book", "a person playing tennis", "a person eating spaghetti"],
            "correct": 2,
        },
    ]

    def sync():
        torch.cuda.synchronize()

    def make_motion_video(path: Path, reverse: bool):
        output = av.open(str(path), mode="w")
        stream = output.add_stream("libx264", rate=8)
        stream.width = 224
        stream.height = 224
        stream.pix_fmt = "yuv420p"
        for i in range(8):
            frame = Image.new("RGB", (224, 224), "white")
            draw = ImageDraw.Draw(frame)
            step = 7 - i if reverse else i
            x = 14 + step * 24
            draw.rectangle((x, 90, x + 30, 120), fill="red")
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()

    def sample_frames(path: Path, count: int = 8):
        container = av.open(str(path))
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        container.close()
        if not frames:
            raise ValueError(f"No video frames in {path}")
        indices = np.linspace(0, len(frames) - 1, count).astype(int)
        return [frames[i] for i in indices], len(frames)

    media_dir = Path("/tmp/vl-jev-probe")
    media_dir.mkdir(exist_ok=True)
    for case in cases:
        local_path = media_dir / (case["id"] + ".mp4")
        response = requests.get(case["url"], timeout=90)
        response.raise_for_status()
        local_path.write_bytes(response.content)
        case["path"] = local_path
    for reverse in (False, True):
        case_id = "square_right_to_left" if reverse else "square_left_to_right"
        local_path = media_dir / (case_id + ".mp4")
        make_motion_video(local_path, reverse)
        cases.append(
            {
                "id": case_id,
                "path": local_path,
                "choices": ["a red square moving from left to right", "a red square moving from right to left"],
                "correct": 1 if reverse else 0,
            }
        )

    xclip_name = "microsoft/xclip-base-patch32"
    t0 = time.perf_counter()
    x_processor = AutoProcessor.from_pretrained(xclip_name)
    x_model = XCLIPModel.from_pretrained(xclip_name).to(device).eval()
    sync()
    x_load_s = time.perf_counter() - t0
    x_results = []
    for case in cases:
        t0 = time.perf_counter()
        frames, source_frame_count = sample_frames(case["path"])
        x_inputs = x_processor(
            text=case["choices"], videos=frames, return_tensors="pt", padding=True
        ).to(device)
        prep_ms = (time.perf_counter() - t0) * 1000
        sync()
        t0 = time.perf_counter()
        x_output = x_model(**x_inputs)
        sync()
        model_ms = (time.perf_counter() - t0) * 1000
        scores = x_output.logits_per_video[0].float()
        probabilities = scores.softmax(dim=-1).cpu().tolist()
        x_results.append(
            {
                "case": case["id"],
                "source_frames": source_frame_count,
                "sampled_frames": len(frames),
                "correct": case["correct"],
                "selected": int(scores.argmax().item()),
                "probabilities": [round(p, 4) for p in probabilities],
                "preprocess_ms": round(prep_ms, 2),
                "model_ms": round(model_ms, 2),
            }
        )
    del x_model
    torch.cuda.empty_cache()

    qwen_name = "Qwen/Qwen3.5-2B"
    t0 = time.perf_counter()
    q_processor = AutoProcessor.from_pretrained(
        qwen_name, max_image_size={"longest_edge": 50176}
    )
    q_processor.video_processor.fps = None
    q_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        qwen_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    sync()
    q_load_s = time.perf_counter() - t0
    tokenizer = q_processor.tokenizer
    q_results = []
    for case in cases:
        for rotate in (False, True):
            order = list(range(len(case["choices"])))
            if rotate:
                order = order[1:] + order[:1]
            ordered_choices = [case["choices"][i] for i in order]
            labels = [chr(ord("A") + i) for i in range(len(order))]
            label_ids = []
            for label in labels:
                ids = tokenizer.encode(label, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Answer label {label!r} is not one token: {ids}")
                label_ids.append(ids[0])
            question = "Which description best matches this video?\n" + "\n".join(
                f"{label}. {choice}" for label, choice in zip(labels, ordered_choices)
            ) + "\nReply with one letter only: "
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": str(case["path"])},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            t0 = time.perf_counter()
            inputs = q_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                num_frames=8,
            ).to(device)
            prep_ms = (time.perf_counter() - t0) * 1000
            sync()
            t0 = time.perf_counter()
            output = q_model(**inputs, logits_to_keep=1, use_cache=False)
            sync()
            model_ms = (time.perf_counter() - t0) * 1000
            all_logits = output.logits[0, -1].float()
            choice_logits = all_logits[label_ids]
            probabilities = choice_logits.softmax(dim=-1).cpu().tolist()
            selected_position = int(choice_logits.argmax().item())
            full_probs = all_logits.softmax(dim=-1)
            label_mass = float(full_probs[label_ids].sum().item())
            top_token = tokenizer.decode([int(all_logits.argmax().item())])
            q_results.append(
                {
                    "case": case["id"],
                    "rotated": rotate,
                    "correct": case["correct"],
                    "selected": order[selected_position],
                    "probabilities_by_original_choice": {
                        str(original): round(probabilities[position], 4)
                        for position, original in enumerate(order)
                    },
                    "unconditional_label_mass": round(label_mass, 6),
                    "top_next_token": top_token,
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                    "preprocess_ms": round(prep_ms, 2),
                    "model_ms": round(model_ms, 2),
                }
            )

    cache.commit()
    return {
        "model_versions": {"xclip": xclip_name, "qwen": qwen_name},
        "gpu": torch.cuda.get_device_name(),
        "load_seconds": {"xclip": round(x_load_s, 2), "qwen": round(q_load_s, 2)},
        "xclip": x_results,
        "qwen": q_results,
        "note": "Normalized candidate scores are not calibrated correctness probabilities.",
    }


@app.local_entrypoint()
def main():
    result = run_probe.remote()
    print(json.dumps(result, indent=2, ensure_ascii=False))
