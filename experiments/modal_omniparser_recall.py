"""Measure OmniParser detector coverage for held-out Webintosh targets on Modal."""

import json
import modal

from experiments.modal_jev_omni_sports_train import base_image, general_data, general_media, model_cache

app = modal.App("vl-jev-omniparser-recall")


@app.function(image=base_image, gpu="H100", timeout=3600,
              volumes={"/general-data": general_data,
                       "/general-media": general_media,
                       "/model-cache": model_cache})
def measure(limit: int = 40):
    import hashlib
    import re
    from pathlib import Path
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.ops import nms
    from huggingface_hub import hf_hub_download

    if not 1 <= limit <= 155:
        raise ValueError("Limit must be 1..155 dev rows")
    path = hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect_v3/model.pt",
                           revision="refs/pr/37")
    checkpoint_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    detector = torch.jit.load(path, map_location="cuda").eval()
    rows = [row for row in (json.loads(line) for line in
            Path("/general-data/v2/dev.jsonl").open())
            if row["task_family"] == "gui_next_click"][:limit]
    report = {"rows": len(rows), "target_center_covered": 0,
              "target_center_top10_covered": 0,
              "detected_counts": [], "checkpoint_sha256": checkpoint_sha256,
              "source": "microsoft/OmniParser-v2.0 icon_detect_v3 refs/pr/37"}
    for index, row in enumerate(rows):
        source = Image.open(row["media"]["path"]).convert("RGB")
        width, height = source.size
        label = row["gold"]["key"]
        option = row["question"]["criteria"][label]
        match = re.search(r"\((\d+), (\d+)\)", option)
        if not match:
            raise ValueError("Missing target center")
        x_gold = int(match.group(1)) * width / 1000
        y_gold = int(match.group(2)) * height / 1000
        scale = min(1280 / width, 1280 / height)
        resized = source.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
        canvas = Image.new("RGB", (1280, 1280), (114, 114, 114))
        canvas.paste(resized, (0, 0))
        tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1)
        tensor = tensor.float()[None].to("cuda") / 255.0
        with torch.no_grad():
            outputs = detector(tensor)
        boxes, scores = [], []
        for level, stride in enumerate((8, 16, 32)):
            cls = outputs[2 * level].sigmoid()[0, 0]
            ltrb = outputs[2 * level + 1][0]
            grid = cls.shape[-1]
            gy, gx = torch.meshgrid(torch.arange(grid, device="cuda"),
                                    torch.arange(grid, device="cuda"), indexing="ij")
            cx, cy = gx + 0.5, gy + 0.5
            left, top, right, bottom = ltrb
            boxes.append(torch.stack(
                [(cx - left) * stride, (cy - top) * stride,
                 (cx + right) * stride, (cy + bottom) * stride], dim=-1
            ).reshape(-1, 4))
            scores.append(cls.reshape(-1))
        boxes, scores = torch.cat(boxes), torch.cat(scores)
        keep = scores > 0.05
        boxes, scores = boxes[keep], scores[keep]
        keep = nms(boxes, scores, 0.45)
        boxes = boxes[keep] / scale
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        hits = ((boxes[:, 0] <= x_gold) & (x_gold <= boxes[:, 2]) &
                (boxes[:, 1] <= y_gold) & (y_gold <= boxes[:, 3]))
        report["target_center_covered"] += int(bool(hits.any()))
        report["target_center_top10_covered"] += int(bool(hits[:10].any()))
        report["detected_counts"].append(len(boxes))
        if (index + 1) % 10 == 0:
            print(json.dumps({"processed": index + 1,
                              "covered": report["target_center_covered"]}), flush=True)
    report["coverage"] = report["target_center_covered"] / report["rows"]
    report["top10_coverage"] = report["target_center_top10_covered"] / report["rows"]
    report["mean_candidates"] = sum(report["detected_counts"]) / report["rows"]
    report.pop("detected_counts")
    return report


@app.local_entrypoint()
def main(limit: int = 40):
    print(json.dumps(measure.remote(limit), indent=2))
