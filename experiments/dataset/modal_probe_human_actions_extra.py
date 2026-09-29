"""Modal-only discovery and media probe for extra human action datasets.

No media is fetched on the workstation. The remote function checks Hugging Face
metadata, split availability, row schemas and whether a sample image is decodable.
"""
import json
import modal

app = modal.App("vl-jev-extra-human-action-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests==2.32.5", "pillow==11.3.0"
)


@app.function(image=image, timeout=900)
def probe():
    import io
    import requests
    from PIL import Image

    s = requests.Session()
    s.headers.update({"User-Agent": "vl-jev-dataset-probe/1.0"})
    queries = ["Stanford40", "HMDB51", "Charades", "Something-Something V2",
               "HACS", "AVA actions", "MPII action", "human action recognition"]
    search = {}
    for query in queries:
        r = s.get("https://huggingface.co/api/datasets", params={"search": query, "limit": 8}, timeout=60)
        r.raise_for_status()
        search[query] = [
            {"id": d["id"], "private": d.get("private"), "gated": d.get("gated"),
             "sha": d.get("sha")}
            for d in r.json()
        ]

    candidates = [
        "zrchen03/Stanford40_Dataset", "innat/HMDB51", "jxie/hmdb51",
        "HuggingFaceM4/charades", "Bingsu/Human_Action_Recognition",
        "visual-layer/human-action-recognition-vl-enriched",
        "Shreyassanti/Human_Action_Recognition", "TornadoLabs/ava-actions",
        "kiyoonkim/hmdb51-gulprgb",
    ]
    result = {"search": search, "candidates": {}}
    for repo in candidates:
        item = {"repo": repo}
        meta = s.get(f"https://huggingface.co/api/datasets/{repo}", timeout=60)
        item["metadata_status"] = meta.status_code
        if not meta.ok:
            item["detail"] = meta.text[:180]
            result["candidates"][repo] = item
            continue
        md = meta.json()
        card = md.get("cardData") or {}
        siblings = [x.get("rfilename", "") for x in md.get("siblings", [])]
        item.update({"sha": md.get("sha"), "private": md.get("private"),
                     "gated": md.get("gated"), "license": card.get("license"),
                     "license_other": card.get("license_name") or card.get("license_link"),
                     "dataset_card": card, "file_count": len(siblings),
                     "sample_files": siblings[:14]})
        # Read dataset card and label schema at the immutable source revision.
        readme = s.get(f"https://huggingface.co/datasets/{repo}/resolve/{md.get('sha')}/README.md", timeout=60)
        item["readme_status"] = readme.status_code
        item["readme_excerpt"] = readme.text[:3500] if readme.ok else readme.text[:160]
        if "dataset_infos.json" in siblings:
            info_response = s.get(f"https://huggingface.co/datasets/{repo}/resolve/{md.get('sha')}/dataset_infos.json", timeout=60)
            item["dataset_infos_excerpt"] = info_response.text[:3500] if info_response.ok else None
        # The viewer API reveals actual config and split names without a dataset download.
        cfg = s.get("https://datasets-server.huggingface.co/splits",
                    params={"dataset": repo}, timeout=60)
        item["splits_status"] = cfg.status_code
        split_rows = []
        if cfg.ok:
            for sp in cfg.json().get("splits", [])[:20]:
                split_rows.append({"config": sp.get("config"), "split": sp.get("split")})
        item["splits"] = split_rows
        for split_record in split_rows:
            split_name = split_record["split"]
            if split_name not in ("train", "test", "validation", "valid", "val"):
                continue
            rows = s.get("https://datasets-server.huggingface.co/rows",
                         params={"dataset": repo, "config": split_record["config"],
                                 "split": split_name, "offset": 0, "length": 2}, timeout=120)
            item[f"{split_name}_rows_status"] = rows.status_code
            if not rows.ok:
                continue
            payload = rows.json()
            samples = [entry["row"] for entry in payload.get("rows", [])]
            item[f"{split_name}_rows"] = payload.get("num_rows_total")
            if split_name in ("train", "training") and samples:
                row = samples[0]
                item["row_fields"] = list(row)
                item["row_sample"] = {}
                for k, v in row.items():
                    if k.lower() in {"image", "video", "frames", "clip"}:
                        item["row_sample"][k] = ({"type": type(v).__name__, "keys": list(v)[:10]}
                                                   if isinstance(v, dict) else type(v).__name__)
                    else:
                        item["row_sample"][k] = str(v)[:180]
                for value in row.values():
                    if isinstance(value, dict) and value.get("src"):
                        try:
                            media = s.get(value["src"], timeout=60)
                            media.raise_for_status()
                            if value.get("src", "").lower().endswith((".mp4", ".webm", ".avi")) or media.headers.get("content-type", "").startswith("video/"):
                                item["sample_media"] = {"kind": "video", "http": media.status_code,
                                                         "bytes": len(media.content), "content_type": media.headers.get("content-type"),
                                                         "magic": media.content[:16].hex()}
                            else:
                                im = Image.open(io.BytesIO(media.content))
                                im.verify()
                                item["sample_media"] = {"kind": "image", "http": media.status_code,
                                                         "bytes": len(media.content), "format": im.format}
                        except Exception as e:
                            item["sample_media"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                        break
        result["candidates"][repo] = item
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2, ensure_ascii=False))
