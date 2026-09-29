"""Check CARLA maneuver labels and episode separation on Modal."""

import json
import modal

app = modal.App("vl-jev-carla-split-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.5")


@app.function(image=image, timeout=600)
def probe():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from collections import Counter

    repo = "urjc-deepracer/carla-expert-racing"
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=8, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    response = session.get("https://datasets-server.huggingface.co/splits",
                           params={"dataset": repo}, timeout=90)
    response.raise_for_status()
    split_info = response.json()
    result = {"splits": split_info.get("splits"), "samples": {}}
    for split in ("train", "valid", "test"):
        response = session.get("https://datasets-server.huggingface.co/rows",
                               params={"dataset": repo, "config": "default",
                                       "split": split, "offset": 0, "length": 100},
                               timeout=90)
        response.raise_for_status()
        payload = response.json()
        rows = [item["row"] for item in payload["rows"]]
        result["samples"][split] = {
            "total": payload["num_rows_total"],
            "maneuvers": dict(Counter(str(row.get("maneuver")) for row in rows)),
            "experiments": dict(Counter(str(row.get("experiment_id")) for row in rows)),
            "image_present": sum(bool((row.get("image_path") or {}).get("src"))
                                 for row in rows),
            "first": {"maneuver": rows[0].get("maneuver"),
                      "steer": rows[0].get("steer"),
                      "experiment_id": rows[0].get("experiment_id")},
        }
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
