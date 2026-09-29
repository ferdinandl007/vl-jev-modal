"""Modal-only metadata probe for extra sports datasets; does not fetch media."""

import json
import modal

app = modal.App("vl-jev-extra-sports-metadata-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==1.32.0", "requests>=2.32,<3"
)


@app.function(image=image, timeout=300)
def inspect():
    from huggingface_hub import HfApi

    api = HfApi()
    repos = ["Lozumi/FineGym-skeleton", "lmwang/MultiSports",
             "shreyansh-sh/MultiSports", "mteb/diving48"]
    result = []
    for repo in repos:
        try:
            info = api.dataset_info(repo, files_metadata=False)
            result.append({
                "repo": repo,
                "exists": True,
                "revision": info.sha,
                "gated": info.gated,
                "private": info.private,
                "card": (info.card_data.to_dict() if info.card_data else {}),
                "files": api.list_repo_files(repo, repo_type="dataset")[:12],
            })
        except Exception as exc:
            result.append({"repo": repo, "exists": False,
                           "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
    # Look up alternate official spelling if the initial upstream identifier is absent.
    result.append({"search_multisports": [
        {"id": item.id, "private": item.private, "gated": item.gated}
        for item in api.list_datasets(search="MultiSports", limit=10)
    ]})
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2, ensure_ascii=False))
