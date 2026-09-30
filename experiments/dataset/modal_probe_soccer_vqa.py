"""Inspect SoccerNet VQA archive metadata on Modal without fetching media locally."""

import json
import modal

app = modal.App("vl-jev-soccer-vqa-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==1.2.3", "requests==2.32.5")


@app.function(image=image, timeout=600)
def inspect():
    import io
    import requests
    import zipfile
    from huggingface_hub import HfApi

    repo = "SoccerNet/SN-VQA-2026"
    info = HfApi().dataset_info(repo, files_metadata=True)
    output = {"repo": repo, "revision": info.sha, "gated": info.gated,
              "files": [{"name": sibling.rfilename, "size": sibling.size}
                        for sibling in info.siblings]}
    for name in ("valid.zip", "test.zip"):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{info.sha}/{name}"
        response = requests.get(url, headers={"Range": "bytes=-65536"},
                                stream=True, timeout=120)
        output[name] = {"status": response.status_code,
                        "content_length": response.headers.get("Content-Length"),
                        "content_range": response.headers.get("Content-Range"),
                        "accept_ranges": response.headers.get("Accept-Ranges"),
                        "content_type": response.headers.get("Content-Type")}
        response.close()
    class RangeFile(io.RawIOBase):
        def __init__(self, url, size):
            self.url, self.size, self.pos = url, size, 0
            self.session = requests.Session()
            self.block = None
            self.block_start = -1

        def seekable(self):
            return True

        def readable(self):
            return True

        def tell(self):
            return self.pos

        def seek(self, offset, whence=0):
            self.pos = (offset if whence == 0 else
                        self.pos + offset if whence == 1 else self.size + offset)
            return self.pos

        def read(self, length=-1):
            if length < 0:
                length = self.size - self.pos
            parts = []
            while length > 0 and self.pos < self.size:
                start = (self.pos // 1_048_576) * 1_048_576
                if start != self.block_start:
                    end = min(start + 1_048_576, self.size) - 1
                    response = self.session.get(self.url,
                        headers={"Range": f"bytes={start}-{end}"}, timeout=180)
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise RuntimeError("Source stopped supporting byte ranges")
                    self.block = response.content
                    self.block_start = start
                offset = self.pos - start
                chunk = self.block[offset:offset + length]
                if not chunk:
                    raise IOError("Empty range response")
                parts.append(chunk)
                self.pos += len(chunk)
                length -= len(chunk)
            return b"".join(parts)

    from collections import Counter
    for name in ("train.zip", "valid.zip", "test.zip"):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{info.sha}/{name}"
        size = next(sibling.size for sibling in info.siblings if sibling.rfilename == name)
        with zipfile.ZipFile(RangeFile(url, size)) as archive:
            members = archive.infolist()
            candidates = [member for member in members if member.filename.lower().endswith(".json")]
            record = output.setdefault(name, {})
            record["member_count"] = len(members)
            record["first_members"] = [member.filename for member in members[:8]]
            record["json_members"] = [(member.filename, member.file_size)
                                      for member in candidates[:20]]
            if candidates:
                blob = archive.read(candidates[0])
                parsed = json.loads(blob)
                if isinstance(parsed, list):
                    counts = Counter(str(row.get("task type", "missing")) for row in parsed)
                    material_counts = Counter(len(row.get("materials") or []) for row in parsed)
                    extensions = Counter(str(path).rsplit(".", 1)[-1].lower()
                                         for row in parsed for path in (row.get("materials") or []))
                    archive_names = set(archive.namelist())
                    prefix = name.removesuffix(".zip") + "/"
                    missing = [path for row in parsed for path in (row.get("materials") or [])
                               if prefix + path not in archive_names]
                    record["json_shape"] = {"count": len(parsed), "first": parsed[0] if parsed else None,
                                            "row_51": parsed[51] if name == "train.zip" else None,
                                            "row_130": parsed[130] if name == "train.zip" else None,
                                            "task_types": dict(counts),
                                            "materials_per_question": dict(material_counts),
                                            "extensions": {key: value for key, value in extensions.items()
                                                           if key in {"mp4", "jpg", "jpeg", "png"}},
                                            "missing_materials": len(missing),
                                            "missing_examples": missing[:3],
                                            "missing_prefix_matches": [
                                                [entry for entry in archive_names
                                                 if entry.startswith(prefix + path + "/")][:3]
                                                for path in missing[:2]],
                                            "task_examples": {
                                                task: [{"question": row.get("Q"),
                                                        "materials": (row.get("materials") or [])[:2],
                                                        "answer": row.get("closeA")}
                                                       for row in parsed if row.get("task type") == task][:1]
                                                for task in counts} if name == "train.zip" else {}}
                else:
                    record["json_shape"] = {"keys": list(parsed)[:8]}
    return output


@app.local_entrypoint()
def main(compact: bool = True):
    result = inspect.remote()
    if compact:
        result = {name: {
            "member_count": result[name]["member_count"],
            "row_count": result[name]["json_shape"]["count"],
            "task_types": result[name]["json_shape"]["task_types"],
            "materials_per_question": result[name]["json_shape"]["materials_per_question"],
            "missing_materials": result[name]["json_shape"]["missing_materials"],
            "missing_examples": result[name]["json_shape"]["missing_examples"],
            "missing_prefix_matches": result[name]["json_shape"]["missing_prefix_matches"],
            "row_51": result[name]["json_shape"]["row_51"],
            "row_130": result[name]["json_shape"]["row_130"],
        } for name in ("train.zip", "valid.zip", "test.zip")}
    print(json.dumps(result, indent=2))
