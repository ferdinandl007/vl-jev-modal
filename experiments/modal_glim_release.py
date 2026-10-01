"""Metadata-only naming and decision protocol publication, executed on Modal."""
import modal
app=modal.App("glim-release-metadata")
image=modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub==1.33.0").add_local_python_source("experiments").add_local_file("docs/OLLAMA_COMPATIBILITY.md","/root/OLLAMA_COMPATIBILITY.md")
release=modal.Volume.from_name("vl-jev-publication-artifacts")
OLD="ferdinandl007/jev-omni-unified-v4"
NEW="ferdinandl007/glim-12b"
HEAD="e0bfd32690d4656238e29e402b01772ca93e97602c12875b84f61793eef169c6"

@app.function(image=image,volumes={"/release":release},secrets=[modal.Secret.from_name("huggingface-secret")],timeout=300,memory=1024)
def publish():
    import hashlib,json,os
    from pathlib import Path
    from huggingface_hub import HfApi,hf_hub_download
    from experiments.systemone_adapter import evaluate_systemone
    # CPU protocol checks deliberately do not load any model or claim accuracy.
    class ContractBackend:
        def predict(self,**kwargs):
            opts=kwargs["options"]
            return {"probabilities":{v:1/len(opts) for v in opts},"confidence":0.}
    req={"model":"Glim 12B","state":{"ticket":"Please refund the duplicate charge."},"questions":{
        "team":{"type":"choice","instructions":"Select a team","criteria":{"billing":None,"other":"Other"}},
        "refund":{"type":"noul","instructions":"Refund requested?"},
        "urgency":{"type":"score","instructions":"Urgency?","criteria":["Routine","Urgent"]}}}
    answer=evaluate_systemone(ContractBackend(),req,"Glim 12B")
    assert answer["answers"]["team"]["choice"]=="billing"
    assert answer["answers"]["refund"]["noul"]==.5 and answer["answers"]["urgency"]["score"]==.5
    assert set(answer)=={"model","answers"}
    token=next((os.environ[k] for k in ("HF_TOKEN","HUGGINGFACE_TOKEN","HUGGINGFACE_HUB_TOKEN") if os.environ.get(k)),None)
    if not token:raise ValueError("Missing managed publication token")
    api=HfApi(token=token)
    if not api.repo_exists(NEW):api.move_repo(from_id=OLD,to_id=NEW,repo_type="model")
    info=HfApi(token=False).model_info(NEW)
    head=Path(hf_hub_download(NEW,"head.pt",revision=info.sha,token=False))
    if hashlib.sha256(head.read_bytes()).hexdigest()!=HEAD:raise ValueError("Glim head differs")
    card=Path(hf_hub_download(NEW,"README.md",revision=info.sha,token=False)).read_text()
    card=card.replace("# Jev-Omni unified v4","# Glim 12B").replace(OLD,NEW)
    card+='\n## Ollama compatibility\n\nGlim 12B is the public name of this audited unified v4 checkpoint; weights and benchmark scores are unchanged. See `OLLAMA_COMPATIBILITY.md` for the verified runtime differences and publication path. `systemone_adapter.py` preserves the native head behind the shared request/answer shape, but is not a stock Ollama runner. No `ollama pull` or quantized parity claim is made. The new 117,330-example text corpus is not yet included in this checkpoint.\n'
    api.upload_file(repo_id=NEW,path_or_fileobj=card.encode(),path_in_repo="README.md",commit_message="Name the published model Glim 12B")
    for name,path in (("OLLAMA_COMPATIBILITY.md","/root/OLLAMA_COMPATIBILITY.md"),("systemone_adapter.py","/root/experiments/systemone_adapter.py")):
        content=Path(path).read_text()
        if name.endswith(".py"):content=content.replace("from experiments.typed_decisions import evaluate_typed","from typed_decisions import evaluate_typed")
        api.upload_file(repo_id=NEW,path_or_fileobj=content.encode(),path_in_repo=name,commit_message="Publish native-head decision API compatibility research and adapter")
    public=HfApi(token=False).model_info(NEW)
    for name in ("README.md","OLLAMA_COMPATIBILITY.md","systemone_adapter.py"):
        Path(hf_hub_download(NEW,name,revision=public.sha,token=False)).read_text()
    legacy=HfApi(token=False).model_info(OLD)
    if legacy.sha!=public.sha:raise ValueError("Legacy repository redirect differs")
    result={"repo":NEW,"revision":public.sha,"head_sha256":HEAD,"old_link_redirect_verified":True,"protocol_adapter_checked":True,"native_ollama_runtime_verified":False}
    (Path("/release")/"glim-publication.json").write_text(json.dumps(result,indent=2));release.commit();return result

@app.local_entrypoint()
def main():
    print(publish.remote())
