"""Publish verified benchmark visuals and baseline labels from Modal."""
import modal
app=modal.App('glim-benchmark-charts')
image=modal.Image.debian_slim(python_version='3.11').pip_install('huggingface_hub==1.33.0').add_local_dir('docs/assets','/charts')
@app.function(image=image,secrets=[modal.Secret.from_name('huggingface-secret')],timeout=300,memory=512)
def publish():
    import os,re
    from pathlib import Path
    from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd
    repo='ferdinandl007/glim-12b'
    token=next((os.environ[k] for k in ('HF_TOKEN','HUGGINGFACE_TOKEN','HUGGINGFACE_HUB_TOKEN') if os.environ.get(k)),None)
    if not token:raise ValueError('Missing managed publication token')
    api=HfApi(token=token);before=api.model_info(repo)
    card=Path(hf_hub_download(repo,'README.md',revision=before.sha,token=token)).read_text()
    section='''<!-- benchmark-visuals:start -->
## Benchmark snapshot

![Glim compared with open Jev-Omni, Gemma and Qwen](assets/benchmark-comparison.svg)

![Public JevBench comparison](assets/jevbench-comparison.svg)

**Glim 12B is unified v4.** Jev-Omni and Glim use a trained decision head. Gemma and Qwen are **unmodified base-model decision-scoring baselines**: one forward pass, thinking disabled, candidate answer-token logits, without generated explanations or agent loops. They were not fine-tuned into dedicated decision models for these results. This measures constrained decision accuracy; it does not establish equal architectures, runtime efficiency or a native-generation leaderboard rank.

TypeSafe AI's hosted Jev is separate from the original open Jev-Omni checkpoint shown here. A matching TypeSafe score is pending; no score is inferred from other benchmark scopes. The planned Qwen LoRA candidate has no verified result yet.

[Complete protocol, counts and limitations](https://github.com/ferdinandl007/vl-jev-modal/blob/main/docs/UNIFIED_RELEASE_AND_BENCHMARKS.md).
<!-- benchmark-visuals:end -->
'''
    card=re.sub(r'<!-- benchmark-visuals:start -->.*?<!-- benchmark-visuals:end -->\n?', '',card,flags=re.S)
    # Keep YAML metadata at the top, then place the chart after the card heading.
    match=re.search(r'^# .+$',card,re.M)
    if not match:raise ValueError('Model card heading missing')
    card=card[:match.end()]+'\n\n'+section+'\n'+card[match.end():].lstrip('\n')
    files={'README.md':card.encode()}
    for name in ('benchmark-comparison.svg','jevbench-comparison.svg'):files['assets/'+name]=Path('/charts',name).read_bytes()
    commit=api.create_commit(repo_id=repo,parent_commit=before.sha,operations=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=v) for k,v in files.items()],commit_message='Add benchmark charts and clarify base-model decision-scoring baselines')
    public=HfApi(token=False)
    for name,content in files.items():
        if Path(hf_hub_download(repo,name,revision=commit.oid,token=False)).read_bytes()!=content:raise ValueError('Anonymous publication readback differs')
    print({'repo':repo,'revision':commit.oid,'charts':2,'anonymous_readback':True})
@app.local_entrypoint()
def main():publish.remote()
