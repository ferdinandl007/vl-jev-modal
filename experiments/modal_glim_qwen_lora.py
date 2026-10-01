"""Bounded multimodal Qwen3.5-9B LoRA pilot; all data/compute stay on Modal."""
import json, modal
from experiments.modal_official_benchmarks import gpu_image, MODELS
from experiments.modal_jev_omni_sports_train import model_cache,general_data,general_media,ucf_media,media_volume
from experiments.modal_text_decisions import data as text_data
app=modal.App("glim-qwen9b-lora-pilot")
output=modal.Volume.from_name("glim-qwen-lora",create_if_missing=True)
MODEL,REVISION=MODELS["qwen9b"]
RUN="qwen9b-ollama-lora-pilot-v1"
image=gpu_image.pip_install("peft==0.21.1").add_local_python_source("experiments")
mounts={"/model-cache":model_cache,"/general-data":general_data,"/general-media":general_media,"/ucf-media":ucf_media,"/pilot":media_volume,"/text-data":text_data,"/output":output}


def compiled(row):
    """Match Ollama decision JSON field order, candidate codes and requested field."""
    from experiments.modal_text_decisions import typed_options
    question=row["question"];keys,_=typed_options(question);criteria=question.get("criteria")
    if len(keys)>26:raise ValueError("Ollama supports at most 26 candidates")
    if question["type"]=="noul":values=[False,True];descriptions=[(criteria or {}).get("false","No"),(criteria or {}).get("true","Yes")]
    elif question["type"]=="choice":values=keys;descriptions=[criteria[k] if criteria[k] is not None else k for k in keys]
    else:values=keys;descriptions=criteria
    text=lambda v:v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,separators=(",",":"))
    name="decision"
    schema={"name":name,"description":text(question["instructions"]),"choices":[{"code":chr(65+i),"value":v,"description":text(d)} for i,(v,d) in enumerate(zip(values,descriptions))]}
    prompt=json.dumps({"context":text(row["state"]),"schema":[schema]},ensure_ascii=False,separators=(",",":"))+'\n\nRequested field: '+json.dumps(name)
    return prompt,keys,keys.index(row["gold"]["key"])


@app.function(image=image,volumes=mounts,timeout=600,memory=4096,cpu=(2,2))
def prepare():
    import hashlib,random
    from collections import Counter
    from pathlib import Path
    from experiments.modal_text_decisions import text_rows
    from experiments.modal_jev_omni_sports_train import _general_rows
    root=Path("/output")/RUN;root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists():return json.loads((root/"manifest.json").read_text())
    def eligible(row):
        try:compiled(row)
        except (ValueError,KeyError,TypeError):return False
        if len(str(row["state"]))>3000:return False
        if row["modality"]=="text":return True
        media=row.get("media") or {}
        if media.get("kind") not in {"modal_image","modal_video"}:return False
        return Path(media["path"]).is_file()
    manifest={"run":RUN,"model":MODEL,"revision":REVISION,"seed":3407,"split_counts":{},"files":{},"skipped":{},"protocol":"Ollama-shaped schema and A-Z candidate logits; media is an explicit extension; no chain of thought"}
    for split in ("train","dev","test"):
        candidates=text_rows(split)+[r for r in _general_rows(split,"v2") if r["modality"] in {"image","video"}]
        groups={kind:[] for kind in ("text","image","video")}
        for row in candidates:
            if eligible(row):groups[row["modality"]].append(row)
        selected=[]
        for kind,group in groups.items():
            group.sort(key=lambda r:hashlib.sha256(r["id"].encode()).hexdigest())
            cap=320 if split=="train" else 32
            # Text sampling round-robins task families so routing cannot dominate.
            families={}
            for r in group:families.setdefault(r.get("task_family",kind),[]).append(r)
            balanced=[]
            while any(families.values()) and len(balanced)<cap:
                for key in sorted(families):
                    if families[key] and len(balanced)<cap:balanced.append(families[key].pop())
            selected.extend(balanced)
            if len(balanced)<cap:raise ValueError(f"Insufficient accessible {kind} {split}: {len(balanced)}/{cap}; mounted paths must be repaired")
        random.Random(3407).shuffle(selected)
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
        path=root/f"{split}.jsonl";path.write_text(content)
        manifest["split_counts"][split]=dict(Counter(r["modality"] for r in selected))
        manifest["files"][split]=hashlib.sha256(content.encode()).hexdigest()
    manifest["train_rows"]=960;manifest["dev_rows"]=96;manifest["test_rows"]=96
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2));output.commit();return manifest


def conversation(row):
    from pathlib import Path
    import cv2
    from PIL import Image
    prompt,keys,target=compiled(row);content=[];media=row.get("media") or {}
    if media.get("kind")=="modal_media_sequence":media_assets=media["assets"]
    elif media.get("kind") in {"modal_image","modal_video"}:media_assets=[{"kind":"video" if media["kind"]=="modal_video" else "image","path":media["path"]}]
    else:media_assets=[]
    if not media_assets and row["modality"]!="text":
        folder=Path("/pilot")/row["split"]/row["id"]
        media_assets=[{"kind":"image","path":str(p)} for p in sorted(folder.glob("image_*.jpg"))]
        if not media_assets:raise ValueError("Missing media input: "+row["id"])
    for asset_index,asset in enumerate(media_assets):
        if len(media_assets)>1:content.append({"type":"text","text":f"Source asset {asset_index+1} of {len(media_assets)}; each clip has its own timeline."})
        if asset["kind"]=="image":
            with Image.open(asset["path"]) as im:content.append({"type":"image","image":im.convert("RGB")})
        else:
            cap=cv2.VideoCapture(asset["path"])
            try:
                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
                if n<1 or fps<=0:raise ValueError("Invalid video metadata: "+row['id'])
                for j in range(4):
                    pos=min(n-1,int((j+.5)*n/4));cap.set(cv2.CAP_PROP_POS_FRAMES,pos);ok,frame=cap.read()
                    if not ok:raise ValueError("Video decode failure: "+row['id'])
                    content.extend([{"type":"text","text":f"Time {pos/fps:.3f}s of {n/fps:.3f}s"},{"type":"image","image":Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))}])
            finally:cap.release()
    content.append({"type":"text","text":prompt})
    return [{"role":"user","content":content}],keys,target

def train_impl(run=RUN, runtime_seconds=4800, curate=False):
    import hashlib,random,time
    from pathlib import Path
    import torch,cv2,peft,transformers,triton
    if curate and tuple(int(x) for x in triton.__version__.split(".")[:3])<(3,7,1):raise ValueError("Curated H100 training requires corrected Triton >=3.7.1")
    from PIL import Image
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    started=time.monotonic();deadline=started+runtime_seconds
    root=Path("/output")/run
    if (root/"report.json").exists():return json.loads((root/"report.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    def rows(split):
        path=root/f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["files"][split]:raise ValueError("Pilot data changed")
        return [json.loads(line) for line in path.open()]
    seen={}
    for split in ("train","dev","test"):
        for row in rows(split):
            identities=["id:"+row["id"]]
            if row.get("group_sha256"):identities.append("group:"+row["group_sha256"])
            if row.get("content_sha256"):identities.append("content:"+row["content_sha256"])
            media=row.get("media") or {}
            if media.get("path"):identities.append("media:"+media["path"])
            for asset in media.get("assets",[]):
                identities.append("media:"+asset["path"])
                if asset.get("sha256"):identities.append("mediahash:"+asset["sha256"])
            for identity in identities:
                if identity in seen and seen[identity]!=split:raise ValueError("Cross-split pilot overlap: "+identity)
                seen[identity]=split
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,attn_implementation="sdpa").cuda()
    model.config.use_cache=False
    targets=[name for name,module in model.named_modules() if "language_model" in name and ".self_attn." in name and name.rsplit('.',1)[-1] in {"q_proj","k_proj","v_proj","o_proj"} and isinstance(module,torch.nn.Linear)]
    if not targets:raise ValueError("No verified language attention LoRA targets")
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets,bias="none",task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False});model.enable_input_require_grads()
    letters=[processor.tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(26)]
    if any(len(ids)!=1 for ids in letters):raise ValueError("Answer code must be one tokenizer token")
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
    processor.tokenizer.padding_side="left"
    def encode_batch(batch):
        items=[conversation(r) for r in batch]
        inputs=processor.apply_chat_template([item[0] for item in items],tokenize=True,return_dict=True,return_tensors="pt",add_generation_prompt=True,enable_thinking=False,processor_kwargs={"padding":True,"images_kwargs":{"size":{"shortest_edge":4096,"longest_edge":(262144 if curate and all(r["modality"]=="image" for r in batch) else 65536)}}})
        if inputs["input_ids"].shape[-1]>(4096 if curate else 2048):raise ValueError("Prompt exceeds declared context budget: "+batch[0]['id'])
        return inputs.to("cuda"),[item[1] for item in items],[item[2] for item in items]
    def encode(row):
        inputs,keys,targets=encode_batch([row]);return inputs,keys[0],targets[0]
    optimizer.zero_grad(set_to_none=True);model.train()
    train_rows=rows("train");random.Random(3407).shuffle(train_rows)
    microbatch=2 if curate else 1
    if curate:
        groups={kind:[] for kind in ("text","image","video")}
        for r in train_rows:groups[r["modality"]].append(r)
        batches=[group[j:j+microbatch] for group in groups.values() for j in range(0,len(group),microbatch)]
        random.Random(3407).shuffle(batches)
    else:batches=[[r] for r in train_rows]
    processed=0;training_started=time.monotonic();warmup=None;recent_losses=[]
    progress_path=root/"progress.json"
    for batch_index,batch in enumerate(batches):
        if time.monotonic()>deadline:
            model.save_pretrained(root/"partial-adapter");processor.save_pretrained(root/"partial-adapter");output.commit();raise TimeoutError("Runtime budget reached; partial adapter saved")
        inputs,keys,targets_for_batch=encode_batch(batch)
        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        losses=[torch.nn.functional.cross_entropy(logits[j,torch.tensor([letters[k][0] for k in range(len(kk))],device="cuda")][None],torch.tensor([target],device="cuda")) for j,(kk,target) in enumerate(zip(keys,targets_for_batch))]
        loss=torch.stack(losses).mean()
        recent_losses.append(float(loss.detach()))
        if not torch.isfinite(loss):raise ValueError("Non-finite training loss")
        (loss*len(batch)/8).backward();processed+=len(batch)
        if processed%8==0 or processed==len(train_rows):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        if processed==160:warmup=(processed,time.monotonic())
        if processed%32==0:
            elapsed=time.monotonic()-started
            progress={"trained_rows":processed,"total_rows":len(train_rows),"optimizer_steps":processed//8,"microbatch":microbatch,"loss":float(loss.detach()),"mean_recent_loss":sum(recent_losses)/len(recent_losses),"elapsed_seconds":elapsed,"training_seconds":time.monotonic()-training_started}
            if warmup and processed>160:
                speed=(processed-warmup[0])/(time.monotonic()-warmup[1]);progress["examples_per_second_after_warmup"]=speed
                progress["projected_completion_seconds"]=elapsed+(len(train_rows)-processed)/speed
                # Conservative full-job bound: H100 + 4 CPU + 64 GiB memory.
                progress["estimated_compute_usd_so_far"]=elapsed*((.001097 if curate else .000694)+4*.0000131+64*.00000222)
                if curate and processed>=512 and progress["projected_completion_seconds"]*1.10+300>runtime_seconds:
                    model.save_pretrained(root/"partial-adapter");processor.save_pretrained(root/"partial-adapter")
                    (root/"budget-stop.json").write_text(json.dumps({"status":"budget_guard_stopped","progress":progress,"reason":"Measured throughput cannot finish with evaluation reserve within runtime allocation"},indent=2));output.commit();return json.loads((root/"budget-stop.json").read_text())
            progress_path.write_text(json.dumps(progress));output.commit();print(json.dumps(progress),flush=True);recent_losses=[]
        if processed%320==0 or (not curate and processed%160==0):model.save_pretrained(root/"partial-adapter");output.commit()
    model.save_pretrained(root/"adapter");processor.save_pretrained(root/"adapter");output.commit()
    model.eval()
    def evaluate(split):
        records=[]
        with torch.inference_mode():
            for row in rows(split):
                if time.monotonic()>deadline:raise TimeoutError("Budget reached before evaluation completed")
                inputs,keys,target=encode(row)
                logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1,torch.tensor([letters[j][0] for j in range(len(keys))],device="cuda")].float()
                record={"id":row["id"],"modality":row["modality"],"family":row["task_family"],"type":row["question"]["type"],"correct":int(logits.argmax())==target}
                if record["type"]=="score":record["expected_level_absolute_error"]=abs(float((logits.softmax(-1)*torch.arange(len(keys),device="cuda")).sum())-target)
                records.append(record)
        result={}
        for kind in ("text","image","video"):
            subset=[r for r in records if r["modality"]==kind];result[kind]={"n":len(subset),"correct":sum(r["correct"] for r in subset)}
        for family in sorted(set(r["family"] for r in records)):
            subset=[r for r in records if r["family"]==family];result["family:"+family]={"n":len(subset),"correct":sum(r["correct"] for r in subset)}
        scores=[r["expected_level_absolute_error"] for r in records if "expected_level_absolute_error" in r]
        if scores:result["score_expected_level_mae"]=sum(scores)/len(scores)
        (root/f"{split}-predictions.json").write_text(json.dumps(records));return result
    report={"run":run,"status":"candidate_completed_not_promoted","base":MODEL,"revision":REVISION,"rank":8,"microbatch":microbatch,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"target_modules":targets,"manifest":manifest,"dev":evaluate("dev"),"test":evaluate("test"),"elapsed_seconds":time.monotonic()-started,"torch_version":torch.__version__,"triton_version":triton.__version__,"peft_version":peft.__version__,"transformers_version":transformers.__version__,"ollama":"Ollama-shaped single-question schema; chat template and media extension; import not verified","benchmark_status":"derived pilot diagnostics only; no external rank claim"}
    (root/"report.json").write_text(json.dumps(report,indent=2));output.commit();model_cache.commit();return report


@app.function(image=image,volumes=mounts,gpu="A100-80GB",timeout=5400,cpu=(4,4),memory=(32768,65536),max_containers=1,scaledown_window=10)
def train():
    return train_impl()


@app.function(image=image,volumes={"/output":output},timeout=300)
def status():
    from pathlib import Path
    output.reload()
    root=Path("/output")/RUN
    return {name:json.loads((root/name).read_text()) for name in ("manifest.json","progress.json","report.json") if (root/name).exists()}

@app.local_entrypoint()
def main(mode:str="prepare"):
    print(json.dumps({"prepare":prepare,"train":train,"status":status}[mode].remote(),indent=2))
