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


@app.function(image=image,volumes=mounts,gpu="A100-80GB",timeout=5400,cpu=(4,4),memory=(32768,65536),max_containers=1,scaledown_window=10)
def train():
    import hashlib,random,time
    from pathlib import Path
    import torch,cv2,peft,transformers
    from PIL import Image
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    started=time.monotonic();deadline=started+4800
    root=Path("/output")/RUN
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
            if row.get("media",{}):identities.append("media:"+row["media"]["path"])
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
    def encode(row):
        prompt,keys,target=compiled(row);content=[];media=row.get("media")
        if row["modality"]=="image":
            content.append({"type":"image","image":Image.open(media["path"]).convert("RGB")})
        elif row["modality"]=="video":
            # Genuine ordered frames, actual timestamps and duration; decode only
            # a bounded number of samples, preserving the temporal input.
            cap=cv2.VideoCapture(media["path"]);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));frames=[]
            if n<1 or fps<=0:raise ValueError("Invalid video metadata")
            for i in range(4):
                pos=min(n-1,int((i+.5)*n/4));cap.set(cv2.CAP_PROP_POS_FRAMES,pos);ok,frame=cap.read()
                if not ok:raise ValueError("Video decode failure")
                content.extend([{"type":"text","text":f"Time {pos/fps:.3f}s of {n/fps:.3f}s"},{"type":"image","image":Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))}])
            cap.release()
        content.append({"type":"text","text":prompt})
        inputs=processor.apply_chat_template([{"role":"user","content":content}],tokenize=True,return_dict=True,return_tensors="pt",add_generation_prompt=True,enable_thinking=False,images_kwargs={"size":{"shortest_edge":4096,"longest_edge":65536}})
        if inputs["input_ids"].shape[-1]>2048:raise ValueError(f"Pilot prompt too long: {row['id']}")
        return inputs.to("cuda"),keys,target
    history=[];optimizer.zero_grad(set_to_none=True);model.train()
    train_rows=rows("train");random.Random(3407).shuffle(train_rows)
    progress_path=root/"progress.json"
    for i,row in enumerate(train_rows):
        if time.monotonic()>deadline:
            model.save_pretrained(root/"partial-adapter");processor.save_pretrained(root/"partial-adapter");output.commit();raise TimeoutError("Pilot runtime budget reached; partial adapter saved")
        inputs,keys,target=encode(row)
        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1,torch.tensor([letters[j][0] for j in range(len(keys))],device="cuda")].float()
        loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([target],device="cuda"))
        if not torch.isfinite(loss):raise ValueError("Non-finite training loss")
        (loss/8).backward()
        if (i+1)%8==0 or i+1==len(train_rows):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        if (i+1)%32==0:
            progress={"trained_rows":i+1,"total_rows":len(train_rows),"loss":float(loss.detach()),"elapsed_seconds":time.monotonic()-started}
            progress_path.write_text(json.dumps(progress));output.commit();print(json.dumps(progress),flush=True)
        if (i+1)%160==0:model.save_pretrained(root/"partial-adapter");output.commit()
    model.save_pretrained(root/"adapter");processor.save_pretrained(root/"adapter");output.commit()
    model.eval()
    def evaluate(split):
        records=[]
        with torch.inference_mode():
            for row in rows(split):
                if time.monotonic()>deadline:raise TimeoutError("Budget reached before evaluation completed")
                inputs,keys,target=encode(row)
                logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1,torch.tensor([letters[j][0] for j in range(len(keys))],device="cuda")].float()
                records.append({"id":row["id"],"modality":row["modality"],"correct":int(logits.argmax())==target})
        result={}
        for kind in ("text","image","video"):
            subset=[r for r in records if r["modality"]==kind];result[kind]={"n":len(subset),"correct":sum(r["correct"] for r in subset)}
        (root/f"{split}-predictions.json").write_text(json.dumps(records));return result
    report={"run":RUN,"status":"pilot_completed_not_promoted","base":MODEL,"revision":REVISION,"rank":8,"target_modules":targets,"manifest":manifest,"dev":evaluate("dev"),"test":evaluate("test"),"elapsed_seconds":time.monotonic()-started,"peft_version":peft.__version__,"transformers_version":transformers.__version__,"ollama":"exact single-question schema; media extension; import not verified","benchmark_status":"derived pilot diagnostics only; no external rank claim"}
    (root/"report.json").write_text(json.dumps(report,indent=2));output.commit();model_cache.commit();return report


@app.function(image=image,volumes={"/output":output},timeout=300)
def status():
    from pathlib import Path
    root=Path("/output")/RUN
    return {name:json.loads((root/name).read_text()) for name in ("manifest.json","progress.json","report.json") if (root/name).exists()}

@app.local_entrypoint()
def main(mode:str="prepare"):
    print(json.dumps({"prepare":prepare,"train":train,"status":status}[mode].remote(),indent=2))
