"""Large, labeled text decision corpus and protected public diagnostics on Modal."""
import json
import modal

app = modal.App("vl-jev-text-decisions-20260930")
data = modal.Volume.from_name("vl-jev-text-decisions", create_if_missing=True)
cpu = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub==1.33.0", "pyarrow==21.0.0", "requests==2.32.5", "pyyaml==6.0.3"
).add_local_python_source("experiments")
SOURCES = ("AmazonScience/massive", "clinc/clinc_oos", "PolyAI/banking77",
           "nyu-mll/multi_nli", "Yelp/yelp_review_full", "tasksource/proofwriter",
           "fastino/fast-decisions", "SetFit/enron_spam", "nvidia/HelpSteer", "rajpurkar/squad_v2")
JEVBENCH = "bb05a335bc809e61b20c0f745d25499a82b326fc"
TEXT_RUN = "text-decisions-v2"
PINS = {
    "massive":("AmazonScience/massive","ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"),
    "clinc":("clinc/clinc_oos","155b9c710419136e17307b80d0a13e68cd46b4ec"),
    "mnli":("nyu-mll/multi_nli","da70db2af9d09693783c3320c4249840212ee221"),
    "squad":("rajpurkar/squad_v2","3ffb306f725f7d2ce8394bc1873b24868140c412"),
    "helpsteer":("nvidia/HelpSteer","3ca5d59c1bc1080af195b4254e7407db60b6f450"),
    "banking":("github:PolyAI-LDN/task-specific-datasets","57ec275d8078af65b7731c2a98be812d844a6d6b"),
}


def hash_text(value):
    import hashlib
    import re
    if not isinstance(value,str):value=json.dumps(value,ensure_ascii=False,sort_keys=True)
    return hashlib.sha256(re.sub(r"\s+"," ",value).strip().casefold().encode()).hexdigest()


@app.function(image=cpu, volumes={"/text-data":data,"/general-data":modal.Volume.from_name("vl-jev-general-v1-data")}, timeout=7200, memory=16384)
def build():
    import csv, gzip, hashlib, io, random, tarfile
    from pathlib import Path
    from collections import Counter, defaultdict
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import requests
    root=Path("/text-data")/TEXT_RUN;root.mkdir(parents=True,exist_ok=True)
    if (root/"summary.json").exists():
        return json.loads((root/"summary.json").read_text())
    downloads=[];raw=defaultdict(list);source_counts=Counter();skipped=Counter()
    def fetch(source,name):
        repo,revision=PINS[source]
        path=Path(hf_hub_download(repo,name,repo_type="dataset",revision=revision,cache_dir="/text-data/source-cache",token=False))
        downloads.append({"repo":repo,"revision":revision,"file":name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        return path
    def add(source,original,key,state,question,gold,group,**extra):
        source_counts[f"{source}/{original}"]+=1
        if not state.strip() or len(state)>6000:
            skipped[f"{source}:empty_or_more_than_6000_characters"]+=1;return
        split=("dev" if original in {"validation","dev"} else "test" if original.startswith("test") or original.startswith("validation_") else "train")
        # Sources without a separate test use validation as an untouched test,
        # and reserve group-held train rows for epoch selection/calibration.
        if source in {"squad","helpsteer"} and original=="validation":split="test"
        if split=="train":
            bucket=int(hash_text(source+str(group))[:8],16)%100
            if bucket>=95:split="calibration"
            elif bucket>=90:split="dev"
        raw[split].append({"id":f"{source}:{original}:{key}","split":split,"modality":"text","media":None,
                           "state":state,"question":question,"gold":{"key":gold},
                           "task_family":source+"_"+question["type"],"label_quality":"source_human_annotation",
                           "source":{"repo":PINS[source][0],"revision":PINS[source][1],"original_split":original,"group":str(group)},
                           "content_sha256":hash_text(state),"group_sha256":hash_text(source+str(group)),**extra})
    def intent(source,original,key,text,gold,vocabulary,group):
        # Derive difficult bounded candidates, including absence. This is not
        # the official full-label intent benchmark.
        rng=random.Random(hash_text(f"{source}:{original}:{key}"))
        base=[v for v in vocabulary if v not in {gold,"oos"}]
        words=set(gold.split("_"));base.sort(key=lambda v:(-len(words&set(v.split("_"))),hash_text(str(key)+v)))
        related=base[:min(4,len(base))];rest=[v for v in base if v not in related];rng.shuffle(rest)
        omit=gold=="oos" or rng.random()<0.2
        options=related+rest[:(3 if omit else 2)]
        if not omit: options.append(gold)
        rng.shuffle(options);options.append("none_of_these")
        criteria={v:v.replace("_"," ") for v in options}
        criteria["none_of_these"]="The message does not express any listed intent."
        add(source,original,key,text,{"type":"choice","instructions":"Which listed intent best describes this message? Choose none_of_these when none applies.","criteria":criteria},
            "none_of_these" if omit else gold,group,source_intent=gold,candidate_protocol="8 candidates including none; lexical-neighbor distractors; 20% withheld gold")
    # Full original BANKING77 labels/rows; the pilot used this same source pin.
    bank=[]
    for split in ("train","test"):
        url=f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{PINS['banking'][1]}/banking_data/{split}.csv"
        response=requests.get(url,timeout=120);response.raise_for_status()
        dest=root/f"banking-{split}.csv";dest.write_bytes(response.content)
        downloads.append({"repo":PINS["banking"][0],"revision":PINS["banking"][1],"file":split+".csv","sha256":hashlib.sha256(response.content).hexdigest()})
        bank.extend((split,i,r) for i,r in enumerate(csv.DictReader(io.StringIO(response.text))))
    bank_labels=sorted({r["category"] for _,_,r in bank})
    for split,i,r in bank:intent("banking",split,i,r["text"],r["category"],bank_labels,r["text"])
    meta=json.loads(Path("/text-data/probe/clinc__clinc_oos/metadata.json").read_text())
    info=next(i for i in meta["card"]["dataset_info"] if i["config_name"]=="plus")
    names=next(f for f in info["features"] if f["name"]=="intent")["dtype"]["class_label"]["names"]
    for split in ("train","validation","test"):
        for i,r in enumerate(pq.read_table(fetch("clinc",f"plus/{split}-00000-of-00001.parquet")).to_pylist()):
            intent("clinc",split,i,r["text"],names[str(r["intent"])],list(names.values()),r["text"])
    # MASSIVE is a parallel localization corpus: base IDs, not translations,
    # determine groups and internal splits.
    url="https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
    archive=Path("/text-data/source-cache/massive-1.1.tar.gz");archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        with requests.get(url,stream=True,timeout=180) as response:
            response.raise_for_status()
            with archive.open("wb") as dest:
                for chunk in response.iter_content(4*1024*1024):dest.write(chunk)
    downloads.append({"repo":PINS["massive"][0],"revision":PINS["massive"][1],"external_url":url,"sha256":hashlib.sha256(archive.read_bytes()).hexdigest()})
    with tarfile.open(archive,"r:gz") as tar:
        for member in tar.getmembers():
            locale=Path(member.name).stem
            if locale not in {"en-US","de-DE","zh-CN"} or not member.name.endswith(".jsonl"):continue
            rows=[json.loads(line) for line in tar.extractfile(member)]
            labels=sorted({r["intent"] for r in rows})
            for r in rows:
                intent("massive",r["partition"],f"{locale}:{r['id']}",r["utt"],r["intent"],labels,r["id"])
    # Alternate three-way inference with binary evidence-sufficiency gates.
    for split in ("train","validation_matched","validation_mismatched"):
        for r in pq.read_table(fetch("mnli",f"data/{split}-00000-of-00001.parquet")).to_pylist():
            if r["label"] not in (0,1,2):skipped["mnli:invalid_label"]+=1;continue
            kind="noul" if int(hash_text(r["pairID"])[:8],16)%2 else "choice"
            if kind=="noul":
                question={"type":"noul","instructions":"Do the supplied premise facts establish that the hypothesis is true? Unresolved claims count as false for this evidence-sufficiency gate.","criteria":{"false":"The hypothesis is contradicted or not established by the premise.","true":"The hypothesis follows from the supplied premise."}}
                gold="true" if r["label"]==0 else "false"
            else:
                question={"type":"choice","instructions":"Using only the premise, classify the hypothesis.","criteria":{"entailed":"The premise establishes the hypothesis.","unresolved":"The premise establishes neither the hypothesis nor its negation.","contradicted":"The premise establishes the negation of the hypothesis."}}
                gold=("entailed","unresolved","contradicted")[r["label"]]
            add("mnli",split,r["pairID"],f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}",question,gold,r["premise"])
    for split in ("train","validation"):
        for r in pq.read_table(fetch("squad",f"squad_v2/{split}-00000-of-00001.parquet")).to_pylist():
            add("squad",split,r["id"],f"Passage: {r['context']}\nQuestion: {r['question']}",
                {"type":"noul","instructions":"Can this question be answered from the supplied passage alone? Do not use facts from outside the passage.","criteria":{"false":"The passage does not provide an answer.","true":"The passage provides an answer."}},
                "true" if r["answers"]["text"] else "false",r["title"],source_context_sha256=hash_text(r["context"]))
    # Only text-observable style attributes; do not turn unsupported factual
    # correctness ratings into gold decisions.
    legends={"coherence":["No understandable meaning.","Mostly confusing or internally inconsistent.","Some unclear or inconsistent passages.","Mostly clear with minor problems.","Clear and internally consistent throughout."],
             "verbosity":["Very brief with minimal wording.","Concise with little elaboration.","Moderate detail and elaboration.","Extensive detail and elaboration.","Exceptionally wordy or exhaustive detail."]}
    for split in ("train","validation"):
        with gzip.open(fetch("helpsteer",split+".jsonl.gz"),"rt") as stream:
            for i,line in enumerate(stream):
                r=json.loads(line);attribute="coherence" if int(hash_text(r["response"])[:8],16)%2 else "verbosity"
                if not isinstance(r[attribute],int) or not 0<=r[attribute]<=4:raise ValueError("Unexpected human rating")
                add("helpsteer",split,i,f"Prompt: {r['prompt']}\nResponse: {r['response']}",
                    {"type":"score","instructions":f"Rate the response's {attribute} on the ordered 0–4 scale, using the supplied prompt and response.","criteria":legends[attribute]},str(r[attribute]),r["prompt"],score_attribute=attribute)
    # Protect all prior held-out text, public diagnostic text and source groups.
    protected=set(); prior_training=set()
    for line in Path("/general-data/v2/train.jsonl").open():
        r=json.loads(line)
        if r["modality"]=="text":prior_training.add(hash_text(r["state"]))
    for split in ("dev","calibration","test"):
        for line in (Path("/general-data/v2")/f"{split}.jsonl").open():
            r=json.loads(line)
            if r["modality"]=="text":protected.add(hash_text(r["state"]))
    for name in ("original","easy","hard"):
        path=Path("/text-data/public-jevbench")/f"{name}.jsonl"
        response=requests.get(f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEVBENCH}/datasets/public/{name}.jsonl",timeout=120);response.raise_for_status()
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
        protected.update(hash_text(json.loads(line)["state"]) for line in response.text.splitlines())
    diagnostic_meta=json.loads(Path("/text-data/probe/fastino__fast-decisions/metadata.json").read_text())
    for item in diagnostic_meta["files"]:
        if not item["path"].endswith(".jsonl"):continue
        path=Path(hf_hub_download("fastino/fast-decisions",item["path"],repo_type="dataset",revision=diagnostic_meta["revision"],cache_dir="/text-data/source-cache",token=False))
        protected.update(hash_text(json.loads(line)["input"]) for line in path.open())
    priority={"train":0,"dev":1,"calibration":2,"test":3};owner={}
    for split,rows in raw.items():
        for r in rows:
            for key in ("group_sha256","content_sha256"):
                owner[r[key]]=max(owner.get(r[key],-1),priority[split])
    limits={"banking":10000,"clinc":16000,"massive":35000,"mnli":30000,"squad":20000,"helpsteer":12000}
    clean={}; hashes={}
    for split in ("train","dev","calibration","test"):
        buckets=defaultdict(list)
        for r in raw[split]:
            if priority[split]<max(owner[r["group_sha256"]],owner[r["content_sha256"]]):
                skipped[f"{split}:protected_group_or_content"]+=1;continue
            if split=="train" and r["content_sha256"] in protected:
                skipped["train:prior_holdout_or_public_diagnostic"]+=1;continue
            if split!="train" and r["content_sha256"] in prior_training:
                skipped[f"{split}:already_trained_by_v4"]+=1;continue
            source=r["id"].split(":",1)[0]
            buckets[(source,r["question"]["type"],r["gold"]["key"],r.get("score_attribute",""))].append(r)
        # Round robin across label buckets prevents the large easy classes
        # from dominating the initial training subset.
        selected=defaultdict(list)
        for bucket,rows in sorted(buckets.items()):
            rows.sort(key=lambda r:hash_text(r["id"]));buckets[bucket]=rows
        active=list(sorted(buckets));indices=Counter()
        while active:
            pending=[]
            for bucket in active:
                source=bucket[0];cap=limits[source] if split=="train" else 1500 if split=="dev" else 1000
                index=indices[bucket]
                if len(selected[source])>=cap or index>=len(buckets[bucket]):continue
                selected[source].append(buckets[bucket][index]);indices[bucket]+=1;pending.append(bucket)
            active=pending
        rows=[r for source in sorted(selected) for r in selected[source]]
        seen=set();unique=[]
        for r in rows:
            key=hash_text(json.dumps({"state":r["state"],"question":r["question"]},sort_keys=True))
            if key in seen:skipped[f"{split}:duplicate_decision"]+=1;continue
            seen.add(key);unique.append(r)
        clean[split]=unique
    # Multi-message routing is deterministically derived from human intent
    # labels. The label is never copied into an individual message input.
    # Constituents stay in their already-protected split.
    for split,rows in clean.items():
        pools=defaultdict(lambda:defaultdict(list))
        for r in rows:
            label=r.get("source_intent")
            if label and label!="oos":pools[r["source"]["repo"]][label].append(r)
        generated=[];seen_queues=set();rng=random.Random(3407+priority[split])
        eligible=[source for source,labels in pools.items() if len(labels)>8]
        if not eligible:raise ValueError("Missing message-queue constituents")
        for i in range(3000 if split=="train" else 400):
            source=eligible[i%len(eligible)];labels=pools[source];wanted=sorted(labels)[rng.randrange(len(labels))]
            alternatives=[label for label in labels if label!=wanted]
            alternatives.sort(key=lambda label:(-len(set(wanted.split("_"))&set(label.split("_"))),hash_text(str(i)+label)))
            other=rng.sample(alternatives[:min(10,len(alternatives))],3)
            constituents=[rng.choice(labels[label]) for label in other]
            absent=rng.random()<.2
            if not absent:
                matching=labels[wanted]
                constituents+=rng.sample(matching,min(len(matching),2 if rng.random()<.3 else 1))
            rng.shuffle(constituents)
            identity=hash_text(json.dumps({"ids":[r["id"] for r in constituents],"intent":wanted}))
            if identity in seen_queues:continue
            seen_queues.add(identity)
            ages=rng.sample(range(1,241),len(constituents))
            messages=[{"id":f"message_{j+1}","age_minutes":ages[j],"text":r["state"]} for j,r in enumerate(constituents)]
            matching=[j for j,r in enumerate(constituents) if r["source_intent"]==wanted]
            kind="noul" if i%3==0 else "choice"
            if kind=="noul":
                question={"type":"noul","instructions":f"Does any supplied message request {wanted.replace('_',' ')}?",
                          "criteria":{"false":"No supplied message expresses the requested intent.","true":"At least one supplied message expresses the requested intent."}}
                gold="true" if matching else "false"
            else:
                criteria={m["id"]:f"Select {m['id']}" for m in messages};criteria["none"]="None of the supplied messages expresses the requested intent."
                question={"type":"choice","instructions":f"Select the oldest message requesting {wanted.replace('_',' ')}. A greater age_minutes value means an older message. Choose none when no message matches.","criteria":criteria}
                gold=messages[max(matching,key=lambda j:ages[j])]["id"] if matching else "none"
            state=json.dumps({"messages":messages},ensure_ascii=False,sort_keys=True)
            generated.append({"id":"queue:"+split+":"+str(i),"split":split,"modality":"text","media":None,"state":state,"question":question,"gold":{"key":gold},
                              "task_family":"message_queue_"+kind,"label_quality":"deterministic_composition_of_source_human_labels_and_visible_ages",
                              "source":{"repo":"derived:message-queue","revision":"vl-jev-queue-v1","original_split":split,"group":identity},
                              "content_sha256":hash_text(state),"group_sha256":identity,"constituent_ids":[r["id"] for r in constituents]})
        rows.extend(generated)
    for split,rows in clean.items():
        path=root/f"{split}.jsonl";path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows));hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary={"run":TEXT_RUN,"source_pins":PINS,"files":downloads,"source_rows_inspected":dict(source_counts),
             "splits":{s:{"n":len(rows),"sources":dict(Counter(r["source"]["repo"] for r in rows)),"types":dict(Counter(r["question"]["type"] for r in rows))} for s,rows in clean.items()},
             "split_sha256":hashes,"skipped":dict(skipped),"max_state_characters":6000,
             "held_out_public_benchmarks_used_for_training":False,"unadmitted":{"ProofWriter":"Source-specific data license unconfirmed.","Enron spam":"Mirror lacks data license; not a verified email-routing gold corpus.","Yelp":"Restricted academic-use terms; prefer permissive human ratings for Score."},
             "benchmark_limits":"Intent decisions use 8 derived candidates, not official full-label intent accuracy. MNLI 3-way is source-label preserving; SQuAD answerability excludes extractive EM/F1. Held-out subsets are diagnostics, not full-source leaderboard scores.",
             "multilingual_limits":"MASSIVE en-US, de-DE, zh-CN are translations of the same underlying IDs, not independent million-example English decisions."}
    (root/"summary.json").write_text(json.dumps(summary,indent=2));data.commit();return summary


def typed_options(question, labels=None):
    """Identical key-prefixed options to the public Choice/Noul/Score adapter."""
    from experiments.typed_decisions import _text
    kind = question["type"]
    if kind == "noul":
        keys = ["false", "true"]
        criteria=question.get("criteria") or {}
        return keys, [f"{k}: {_text(criteria.get(k, 'No' if k=='false' else 'Yes'))}" for k in keys]
    if kind == "choice":
        keys = labels or list(question["criteria"])
        return keys, [f"{k}: {_text(question['criteria'][k])}" for k in keys]
    keys = [str(i) for i in range(len(question["criteria"]))]
    return keys, [f"{i}: {_text(q)}" for i,q in enumerate(question["criteria"])]


from experiments.modal_jev_omni_sports_train import base_image, model_cache, general_training
audit = modal.Volume.from_name("vl-jev-benchmark-audit")
mounts = {"/text-data":data, "/audit":audit, "/model-cache":model_cache,
          "/general-runs":general_training}


def text_rows(split):
    import hashlib
    from pathlib import Path
    root=Path("/text-data")/TEXT_RUN
    summary=json.loads((root/"summary.json").read_text());path=root/f"{split}.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=summary["split_sha256"][split]:
        raise ValueError("Text manifest changed")
    return [json.loads(line) for line in path.open()]


@app.function(image=base_image, volumes=mounts, gpu="H100", timeout=14400, memory=32768, max_containers=8)
def extract(shard: int=0, shards: int=8):
    import hashlib
    from pathlib import Path
    from experiments.modal_jev_omni_sports_train import _load_classifier, _feature_content, MODEL_REVISION
    torch,classifier,package=_load_classifier()
    completed=0
    for split in ("train","dev","calibration","test"):
        rows=[r for i,r in enumerate(text_rows(split)) if i%shards==shard]
        root=Path("/text-data")/TEXT_RUN/"features"/split/f"shard{shard:02d}of{shards:02d}"
        root.mkdir(parents=True,exist_ok=True)
        for offset in range(0,len(rows),64):
            batch=rows[offset:offset+64];dest=root/f"pack-{offset//64:05d}.pt"
            lock=hashlib.sha256(json.dumps(batch,sort_keys=True).encode()).hexdigest()
            if dest.exists():
                cached=torch.load(dest,map_location="cpu",weights_only=True)
                if cached["manifest_sha256"]!=lock or cached["model_revision"]!=MODEL_REVISION:
                    raise ValueError("Cached text features differ")
                completed+=len(batch);continue
            records=[]
            for row in batch:
                labels,options=typed_options(row["question"])
                feature=_feature_content(torch,classifier,package,[],row["state"],row["question"]["instructions"],options)
                if not bool(torch.isfinite(feature).all()):raise ValueError("Nonfinite text feature")
                records.append({"id":row["id"],"feature":feature,"count":len(labels),"target":labels.index(row["gold"]["key"]),
                                "modality":"text","family":row["task_family"],"source_repo":row["source"]["repo"],
                                "type":row["question"]["type"],"group":row["group_sha256"],"score_attribute":row.get("score_attribute")})
            torch.save({"manifest_sha256":lock,"model_revision":MODEL_REVISION,"rows":records},dest)
            completed+=len(records)
            if (offset//64)%10==0:data.commit();print(f"text shard {shard}: {split} {offset+len(batch)}/{len(rows)}",flush=True)
    data.commit();model_cache.commit();return {"shard":shard,"shards":shards,"rows":completed}


def load_features(torch, split):
    import hashlib
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from experiments.modal_jev_omni_sports_train import MODEL_REVISION
    rows=text_rows(split)
    root=Path("/text-data")/TEXT_RUN/"features"/split
    jobs=[]
    for shard in range(8):
        subset=[(i,r) for i,r in enumerate(rows) if i%8==shard]
        for offset in range(0,len(subset),64):
            jobs.append((root/f"shard{shard:02d}of08"/f"pack-{offset//64:05d}.pt",subset[offset:offset+64]))
    if set(root.glob("shard*of08/pack-*.pt"))!={path for path,_ in jobs}:
        raise ValueError("Text feature pack coverage mismatch")
    def read(job):
        path,batch=job;pack=torch.load(path,map_location="cpu",weights_only=True)
        originals=[r for _,r in batch]
        lock=hashlib.sha256(json.dumps(originals,sort_keys=True).encode()).hexdigest()
        if pack["manifest_sha256"]!=lock or pack["model_revision"]!=MODEL_REVISION:
            raise ValueError(f"Text feature manifest or model mismatch: {path}")
        if len(pack["rows"])!=len(batch):raise ValueError("Text feature pack row count mismatch")
        output=[]
        for (index,original),record in zip(batch,pack["rows"]):
            keys,_=typed_options(original["question"])
            if record["id"]!=original["id"] or record["target"]!=keys.index(original["gold"]["key"]) or record["count"]!=len(keys):
                raise ValueError("Text feature identity or target mismatch")
            # MultiNLI pair IDs can repeat for distinct annotated inputs. The
            # authenticated pack manifest and position identify each feature.
            output.append((index,record))
        return output
    with ThreadPoolExecutor(max_workers=8) as pool:
        found=[item for batch in pool.map(read,jobs) for item in batch]
    if sorted(i for i,_ in found)!=list(range(len(rows))):raise ValueError("Missing text feature positions")
    return [r for _,r in sorted(found,key=lambda item:item[0])]


def text_metrics(torch,head,rows):
    from collections import defaultdict
    metrics=defaultdict(lambda:{"n":0,"correct":0,"loss":0.,"score_absolute_error":0.,"score_n":0})
    head.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]
            logits=head(torch.stack([r["feature"] for r in batch]).cuda().float(),torch.tensor([r["count"] for r in batch],device="cuda"))
            for i,r in enumerate(batch):
                p=logits[i,:r["count"]].softmax(-1)
                for key in ("overall","family:"+r["family"],"type:"+r["type"]):
                    m=metrics[key];m["n"]+=1;m["correct"]+=int(p.argmax().item()==r["target"]);m["loss"]+=-p[r["target"]].clamp_min(1e-9).log().item()
                    if r["type"]=="score":
                        ev=(p*torch.arange(r["count"],device="cuda")).sum().item();m["score_absolute_error"]+=abs(ev-r["target"]);m["score_n"]+=1
    out={k:{"n":m["n"],"correct":m["correct"],"accuracy":m["correct"]/m["n"],"log_loss":m["loss"]/m["n"],
            "ordinal_expected_value_MAE":m["score_absolute_error"]/m["score_n"] if m["score_n"] else None} for k,m in metrics.items()}
    families=[v["accuracy"] for k,v in out.items() if k.startswith("family:")]
    out["family_macro_accuracy"]=sum(families)/len(families)
    return out


@app.function(image=base_image,volumes={**mounts,"/general-data":modal.Volume.from_name("vl-jev-general-v1-data"),
              "/gui-data":modal.Volume.from_name("vl-jev-gui-general-data")},gpu="L4",timeout=3600,
              cpu=(2,2),memory=(16384,32768),max_containers=1,scaledown_window=10)
def train():
    import copy, hashlib, random
    from pathlib import Path
    import sys
    import torch
    from huggingface_hub import hf_hub_download
    from experiments.modal_jev_omni_sports_train import MODEL_ID, MODEL_REVISION, _load_general_features, _general_metrics
    from experiments.modal_general_v3_gui import _gui_rows, _load_gui_features
    from experiments.modal_soccer_vqa_posttrain import _load_soccer_features, DIRECT_VISUAL_FAMILIES
    root=Path("/general-runs/general-head-v5-text-decisions")
    if (root/"report.json").exists():return json.loads((root/"report.json").read_text())
    source=hf_hub_download(MODEL_ID,"jev_omni.py",revision=MODEL_REVISION)
    sys.path.insert(0,str(Path(source).parent))
    import jev_omni as package
    print("Loading verified packed text features; no backbone weights needed",flush=True)
    text_train=load_features(torch,"train");text_dev=load_features(torch,"dev")
    print(f"Text features loaded: {len(text_train)} train, {len(text_dev)} dev",flush=True)
    old_root=Path("/general-runs/general-head-v2")
    replay=_load_general_features(torch,old_root,"train","v2")+_load_gui_features(torch,"train")
    replay += [r for r in _load_soccer_features(torch,"train",32,scope="all") if r["family"] in DIRECT_VISUAL_FAMILIES]
    general_dev=_load_general_features(torch,old_root,"dev","v2")
    shared={r["media"]["sha256"] for r in _gui_rows("train") if r.get("media",{}).get("sha256")}
    excluded={r["id"] for r in _gui_rows("dev") if r.get("media",{}).get("sha256") in shared}
    gui_dev=[r for r in _load_gui_features(torch,"dev") if r["id"] not in excluded]
    soccer_dev=[r for r in _load_soccer_features(torch,"valid",8,scope="all") if r["family"] in DIRECT_VISUAL_FAMILIES]
    if len(replay)!=59000:raise ValueError("Unified v4 full replay changed")
    parent=json.loads(Path("/general-runs/general-head-v4-unified-audited/report.json").read_text())
    path=Path(parent["checkpoint"])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=parent["checkpoint_sha256"]:raise ValueError("V4 parent changed")
    head=package._Head256(3840).cuda();head.load_state_dict(torch.load(path,map_location="cuda",weights_only=True))
    def measure():
        return {"text":text_metrics(torch,head,text_dev),"general":_general_metrics(torch,head,general_dev),
                "gui":_general_metrics(torch,head,gui_dev),"soccer":_general_metrics(torch,head,soccer_dev)}
    print(f"Replay features loaded: {len(replay)}; measuring baseline",flush=True)
    baseline=measure();best=baseline;best_state=copy.deepcopy(head.state_dict());selected=0;history=[]
    rows=replay+text_train;optimizer=torch.optim.AdamW(head.parameters(),lr=2e-5,weight_decay=.01)
    for epoch in range(3):
        head.train();order=list(range(len(rows)));random.Random(3417+epoch).shuffle(order)
        for start in range(0,len(order),128):
            batch=[rows[i] for i in order[start:start+128]]
            logits=head(torch.stack([r["feature"] for r in batch]).cuda().float(),torch.tensor([r["count"] for r in batch],device="cuda"))
            target=torch.tensor([r["target"] for r in batch],device="cuda")
            losses=torch.nn.functional.cross_entropy(logits,target,reduction="none")
            # Score learns ordinal proximity while preserving a distribution.
            ordinal=[]
            for i,r in enumerate(batch):
                if r.get("type")=="score":
                    p=logits[i,:r["count"]].softmax(-1);cdf=p.cumsum(0)
                    truth=(torch.arange(r["count"],device="cuda")>=r["target"]).float()
                    ordinal.append(((cdf-truth)**2).mean())
            loss=losses.mean()+(.2*torch.stack(ordinal).sum()/len(batch) if ordinal else 0.)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        measured=measure();gates={
            "general_text":measured["general"]["text"]["accuracy"]>=baseline["general"]["text"]["accuracy"]-.005,
            "general_vision":measured["general"]["vision"]["accuracy"]>=baseline["general"]["vision"]["accuracy"]-.005,
            "gui":measured["gui"]["overall"]["accuracy"]>=baseline["gui"]["overall"]["accuracy"]-.01,
            "soccer":measured["soccer"]["overall"]["accuracy"]>=baseline["soccer"]["overall"]["accuracy"]-.01,
            "score_MAE":measured["text"]["type:score"]["ordinal_expected_value_MAE"]<=baseline["text"]["type:score"]["ordinal_expected_value_MAE"]+.02}
        if all(gates.values()) and measured["text"]["family_macro_accuracy"]>best["text"]["family_macro_accuracy"]:
            best=measured;best_state=copy.deepcopy(head.state_dict());selected=epoch+1
        history.append({"epoch":epoch+1,"gates":gates,"metrics":measured})
        print(json.dumps({"epoch":epoch+1,"gates":gates,"text_macro":measured["text"]["family_macro_accuracy"]}),flush=True)
    root.mkdir(parents=True,exist_ok=True);head.load_state_dict(best_state);torch.save(best_state,root/"head.pt")
    # Read new text test only after development selection. Public benchmark
    # scores never affect selection or fallback.
    test_rows=load_features(torch,"test");selected_test=text_metrics(torch,head,test_rows)
    head.load_state_dict(torch.load(path,map_location="cuda",weights_only=True));baseline_test=text_metrics(torch,head,test_rows)
    report={"run":root.name,"status":"selected_text_candidate" if selected else "fallback_to_v4","selected_epoch":selected,
            "new_text_train_rows":len(text_train),"full_v4_replay_rows":len(replay),"train_rows":len(rows),
            "parent_sha256":parent["checkpoint_sha256"],"checkpoint":str(root/"head.pt"),"checkpoint_sha256":hashlib.sha256((root/"head.pt").read_bytes()).hexdigest(),
            "baseline_dev":baseline,"selected_dev":best,"history":history,"baseline_text_test":baseline_test,"selected_text_test":selected_test,
            "selection":"text family macro; general text/vision <=0.5pp regression; GUI/Soccer <=1pp; ordinal MAE <=+0.02",
            "test_used_for_selection":False,"external_benchmark_status":"not yet evaluated; candidate does not inherit v4 benchmark results"}
    (root/"report.json").write_text(json.dumps(report,indent=2));general_training.commit();return report


@app.function(image=cpu,volumes={"/text-data":data},timeout=86400)
def pipeline():
    import time
    from pathlib import Path
    deadline=time.monotonic()+7200
    while not (Path("/text-data")/TEXT_RUN/"summary.json").exists():
        if time.monotonic()>deadline:raise TimeoutError("Text build did not complete")
        time.sleep(30);data.reload()
    jobs=[extract.spawn(i,8) for i in range(8)]
    for job in jobs:job.get()
    return train.remote()


@app.function(image=base_image, volumes=mounts, gpu="H100", timeout=7200, memory=32768)
def jevbench():
    import hashlib
    import importlib.util
    import time
    from pathlib import Path
    from types import SimpleNamespace
    from collections import defaultdict
    from huggingface_hub import hf_hub_download
    from experiments.modal_jev_omni_sports_train import _load_classifier, _feature_content
    from experiments.modal_verify_v3_public import REPO, REVISION, SHA256
    torch, classifier, package = _load_classifier()
    source = Path("/audit/jevbench") / JEVBENCH
    spec = importlib.util.spec_from_file_location("frozen_jevbench_scoring", source/"jevbench__scoring.py")
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    parent = Path(hf_hub_download(REPO,"decision_head.pt",revision=REVISION,token=False))
    if hashlib.sha256(parent.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Public parent changed")
    report = json.loads(Path("/general-runs/general-head-v4-unified-audited/report.json").read_text())
    heads = {"upstream":classifier.head}
    for name,path in (("v3",parent),("v4",Path(report["checkpoint"]))):
        head = package._Head256(3840).cuda().eval()
        head.load_state_dict(torch.load(path,map_location="cuda",weights_only=True));heads[name]=head
    records=[]
    for split,expected_count in (("original",72),("easy",48),("hard",111)):
        rows=[json.loads(line) for line in (source/f"datasets__public__{split}.jsonl").open()]
        if len(rows)!=expected_count:
            raise ValueError("Frozen public task count changed")
        for row in rows:
            keys,options=typed_options(row["question"],row["labels"] if row["question"]["type"]=="choice" else None)
            start=time.perf_counter()
            state=row["state"] if isinstance(row["state"],str) else json.dumps(row["state"],ensure_ascii=False,sort_keys=True)
            instructions=row["question"]["instructions"]
            if not isinstance(instructions,str):instructions=json.dumps(instructions,ensure_ascii=False,sort_keys=True)
            feature=_feature_content(torch,classifier,package,[],state,instructions,options).cuda()
            predictions={}
            for name,head in heads.items():
                with torch.inference_mode():
                    probs=head(feature.unsqueeze(0),torch.tensor([len(keys)],device="cuda"))[0,:len(keys)].softmax(-1).tolist()
                labels=["no","yes"] if row["question"]["type"]=="noul" else keys
                if set(labels)!=set(row["labels"]):
                    raise ValueError("Task/API labels differ")
                predictions[name]=scorer.score_task(dict(zip(labels,probs)),SimpleNamespace(**row))
            records.append({"id":row["id"],"group":row.get("group") or row["id"],"split":split,
                            "type":row["question"]["type"],"expected":row["expected"],
                            "forward_and_all_heads_ms":1000*(time.perf_counter()-start),"predictions":predictions})
    summaries={}
    for name in heads:
        groups=defaultdict(list);types=defaultdict(list);tiers=defaultdict(list)
        for row in records:
            value=int(row["predictions"][name]["correct"] is True)
            groups[row["group"]].append(value);types[row["type"]].append(value);tiers[row["split"]].append(value)
        summaries[name]={"n":len(records),"correct":sum(sum(v) for v in groups.values()),
                         "micro_accuracy":sum(sum(v) for v in groups.values())/len(records),
                         "groups":len(groups),"group_macro_accuracy":sum(sum(v)/len(v) for v in groups.values())/len(groups),
                         "by_type":{k:{"n":len(v),"correct":sum(v),"accuracy":sum(v)/len(v)} for k,v in types.items()},
                         "by_tier":{k:{"n":len(v),"correct":sum(v),"accuracy":sum(v)/len(v)} for k,v in tiers.items()},
                         "strict_valid":sum(r["predictions"][name]["strict_valid"] for r in records)}
    output={"source_revision":JEVBENCH,"source_scoring":"unmodified jevbench/scoring.py score_task",
            "protocol":"public 231 tasks; original label order; native key-prefixed typed prompts; no training",
            "not_evaluated":"current sealed composite or hidden tasks","head_sha256":report["checkpoint_sha256"],
            "models":summaries}
    root=Path("/text-data/jevbench")/JEVBENCH;root.mkdir(parents=True,exist_ok=True)
    (root/"predictions.json").write_text(json.dumps(records,indent=2))
    (root/"report.json").write_text(json.dumps(output,indent=2));data.commit();model_cache.commit()
    return output


@app.function(image=cpu, volumes={"/text-data": data}, timeout=3600, memory=8192)
def probe(additional_only: bool=False):
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download
    import requests
    root = Path("/text-data/probe"); root.mkdir(parents=True, exist_ok=True)
    out = {}
    for repo in SOURCES[-2:] if additional_only else SOURCES:
        info = HfApi(token=False).dataset_info(repo, files_metadata=True)
        card = info.card_data.to_dict() if info.card_data else {}
        files = [{"path": f.rfilename, "bytes": f.size} for f in info.siblings]
        key = repo.replace("/", "__")
        Path(root/key).mkdir(exist_ok=True)
        metadata = {"repo": repo, "revision": info.sha, "card": card, "files": files}
        (root/key/"metadata.json").write_text(json.dumps(metadata, indent=2))
        configs = card.get("configs", [])
        config = ("en-US" if repo.endswith("massive") else "plus" if repo.endswith("clinc_oos") else
                  configs[0].get("config_name", "default") if configs else "default")
        response = requests.get("https://datasets-server.huggingface.co/first-rows",
                                params={"dataset":repo, "config":config, "split":"train"}, timeout=120)
        sample = response.json()
        # Full schemas and sample stay on Modal; return only concise inspection.
        (root/key/"sample.json").write_text(json.dumps(sample, indent=2))
        first = sample.get("rows", [{}])[0].get("row", {})
        out[repo] = {"revision":info.sha, "license":card.get("license"),
                     "dataset_info":card.get("dataset_info"), "config":config,
                     "data_files":files[:30], "first_row":first,
                     "features":sample.get("features"), "sample_status":response.status_code}
    data.commit()
    return out


@app.local_entrypoint()
def main(mode: str="probe"):
    operations={"build":build,"jevbench":jevbench,"inspect":inspect,"pipeline":pipeline,"train":train,"probe":probe,"status":status}
    if mode=="probe-new":answer=probe.remote(True)
    elif mode in operations:answer=operations[mode].remote()
    else:raise ValueError("Unknown text decision mode")
    if mode in {"pipeline","train"}:
        answer={k:answer[k] for k in ("run","status","selected_epoch","new_text_train_rows","full_v4_replay_rows","train_rows","checkpoint_sha256","baseline_text_test","selected_text_test")}
    print(json.dumps(answer,indent=2))


@app.function(image=cpu,volumes={"/text-data":data,"/general-runs":general_training},timeout=300)
def status():
    from pathlib import Path
    root=Path("/text-data")/TEXT_RUN;out={}
    if (root/"summary.json").exists():
        r=json.loads((root/"summary.json").read_text())
        out["dataset"]={k:r[k] for k in ("run","splits","split_sha256","skipped")}
        out["features_by_split"]={s:len(list((root/"features"/s).glob("shard*of08/pack-*.pt"))) for s in ("train","dev","calibration","test")}
    typed=Path("/text-data/jevbench")/JEVBENCH/"report.json"
    if typed.exists():out["jevbench"]=json.loads(typed.read_text())
    candidate=Path("/general-runs/general-head-v5-text-decisions/report.json")
    if candidate.exists():
        r=json.loads(candidate.read_text());out["candidate"]={k:r[k] for k in ("run","status","selected_epoch","train_rows","checkpoint_sha256","baseline_text_test","selected_text_test")}
    return out


@app.function(image=cpu, volumes={"/text-data": data}, timeout=600)
def inspect():
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    out={}
    for folder in Path("/text-data/probe").iterdir():
        meta=json.loads((folder/"metadata.json").read_text())
        sample=json.loads((folder/"sample.json").read_text())
        infos=meta["card"].get("dataset_info") or []
        if isinstance(infos,dict): infos=[infos]
        out[meta["repo"]]={"revision":meta["revision"],"license":meta["card"].get("license"),
                          "configs":[{"name":i.get("config_name"),"splits":i.get("splits")} for i in infos],
                          "files":[f["path"] for f in meta["files"] if f["path"].endswith((".parquet",".jsonl",".py","LICENSE"))][:25],
                          "first_row_keys":list(sample.get("rows",[{}])[0].get("row",{}))}
        if meta["repo"]=="AmazonScience/massive":
            source=Path(hf_hub_download(meta["repo"],"massive.py",repo_type="dataset",revision=meta["revision"],cache_dir="/text-data/source-cache")).read_text()
            out[meta["repo"]]["loader_url_lines"]=[line for line in source.splitlines() if any(k in line for k in ("https://amazon","partition","intent =","utt =","jsonl","TRAIN"))][:30]
    data.commit(); return out
