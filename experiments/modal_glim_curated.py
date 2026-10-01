"""Curated Qwen decision training; data and compute remain on Modal."""
import json, modal
from experiments import modal_glim_qwen_lora as pilot
from experiments.modal_general_v3_gui import gui_data,gui_media,_gui_rows
from experiments.modal_general_v3_gui_test import test_data,test_media
app=modal.App('glim-qwen9b-curated')
RUN='qwen9b-curated-20k-v3'
mounts={**pilot.mounts,'/gui-data':gui_data,'/gui-media':gui_media,'/gui-test-data':test_data,'/gui-test-media':test_media}
image=pilot.image.add_local_python_source('experiments')


def inventory_rows():
    from experiments.modal_text_decisions import text_rows
    from experiments.modal_jev_omni_sports_train import _general_rows
    from experiments.modal_soccer_vqa_posttrain import _rows,DIRECT_VISUAL_FAMILIES
    groups={}
    for split in ('train','dev','test'):
        rows=text_rows(split)+_general_rows(split,'v2')
        if split!='test':rows+=_gui_rows(split)
        else:
            from pathlib import Path
            rows += [json.loads(s) for s in Path('/gui-test-data/mind2web_test_task_v1/test_task.jsonl').open()]
        if split!='test':soccer,_=_rows('train' if split=='train' else 'valid',32 if split=='train' else 8,scope='all')
        else:soccer,_=_rows('test',8,scope='pilot')
        rows += [r for r in soccer if r['task_family'] in DIRECT_VISUAL_FAMILIES]
        groups[split]=rows
    return groups

@app.function(image=image,volumes=mounts,timeout=600,memory=8192,cpu=(2,2))
def audit():
    from collections import Counter
    from pathlib import Path
    from transformers import AutoProcessor
    import inspect
    result={}
    for split,rows in inventory_rows().items():
        result[split]={'families':dict(Counter((r['modality']+' '+r['task_family']) for r in rows)), 'sources':dict(Counter(r['source']['repo'] for r in rows))}
    result['soccer_samples']=[{'family':r['task_family'],'modality':r['modality'],'question':r['question']['instructions'][:150],'media':r.get('media')} for r in inventory_rows()['train'] if r['id'].startswith('soccer-vqa-')][:3]
    proc=AutoProcessor.from_pretrained(pilot.MODEL,revision=pilot.REVISION)
    src=inspect.getsource(proc.apply_chat_template)
    result['processor_kwargs_lines']=[line.strip() for line in src.splitlines() if 'kwargs' in line][-22:]
    result['pilot_status']={p.name:json.loads(p.read_text()) for p in (Path('/output')/pilot.RUN).glob('*.json') if p.name in ('report.json','progress.json')}
    return result


# Source weights follow measured weaknesses; no per-test-example mining.
QUOTAS={'routing':2000,'evidence':1700,'score':3000,'queues':1700,
        'gui':3000,'grounded_images':1500,'action_count':900,'soccer_image':1200,
        'soccer_video':2200,'ucf_video':2800}


def assets(row):
    media=row.get('media') or {}
    if media.get('kind')=='modal_media_sequence':return media['assets']
    if media.get('kind') in ('modal_image','modal_video'):
        return [{'kind':'video' if media['kind']=='modal_video' else 'image','path':media['path'],'sha256':media.get('sha256')}]
    # Older pilot records retain external-source metadata; use only the
    # already materialized Modal images, never fetch during curation/training.
    if row.get('modality') in ('image','image_sequence'):
        from pathlib import Path
        import hashlib
        folder=Path('/pilot')/row['split']/row['id']
        paths=sorted(folder.glob('image_*.jpg'))
        return [{'kind':'image','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    return []


def identity_keys(row):
    import hashlib
    keys={'id:'+row['id']}
    source=row.get('source') or {}
    if source.get('group') is not None:keys.add('group:'+source['repo']+':'+str(source['group']))
    for field in ('group_sha256','content_sha256'):
        if row.get(field):keys.add(field+':'+row[field])
    for asset in assets(row):
        keys.add('path:'+asset['path'])
        if asset.get('sha256'):keys.add('media:'+asset['sha256'])
    for material in row.get('source_materials',[]):keys.add('soccer-material:'+str(material))
    keys.add('decision:'+hashlib.sha256(json.dumps({'state':row['state'],'question':row['question'],'assets':[a.get('sha256') or a['path'] for a in assets(row)]},sort_keys=True).encode()).hexdigest())
    return keys


def bucket(row):
    import re
    family=row['task_family'];modality=row['modality']
    if family.startswith('soccer_vqa_'):return 'soccer_video' if modality=='video' else 'soccer_image' if modality=='image' else None
    if family in ('banking_choice','clinc_choice','massive_choice'):return 'routing'
    if family.startswith(('mnli_','squad_')):return 'evidence'
    if family=='helpsteer_score':return 'score'
    if family.startswith('message_queue_'):return 'queues'
    if modality=='image' and family.startswith('gui_'):return 'gui'
    if family in ('human_action_classification','object_count'):return 'action_count'
    if family=='ucf101_action_classification':return 'ucf_video'
    if modality=='image' and family in ('visual_commonsense','science_reasoning'):
        # Explicit visual cues only. This lexical filter is a conservative
        # eligibility signal, not a claim of human visual sufficiency review.
        q=row['question']['instructions'].lower()
        if re.search(r'\b(why|because|likely|purpose|famous|nationality|typically|usually|should|might|would)\b',q):return None
        if re.search(r'\b(how many|what colou?r|which colou?r|left|right|above|below|in front|behind|holding|wearing|shown on|according to (?:the )?(?:graph|chart|diagram)|based on (?:the )?(?:graph|chart|diagram))\b',q):return 'grounded_images'
    return None


def hardness(row):
    import re
    q=row['question'];criteria=q.get('criteria') or {}
    values=list(criteria.values()) if isinstance(criteria,dict) else criteria
    words=[set(re.findall(r'\w+',str(v).lower())) for v in values]
    overlap=max((len(a&b)/max(1,len(a|b)) for i,a in enumerate(words) for b in words[i+1:]),default=0.)
    return overlap+.01*min(len(values),10)


@app.function(image=image,volumes=mounts,timeout=1800,memory=8192,cpu=(2,2))
def prepare():
    import hashlib,random,copy
    from collections import Counter,defaultdict
    from pathlib import Path
    from experiments.modal_text_decisions import text_rows
    from experiments.modal_jev_omni_sports_train import _general_rows
    from transformers import AutoProcessor
    root=Path('/output')/RUN;root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        saved=json.loads((root/'manifest.json').read_text())
        if saved['actual_buckets'].get('score',0)!=3000:raise ValueError('Existing preparation does not preserve Score quota; rebuild explicitly before training')
        return saved
    corpus=inventory_rows();protected=set()
    for split in ('dev','test'):
        for row in corpus[split]:protected.update(identity_keys(row))
    for row in text_rows('calibration')+_general_rows('calibration','v2'):protected.update(identity_keys(row))
    # Protect the pilot's held-out contexts independently of its source splits.
    for split in ('dev','test'):
        for line in (Path('/output')/pilot.RUN/f'{split}.jsonl').open():protected.update(identity_keys(json.loads(line)))
    processor=AutoProcessor.from_pretrained(pilot.MODEL,revision=pilot.REVISION)
    skipped=Counter();pools=defaultdict(list);seen=set()
    def eligible(row):
        try:
            prompt,keys,target=pilot.compiled(row)
            if len(keys)<2 or len(keys)>26:raise ValueError('candidate_count')
            if len(processor.tokenizer.encode(prompt,add_special_tokens=False))>1400:raise ValueError('text_token_budget')
            aa=assets(row)
            if row['modality']!='text' and (not aa or len(aa)>8):raise ValueError('unsupported_media_count')
            if any(not Path(a['path']).is_file() for a in aa):raise ValueError('missing_media')
            if any(a['kind'] not in ('image','video') for a in aa):raise ValueError('unsupported_media_kind')
            if row['question']['type']=='choice' and len({str(v).strip().lower() for v in (row['question'].get('criteria') or {}).values()})<2:raise ValueError('invalid_candidates')
            return True
        except (ValueError,KeyError,TypeError,AttributeError) as e:skipped[str(e)[:80]]+=1;return False
    for row in corpus['train']:
        key=bucket(row)
        if not key:continue
        identities=identity_keys(row)
        if identities&protected:skipped['protected_holdout']+=1;continue
        exact={x for x in identities if x.startswith(('id:','decision:'))}
        if exact&seen:skipped['duplicate_decision']+=1;continue
        if not eligible(row):continue
        seen.update(exact);pools[key].append(row)
    result=[];actual={};selected_ids=set()
    def choose(key,n):
        group=pools[key]
        # Half from close distractors; half deterministic source/family-balanced
        # diversity. Difficulty is a heuristic, not a model correctness label.
        chosen=[];byfamily=defaultdict(list)
        for r in group:
            stratum=(r['source']['repo'],r['task_family'],r['gold']['key'])
            byfamily[stratum].append(r)
        for stratum,items in byfamily.items():
            items.sort(key=lambda r:(hardness(r),hashlib.sha256(r['id'].encode()).hexdigest()),reverse=True)
            diverse=list(items);random.Random(str(stratum)).shuffle(diverse)
            interleaved=[];used=set();hard_index=diverse_index=0
            while len(interleaved)<len(items):
                if len(interleaved)%2==0:
                    while hard_index<len(items) and items[hard_index]['id'] in used:hard_index+=1
                    if hard_index==len(items):break
                    r=items[hard_index];hard_index+=1
                else:
                    while diverse_index<len(diverse) and diverse[diverse_index]['id'] in used:diverse_index+=1
                    if diverse_index==len(diverse):break
                    r=diverse[diverse_index];diverse_index+=1
                used.add(r['id']);interleaved.append(r)
            byfamily[stratum]=interleaved
        keys=sorted(byfamily)
        while len(chosen)<n and any(byfamily.values()):
            for name in keys:
                if byfamily[name] and len(chosen)<n:
                    r=byfamily[name].pop(0)
                    if r['id'] in selected_ids:continue
                    selected_ids.add(r['id']);chosen.append(r)
        return chosen
    for key,count in QUOTAS.items():
        rows=choose(key,count);result.extend(rows);actual[key]=len(rows)
    # Soccer cap reflects all eligible direct-visual examples, not duplication.
    # Fill source scarcity within its modality, recording every adjustment.
    for modality,total,fallback in [('image',6600,['grounded_images','gui','action_count','soccer_image']),('video',5000,['ucf_video','soccer_video']),('text',8400,['routing','evidence','score','queues'])]:
        need=total-sum(r['modality']==modality for r in result)
        for key in fallback:
            if need<=0:break
            extra=choose(key,need);result.extend(extra);actual[key]+=len(extra);need-=len(extra)
        if need:raise ValueError(f'Not enough eligible {modality} examples: missing {need}; inventory={dict((k,len(v)) for k,v in pools.items())}')
    if len(result)!=20000:raise ValueError('Curated count mismatch')
    # Randomize Choice insertion order, retaining original values and labels.
    normalized=[]
    for row in result:
        r=copy.deepcopy(row)
        if r['question']['type']=='choice':
            items=list(r['question']['criteria'].items());random.Random(r['id']+':choices').shuffle(items);r['question']['criteria']=dict(items)
        if r['modality']!='text' and (r.get('media') or {}).get('kind') not in ('modal_image','modal_video','modal_media_sequence'):r['media']={'kind':'modal_media_sequence','assets':assets(r)}
        r['curation']={'bucket':bucket(r),'selection':'source annotation; lexical visual filter; distractor overlap and source/family/label diversity','visual_sufficiency_human_reviewed':False}
        normalized.append(r)
    random.Random(3407).shuffle(normalized)
    # Diagnostics remain small; public benchmarks are separate, evaluation-only.
    selected_train_keys=set().union(*(identity_keys(r) for r in normalized))
    splits={'train':normalized}
    for split in ('dev','test'):
        available=[r for r in corpus[split] if bucket(r) and eligible(r) and not (identity_keys(r)&selected_train_keys)]
        chosen=[]
        for modality in ('text','image','video'):
            rows=[r for r in available if r['modality']==modality]
            rows.sort(key=lambda r:hashlib.sha256((split+r['id']).encode()).hexdigest())
            chosen.extend(rows[:64])
        if len(chosen)!=192:raise ValueError('Missing held-out diagnostic modality')
        splits[split]=chosen
    manifest={'run':RUN,'model':pilot.MODEL,'revision':pilot.REVISION,'seed':3407,'unique_train_rows':20000,'train_rows':20000,'dev_rows':192,'test_rows':192,'requested_buckets':QUOTAS,'actual_buckets':actual,'eligible_pool_counts':{k:len(v) for k,v in pools.items()},'skipped':dict(skipped),'files':{},'split_counts':{},'train_sources':dict(Counter(r['source']['repo'] for r in normalized)),'train_families':dict(Counter(r['task_family'] for r in normalized)),'asset_count_histogram':dict(Counter(str(len(assets(r))) for r in normalized if r['modality']!='text')),'train_question_types':dict(Counter(r['question']['type'] for r in normalized)),'protocol':'single-question Ollama-shaped token scoring; image and ordered video frame extension','limitations':['Lexical image-grounding filter is not human visual sufficiency certification.','Source grouping and exact media/material checks do not establish universal match-level SoccerNet separation or near-duplicate exclusion.','Human action/UCF tasks remain source-derived class decisions, not official full-label benchmark scoring.','Mixed source permissions remain research-only.']}
    # Protect selected dev/test against each other; reject silent leakage.
    identities={}
    for split,rows in splits.items():
        for r in rows:
            for key in identity_keys(r):
                if key in identities and identities[key]!=split:raise ValueError('Cross-split identity overlap: '+key)
                identities[key]=split
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);(root/f'{split}.jsonl').write_text(content)
        manifest['files'][split]=hashlib.sha256(content.encode()).hexdigest();manifest['split_counts'][split]=dict(Counter(r['modality'] for r in rows))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));pilot.output.commit();return manifest

@app.function(image=image,volumes=mounts,gpu='H100',timeout=13800,memory=(32768,65536),cpu=(4,4),max_containers=1,scaledown_window=10)
def train():
    return pilot.train_impl(run=RUN,runtime_seconds=13200,curate=True)

@app.function(image=image,volumes={'/output':pilot.output},timeout=300)
def status():
    from pathlib import Path
    pilot.output.reload();root=Path('/output')/RUN
    return {p.name:json.loads(p.read_text()) for p in root.glob('*.json') if p.name in ('manifest.json','progress.json','report.json','budget-stop.json','launch.json','launch-intent.json')}

@app.function(image=modal.Image.debian_slim(python_version="3.11").add_local_python_source("experiments"),volumes={"/output":pilot.output},timeout=2100,memory=256,cpu=(.125,.125),max_containers=1)
def launch_when_ready(metered_usd:float):
    import time
    from pathlib import Path
    if 30-metered_usd<26:raise ValueError("Insufficient free credits for $18 training allocation plus $8 evaluation/export reserve")
    root=Path('/output')/RUN;root.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        pilot.output.reload()
        if (root/'launch.json').exists():return json.loads((root/'launch.json').read_text())
        if (root/'manifest.json').exists():break
        time.sleep(20)
    else:raise TimeoutError("Curated preparation did not finish; no GPU job launched")
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['unique_train_rows']!=20000 or manifest['actual_buckets']['score']!=3000 or manifest['train_sources'].get('SoccerNet/SN-VQA-2026',0)<3000:raise ValueError("Curated source/type gates failed; no GPU job launched")
    if (root/'launch-intent.json').exists():raise ValueError("Existing launch intent requires readback before retry; refusing duplicate GPU submission")
    intent={'run':RUN,'gpu':'H100','timeout_seconds':13800,'training_compute_bound_usd':18,'monthly_metered_usd_at_submission':metered_usd,'paid_spend_cap_usd':0,'usage_cap_usd':30}
    (root/'launch-intent.json').write_text(json.dumps(intent,indent=2));pilot.output.commit()
    call=train.spawn();launch={**intent,'training_call_id':call.object_id,'status':'submitted','submitted_unix':time.time()}
    (root/'launch.json').write_text(json.dumps(launch,indent=2));pilot.output.commit();return launch


@app.local_entrypoint()
def main(mode:str="prepare"):
    print(json.dumps({"audit":audit,"prepare":prepare,"train":train,"status":status}[mode].remote(),indent=2))
