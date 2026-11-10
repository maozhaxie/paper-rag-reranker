import json,hashlib,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
chosen=json.loads((ROOT/'artifacts/selected_checkpoints.json').read_text())
best=sorted(chosen,key=lambda r:(-r['validation_mrr10'],r['seed']))[0]
(ROOT/'artifacts/final_selection.json').write_text(json.dumps(best,indent=2))
changes=[]
for s in [42,43,44]:
    a=load_file(ROOT/f'artifacts/initial_{s}/model.safetensors');b=load_file(ROOT/f'artifacts/best_{s}/model.safetensors')
    names=[k for k in a if k.startswith('bert.encoder')]
    delta=sum((a[k]-b[k]).double().square().sum().item() for k in names)**.5
    assert delta>0
    changes.append(dict(seed=s,encoder_parameter_change_l2=delta))
tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
truncation={}
for split in ['train','validation','test']:
    rows=[json.loads(l) for l in (ROOT/f'data/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    pairs=[(r['query'],d) for r in rows for d in r['documents']]
    enc=tok([q for q,d in pairs],[d for q,d in pairs],padding=False,truncation=False)
    truncation[split]=dict(total=len(pairs),over_128=sum(len(ids)>128 for ids in enc['input_ids']))
audit=dict(weight_changes=changes,truncation=truncation,final_selection_rule='highest validation MRR@10, tie lowest seed; never test-selected')
(ROOT/'artifacts/final_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
