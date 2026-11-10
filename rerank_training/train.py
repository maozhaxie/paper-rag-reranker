"""Reproducible CPU pointwise reranking experiment; no cloud model calls."""
import os,sys,json,random,hashlib,time,re,unicodedata,math,platform
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,set_seed
ROOT=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def norm(t):return re.sub(r'\s+','',unicodedata.normalize('NFKC',t)).lower()
def digest(t):return hashlib.sha256(norm(t).encode()).hexdigest()
def prepare():
    raw=pq.read_table(ROOT/'data/t2.parquet').to_pylist()
    order=list(range(len(raw)));random.Random(20261108).shuffle(order)
    usedq,usedd=set(),set();splits={};reasons=Counter();cursor=0
    for split,target in [('train',1000),('validation',200),('test',400)]:
        out=[]
        while len(out)<target and cursor<len(order):
            rowid=order[cursor];cursor+=1;row=raw[rowid];query=row['query'].strip()
            pos=list(dict.fromkeys(t.strip() for t in row['positive'] if t.strip()))[:2]
            neg=list(dict.fromkeys(t.strip() for t in row['negative'] if t.strip()))[:6]
            docs=pos+neg;labels=[1]*len(pos)+[0]*len(neg)
            hashes=[digest(t) for t in docs]
            if not pos or not neg:reasons['empty_class']+=1;continue
            if len(set(hashes))!=len(hashes):reasons['within_query_duplicate']+=1;continue
            if digest(query) in usedq or any(h in usedd for h in hashes):reasons['cross_split_or_query_duplicate']+=1;continue
            ix=list(range(len(docs)));random.Random(rowid+97).shuffle(ix)
            out.append(dict(qid=rowid,query=query,documents=[docs[i] for i in ix],labels=[labels[i] for i in ix]))
            usedq.add(digest(query));usedd.update(hashes)
        if len(out)!=target:raise RuntimeError(f'insufficient disjoint groups {split}: {len(out)}')
        splits[split]=out
        with (ROOT/f'data/{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    audit=dict(raw_groups=len(raw),examined=cursor,excluded=dict(reasons),splits={k:dict(queries=len(v),pairs=sum(len(r['labels']) for r in v)) for k,v in splits.items()})
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not {digest(r['query']) for r in splits[a]} & {digest(r['query']) for r in splits[b]}
        assert not {digest(t) for r in splits[a] for t in r['documents']} & {digest(t) for r in splits[b] for t in r['documents']}
    audit['exact_query_and_passage_overlap']=0
    dump(ROOT/'artifacts/data_audit.json',audit)
    return splits
def metrics(scores,labels):
    order=sorted(range(len(scores)),key=lambda i:-float(scores[i]))
    relevance=[labels[i] for i in order]
    rank=next((i+1 for i,r in enumerate(relevance) if r),None)
    dcg=sum(r/math.log2(i+2) for i,r in enumerate(relevance[:10]))
    ideal=sum(1/math.log2(i+2) for i in range(min(10,sum(labels))))
    return dict(mrr10=1/rank if rank and rank<=10 else 0,ndcg10=dcg/ideal if ideal else 0,hit1=float(bool(relevance[0])),hit5=float(any(relevance[:5])))
def aggregate(rows):return {k:float(np.mean([r[k] for r in rows])) for k in ['mrr10','ndcg10','hit1','hit5']}
def encode(rows,tok):
    pairs=[(r['query'],d) for r in rows for d in r['documents']]
    enc=tok([q for q,d in pairs],[d for q,d in pairs],max_length=128,truncation='longest_first',padding='max_length',return_tensors='pt')
    labels=torch.tensor([l for r in rows for l in r['labels']],dtype=torch.long)
    return dict(enc),labels
def predict(model,enc,rows):
    model.eval();scores=[]
    with torch.inference_mode():
        for s in range(0,len(enc['input_ids']),32):
            scores.extend(torch.softmax(model(**{k:v[s:s+32] for k,v in enc.items()}).logits,dim=1)[:,1].tolist())
    results=[];offset=0
    for row in rows:
        n=len(row['labels']);sc=scores[offset:offset+n];offset+=n
        results.append(dict(qid=row['qid'],scores=sc,**metrics(sc,row['labels'])))
    return results
def tokens(text):
    text=norm(text);return list(text)+[text[i:i+2] for i in range(len(text)-1)]
def bm25(row):
    docs=[tokens(d) for d in row['documents']];query=Counter(tokens(row['query']));n=len(docs);avg=sum(map(len,docs))/n
    df=Counter(t for d in docs for t in set(d));scores=[]
    for d in docs:
        tf=Counter(d);score=0
        for t in query:
            if tf[t]:score+=math.log(1+(n-df[t]+.5)/(df[t]+.5))*tf[t]*2.5/(tf[t]+1.5*(.25+.75*len(d)/avg))
        scores.append(score)
    return dict(qid=row['qid'],scores=scores,**metrics(scores,row['labels']))
def main():
    start=time.monotonic();out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    assert metrics([.1,.8],[0,1])['mrr10']==1
    assert metrics([.8,.1],[0,1])['mrr10']==.5
    splits=prepare();tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    encoded={k:encode(v,tok) for k,v in splits.items()}
    cfg=dict(seeds=[42,43,44],epochs=3,learning_rate=2e-4,batch_size=16,max_length=128,weight_decay=.01,threads=4,device='cpu',python=sys.executable,torch=torch.__version__,platform=platform.platform())
    dump(out/'config.json',cfg)
    trainenc,labels=encoded['train'];history=[];chosen=[]
    for seed in cfg['seeds']:
        set_seed(seed)
        model=AutoModelForSequenceClassification.from_pretrained(ROOT/'model',num_labels=2,local_files_only=True)
        model.save_pretrained(out/f'initial_{seed}',safe_serialization=True)
        optim=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
        best=-1;best_epoch=None;seedstart=time.monotonic()
        for epoch in range(1,4):
            model.train();permutation=torch.randperm(len(labels),generator=torch.Generator().manual_seed(seed*100+epoch));losses=[]
            for step,s in enumerate(range(0,len(labels),16)):
                ix=permutation[s:s+16];optim.zero_grad(set_to_none=True)
                loss=model(**{k:v[ix] for k,v in trainenc.items()},labels=labels[ix]).loss
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step();losses.append(loss.item())
                if step%100==0:print(f'seed={seed} epoch={epoch} step={step} loss={loss.item():.4f}',flush=True)
            val=predict(model,encoded['validation'][0],splits['validation']);summary=aggregate(val)
            record=dict(seed=seed,epoch=epoch,train_loss=float(np.mean(losses)),validation=summary,elapsed_seconds=time.monotonic()-seedstart)
            history.append(record);dump(out/'history.json',history);print(json.dumps(record),flush=True)
            if summary['mrr10']>best:
                best=summary['mrr10'];best_epoch=epoch;model.save_pretrained(out/f'best_{seed}',safe_serialization=True);tok.save_pretrained(out/f'best_{seed}')
        chosen.append(dict(seed=seed,epoch=best_epoch,validation_mrr10=best))
        del model,optim
    # Holdout evaluation starts only after all validation-based selections are fixed.
    dump(out/'selected_checkpoints.json',chosen)
    baseline=[bm25(r) for r in splits['test']];allruns=[]
    for seed in cfg['seeds']:
        run={'seed':seed}
        for mode,path in [('untrained',out/f'initial_{seed}'),('finetuned',out/f'best_{seed}')]:
            model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True)
            predictions=predict(model,encoded['test'][0],splits['test']);dump(out/f'{mode}_test_{seed}.json',predictions)
            run[mode]=aggregate(predictions);del model
        allruns.append(run)
    rng=np.random.default_rng(20261108)
    trained=np.array([[r['mrr10'] for r in json.loads((out/f'finetuned_test_{s}.json').read_text())] for s in cfg['seeds']]).mean(axis=0)
    delta=trained-np.array([r['mrr10'] for r in baseline]);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
    summary={mode:{metric:dict(mean=float(np.mean([r[mode][metric] for r in allruns])),std=float(np.std([r[mode][metric] for r in allruns],ddof=1))) for metric in ['mrr10','ndcg10','hit1','hit5']} for mode in ['untrained','finetuned']}
    report=dict(config=cfg,parameter_count=sum(p.numel() for p in AutoModelForSequenceClassification.from_pretrained(out/'best_42',local_files_only=True).parameters()),bm25=aggregate(baseline),runs=allruns,summary=summary,paired_bootstrap_mrr10_delta_vs_bm25=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),queries=len(delta),resamples=2000),elapsed_seconds=time.monotonic()-start)
    dump(out/'bm25_test.json',baseline);dump(out/'metrics.json',report)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.jsonl','.safetensors','.md'] and '.venv' not in p.parts}
    dump(out/'file_hashes.json',files)
    print('TRAINING_COMPLETE',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
