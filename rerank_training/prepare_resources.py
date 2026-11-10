import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL='151f4204cbb5784c4126a89a94ef0bb044273ac0'
DATA='76631901a18387f85eaa53e5450019b87ad58ef9'
items={f'model/{name}':f'https://huggingface.co/uer/chinese_roberta_L-2_H-128/resolve/{MODEL}/{name}' for name in ['config.json','pytorch_model.bin','special_tokens_map.json','tokenizer_config.json','vocab.txt','README.md']}
items['data/t2.parquet']=f'https://huggingface.co/datasets/C-MTEB/T2Reranking/resolve/{DATA}/data/dev-00000-of-00001-65d96bde8023d9b9.parquet'
items['data/README.md']=f'https://huggingface.co/datasets/C-MTEB/T2Reranking/resolve/{DATA}/README.md'
records=[]
for rel,url in items.items():
    path=ROOT/rel;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.is_file():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=90) as response, path.with_suffix(path.suffix+'.part').open('wb') as out:
                    while chunk:=response.read(1024*1024):out.write(chunk)
                path.with_suffix(path.suffix+'.part').replace(path)
                break
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    records.append(dict(file=rel,url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    print(rel,path.stat().st_size,flush=True)
(ROOT/'resources.json').write_text(json.dumps(dict(model_revision=MODEL,data_revision=DATA,files=records),indent=2),encoding='utf-8')
