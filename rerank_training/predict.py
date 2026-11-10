"""Offline host inference. Input JSON contains query and documents."""
import argparse,json,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parent
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);args=parser.parse_args()
    selection=json.loads((ROOT/'artifacts/final_selection.json').read_text())
    modelpath=ROOT/'artifacts'/f"best_{selection['seed']}"
    tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(modelpath,local_files_only=True).eval()
    row=json.loads(Path(args.input).read_text(encoding='utf-8'))
    docs=row['documents'];enc=tok([row['query']]*len(docs),docs,max_length=128,truncation=True,padding=True,return_tensors='pt')
    with torch.inference_mode():scores=torch.softmax(model(**enc).logits,dim=1)[:,1].tolist()
    results=sorted([dict(index=i,score=s,text=docs[i]) for i,s in enumerate(scores)],key=lambda r:-r['score'])
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
