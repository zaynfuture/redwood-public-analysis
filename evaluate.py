"""Evaluate saved classifier weights without training or changing the checkpoint."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import metrics, validate_data
from geoimpact.core import LABELS


def evaluate_batches(model, tokenizer, frame, batch_size=4, device='cpu'):
    if batch_size < 1 or frame.empty:
        raise ValueError('Use a positive batch size and a nonempty evaluation split.')
    mapping = {str(k): int(v) for k,v in model.config.label2id.items()}
    if set(mapping) != set(LABELS) or set(mapping.values()) != set(range(len(LABELS))):
        raise ValueError('Checkpoint label mapping does not match this project.')
    reverse = {v:k for k,v in mapping.items()}
    model.to(device); model.eval()
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    start=time.perf_counter(); loss_total=0.; predictions=[]
    with torch.inference_mode():
        for offset in range(0,len(frame),batch_size):
            batch=frame.iloc[offset:offset+batch_size]
            tokens=tokenizer(batch.text.tolist(),padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
            target=torch.tensor([mapping[label] for label in batch.label],device=device)
            result=model(**tokens,labels=target)
            loss_total+=float(result.loss)*len(batch)
            predictions.extend(reverse[i] for i in result.logits.argmax(-1).cpu().tolist())
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    runtime=time.perf_counter()-start
    scores=metrics(frame.label,predictions)
    return {'eval_loss':loss_total/len(frame),'eval_accuracy':scores['accuracy'],
        'eval_macro_f1':scores['macro_f1'],'eval_runtime':runtime,
        'eval_samples_per_second':len(frame)/runtime,
        'eval_steps_per_second':math.ceil(len(frame)/batch_size)/runtime,
        'sample_count':len(frame),'metrics':scores,
        'predictions':[{'event_id':str(row.event_id),'gold':row.label,'predicted':pred}
                       for row,pred in zip(frame.itertuples(),predictions)]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',default='artifacts/transformer')
    parser.add_argument('--data',default='data/smoke.csv')
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--output')
    args=parser.parse_args()
    frame=pd.read_csv(args.data,keep_default_na=False);validate_data(frame)
    part=frame[frame.split==args.split].reset_index(drop=True)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.model_dir,local_files_only=True)
    result=evaluate_batches(model,tokenizer,part,args.batch_size,device)
    result.update(split=args.split,model_dir=args.model_dir,device=device,
        batch_size=args.batch_size,max_length=256,
        synthetic=bool(part.synthetic.astype(str).str.lower().isin(['true','1']).any()),
        dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        timing_scope='Tokenization and inference; excludes model loading. Single pass, not a latency benchmark.')
    output=Path(args.output or f'artifacts/evaluation-{args.split}.json')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
