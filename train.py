"""Reproducible baseline and Transformer experiments. Run from the project root."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from geoimpact.core import LABELS


def validate_data(frame):
    required = {'text','label','event_id','split','source_url','license','synthetic'}
    if not required <= set(frame.columns):
        raise ValueError('Missing columns: ' + str(required-set(frame.columns)))
    if frame[list(required)].isna().any().any():
        raise ValueError('Dataset fields cannot be empty.')
    if not set(frame.label) <= set(LABELS):
        raise ValueError('Unknown event label.')
    if set(frame.split) != {'train','validation','test'}:
        raise ValueError('Supply train, validation, and test splits.')
    for column in ['event_id', 'text']:
        normalized = frame[column].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
        if normalized.eq('').any():
            raise ValueError('Empty ' + column)
        if frame.assign(normalized=normalized).groupby('normalized').split.nunique().max()>1:
            raise ValueError('Cross-split leakage in ' + column)
    for split in ['train','validation','test']:
        if set(frame[frame.split==split].label) != set(LABELS):
            raise ValueError('Each split must contain every label.')


def metrics(y, pred):
    return {'accuracy': accuracy_score(y,pred), 'macro_f1': f1_score(y,pred,labels=LABELS,average='macro',zero_division=0),
        'per_class': classification_report(y,pred,labels=LABELS,output_dict=True,zero_division=0),
        'confusion_matrix': confusion_matrix(y,pred,labels=LABELS).tolist(), 'label_order': LABELS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline','transformer'], required=True)
    parser.add_argument('--data', default='data/smoke.csv')
    parser.add_argument('--model', default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Epochs and batch size must be positive.')
    random.seed(453); np.random.seed(453)
    frame = pd.read_csv(args.data, keep_default_na=False)
    validate_data(frame)
    splits = {s: frame[frame.split==s].reset_index(drop=True) for s in ['train','validation','test']}
    start = time.perf_counter()
    out = Path('artifacts'); out.mkdir(exist_ok=True)
    metadata = {'mode': args.mode, 'seed': 453, 'dataset_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'synthetic': bool(frame.synthetic.astype(str).str.lower().isin(['true','1']).any()),
        'split_sizes': {s:len(v) for s,v in splits.items()}, 'configuration': vars(args)}
    if args.mode == 'baseline':
        import joblib
        from sklearn.pipeline import make_pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), LogisticRegression(max_iter=1000, random_state=453))
        model.fit(splits['train'].text, splits['train'].label)
        metadata['validation'] = metrics(splits['validation'].label, model.predict(splits['validation'].text))
        prediction = model.predict(splits['test'].text)
        joblib.dump(model, out/'baseline.joblib')
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.manual_seed(453); torch.set_num_threads(4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)), label2id={v:i for i,v in enumerate(LABELS)},
            local_files_only=args.local_files_only).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
        def predict(part):
            model.eval(); predicted = []
            with torch.no_grad():
                for i in range(0,len(part),args.batch_size):
                    tokens = tokenizer(part.text.iloc[i:i+args.batch_size].tolist(), padding=True,
                        truncation=True, max_length=256, return_tensors='pt').to(device)
                    predicted.extend(LABELS[k] for k in model(**tokens).logits.argmax(-1).cpu().tolist())
            return predicted
        best = -1.; history = []
        for epoch in range(args.epochs):
            model.train(); loss_sum = 0.
            order = np.random.default_rng(453+epoch).permutation(len(splits['train']))
            for i in range(0,len(order),args.batch_size):
                batch = splits['train'].iloc[order[i:i+args.batch_size]]
                tokens = tokenizer(batch.text.tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
                labels = torch.tensor([LABELS.index(v) for v in batch.label],device=device)
                optimizer.zero_grad(); loss = model(**tokens, labels=labels).loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); loss_sum += float(loss.detach())*len(batch)
            val = metrics(splits['validation'].label,predict(splits['validation']))
            history.append({'epoch':epoch+1,'train_loss':loss_sum/len(order),'validation_macro_f1':val['macro_f1']})
            print(json.dumps(history[-1]),flush=True)
            if val['macro_f1']>best:
                best = val['macro_f1']; metadata['best_epoch'] = epoch+1; metadata['validation'] = val
                model.save_pretrained(out/'transformer'); tokenizer.save_pretrained(out/'transformer')
        model = AutoModelForSequenceClassification.from_pretrained(out/'transformer',local_files_only=True).to(device)
        prediction = predict(splits['test'])
        metadata.update(device=str(device),total_parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),history=history)
    metadata['test'] = metrics(splits['test'].label,prediction)
    metadata['runtime_seconds'] = round(time.perf_counter()-start,2)
    metadata['interpretation'] = 'Synthetic pipeline smoke test only; not evidence of real-world performance.' if metadata['synthetic'] else 'Review dataset provenance and held-out evaluation before deployment.'
    (out/f'{args.mode}-metrics.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__':
    main()
