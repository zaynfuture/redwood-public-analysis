"""Upload a reviewed checkpoint, then verify downloaded bytes and inference."""
import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from geoimpact.extract import make_pipeline, classify_batch

ALLOWED = {'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
           'vocab.txt','vocab.json','merges.txt','added_tokens.json','model.safetensors.index.json'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def stage_checkpoint(model_dir, metrics_path, destination, repo_id):
    metrics=json.loads(metrics_path.read_text())
    if metrics.get('mode')!='transformer': raise ValueError('Transformer metrics are required.')
    if not (model_dir/'config.json').exists(): raise ValueError('Missing trained model config.')
    destination.mkdir(parents=True,exist_ok=True)
    files=[p for p in model_dir.iterdir() if p.name in ALLOWED or p.name.endswith('.safetensors')]
    if not any(p.suffix=='.safetensors' for p in files): raise ValueError('Missing safetensors checkpoint.')
    for path in files: shutil.copy2(path,destination/path.name)
    # Keep local paths and credentials out of metadata.
    summary={k:metrics[k] for k in ['mode','seed','dataset_sha256','synthetic','split_sizes','test','validation','runtime_seconds','interpretation']}
    (destination/'evaluation.json').write_text(json.dumps(summary,indent=2))
    status='SYNTHETIC SMOKE TEST — NOT A VALIDATED NEWS MODEL' if metrics['synthetic'] else 'Research checkpoint — review evaluation before use'
    card=f'''---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert/distilbert-base-uncased
tags:
- news-event-classification
- academic-project
- redwood-public-analysis
---
# Redwood Public Analysis — Event Classifier

**{status}**

This is a six-class DistilBERT event classifier fine-tuned for the Redwood Public
Analysis Streamlit research application. It does not generate geographic risk
predictions and is not the generative news extractor.

## Training and evaluation

Seed: {metrics['seed']}. Split sizes: {metrics['split_sizes']}.
Dataset SHA256: `{metrics['dataset_sha256']}`.
Test accuracy: {metrics['test']['accuracy']:.4f}.
Test macro-F1: {metrics['test']['macro_f1']:.4f}.
Runtime: {metrics['runtime_seconds']} seconds on the recorded training environment.
See evaluation.json for per-class metrics and the confusion matrix.

{metrics['interpretation']}

The included smoke checkpoint used original fictional examples with the split
sizes above for {metrics['configuration']['epochs']} epoch(s). It is undertrained and must not support operational decisions.
Its poor metrics are reported rather than hidden. For a subsequent real-data
release, update this model card to describe the actual dataset and training run.

## Intended use and limitations

Academic pipeline demonstration and reproducibility only. Labels:
natural_hazard, conflict, public_health, infrastructure, trade_policy, other.
Not intended for hazard warnings, financial decisions, emergency response, or
inference about individuals. Scores are uncalibrated. English-only evaluation;
no demonstrated robustness to news domains, ambiguity, or adversarial content.
Classification is not factual verification or a forecast of business consequences.

## Load the uploaded model

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="{repo_id}", token=True)
classifier(["A flood interrupted deliveries to the warehouse."],
           batch_size=8, truncation=True, max_length=256)
```

Private repositories require an authenticated account with access. Pin the commit
revision recorded in the local hub-publication.json for reproducible inference.
The base model is Apache-2.0; the original fictional fixture is CC0-1.0.
No source articles, secrets, account tokens, or application data are included.
'''
    if not metrics['synthetic']:
        paragraph_start=card.index('The included smoke checkpoint')
        paragraph_end=card.index('## Intended use',paragraph_start)
        card=card[:paragraph_start]+('Review data provenance, annotation quality, and held-out results before use.\n'
                                  'Operational validity is not established by publishing a checkpoint.\n\n')+card[paragraph_end:]
    (destination/'README.md').write_text(card)
    return {p.name:digest(p) for p in destination.iterdir() if p.is_file()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-id',required=True)
    p.add_argument('--model-dir',default='artifacts/transformer')
    p.add_argument('--metrics',default='artifacts/transformer-metrics.json')
    p.add_argument('--public',action='store_true',help='Explicitly create a public repository instead of private.')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--update-existing',action='store_true',help='Explicitly publish a new revision of an existing repository.')
    args=p.parse_args()
    if '/' not in args.repo_id: p.error('Use owner/repository.')
    model_dir=Path(args.model_dir)
    staged=Path('artifacts/releases') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest=stage_checkpoint(model_dir,Path(args.metrics),staged,args.repo_id)
    if args.prepare_only:
        print(json.dumps({'staged_files':list(manifest),'directory':str(staged)},indent=2));return
    api=HfApi()
    info=api.whoami()
    owners=[info['name']]+[org['name'] for org in info.get('orgs',[])]
    if args.repo_id.split('/')[0] not in owners: raise ValueError('Repository owner is not the authenticated user or organization.')
    # Do not overwrite an existing repository or change its visibility implicitly.
    if args.update_existing:
        existing = api.model_info(args.repo_id)
        if existing.private != (not args.public):
            raise ValueError('Requested visibility differs from the existing repository; no visibility changes are made.')
    else:
        api.create_repo(repo_id=args.repo_id,repo_type='model',private=not args.public,exist_ok=False)
    commit=api.upload_folder(repo_id=args.repo_id,repo_type='model',folder_path=staged,
        commit_message='Add fine-tuned event classifier with explicit smoke-test evaluation')
    report={'repo_id':args.repo_id,'url':f'https://huggingface.co/{args.repo_id}',
            'revision':commit.oid,'visibility':'public' if args.public else 'private','verification':'pending'}
    report_path=Path('artifacts/hub-publication.json')
    report_path.write_text(json.dumps(report,indent=2))
    with tempfile.TemporaryDirectory(prefix='redwood-hf-verify-') as temp:
        downloaded=Path(snapshot_download(args.repo_id,revision=commit.oid,local_dir=temp))
        for name,expected in manifest.items():
            if digest(downloaded/name)!=expected: raise ValueError('Downloaded file checksum mismatch: '+name)
        samples=['A flood interrupted warehouse deliveries.','New tariffs increased import costs.']
        remote=classify_batch(samples,make_pipeline('text-classification',str(downloaded),True))
        local=classify_batch(samples,make_pipeline('text-classification',str(model_dir),True))
        if remote!=local: raise ValueError('Downloaded predictions differ from the local checkpoint.')
        report.update(verification='passed',verified_file_count=len(manifest),inference=remote,
                      checks=['SHA256 of every staged file','Downloaded revision loads through Hugging Face pipeline','Predictions match local checkpoint'])
        report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
