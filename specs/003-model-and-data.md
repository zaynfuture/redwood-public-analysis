# Model and data specification

## Training task
Single-label event classification: natural_hazard, conflict, public_health,
infrastructure, trade_policy, other. Each row is one event-focused excerpt.
For multiple events, annotate separate excerpts. Unknown/out-of-domain cases use
other; do not force the most dramatic category.

Baseline: TF-IDF word/bi-gram features + logistic regression.
Deep learning: fine-tune distilbert-base-uncased with a six-class classification
head. Fix seed 453, max length 256, AdamW, learning rate 2e-5. Select the best epoch
by validation macro-F1; evaluate test once after selection. Report all class F1,
confusion matrix, accuracy, runtime, parameter counts, and dataset fingerprint.
Scores are uncalibrated. The model sees event evidence, not geographic risk labels.

## Data contract
CSV columns: text,label,event_id,split,source_url,license,synthetic.
Splits: train,validation,test. Event IDs and normalized text must not cross splits.
Real dataset construction must also group syndicated and near-duplicate stories,
use a temporal holdout and source diversity, and freeze the test set before tuning.
The code checks exact duplicates/group overlap; humans must check near duplicates.

## Annotation plan
Collect 600–1,200 licensed/permission-compatible excerpts, roughly class-balanced,
with publication time and source metadata in a private annotation ledger. Two
annotators label a subset independently and adjudicate disagreements. Record
agreement, inclusion rules, and ambiguity decisions. News access does not grant
redistribution rights; keep full copyrighted articles out of public repositories.

For extraction evaluation, add human gold event/location/source spans and review
possible impacts for support versus speculation. A classification score alone
cannot validate the business impact analysis.

The checked-in fixture is original synthetic text, explicitly marked synthetic.
Training on it proves that weights update and artifacts load. Its tiny test set
cannot support real-world accuracy claims. All final-project performance targets
remain unverified until evaluation on a real independently annotated test set.

## Rubric alignment
Business app (25%): operational use case, interactive workflow, user study.
Model (15%): baseline vs fine-tuned deep model, selection rationale, resource costs.
Expected results (10%): targets vs actual results and error analysis.
Coding (20%): modular Python, contracts, tests, failure handling.
Documentation/presentation (10%): README, specs, model/data cards, demo script.
The missing screenshot section A must be supplied before claiming complete coverage.
