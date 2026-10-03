# How to fine-tune and evaluate the model

## 1. Identify which model is trained

Redwood Public Analysis has two different model tasks:

| Component | Model | Training status | Output |
|---|---|---|---|
| Event classifier | Hugging Face `distilbert-base-uncased` | Full supervised fine-tuning implemented and run on synthetic smoke data | One of six event labels |
| News extractor | Hugging Face `Qwen/Qwen3-0.6B` | Pretrained model; not fine-tuned by this project | Primary event, source sentence references, location, scope, possible impacts |

Changing a prompt, applying JSON constraints, or using `pipeline()` does not update
weights and is not fine-tuning. The classifier trains all 66,958,086 parameters,
including the encoder and a newly initialized six-class classification head.
The user-supplied screenshot demonstrates evaluation; its metric values are not
our results, and it does not require replacing our explicit PyTorch loop with Trainer.

## 2. Prepare labeled data and prevent leakage

Create a CSV with these columns:

| Column | Meaning |
|---|---|
| `text` | One event-focused English news excerpt |
| `label` | `natural_hazard`, `conflict`, `public_health`, `infrastructure`, `trade_policy`, or `other` |
| `event_id` | Identifier shared by excerpts describing the same event |
| `split` | `train`, `validation`, or `test` |
| `source_url` | Source reference; use `original-fiction` for fictional fixtures |
| `license` | Verified reuse permission or license |
| `synthetic` | `true` for synthetic examples; `false` for real annotated examples |

Each split must contain all six labels. Keep the same event and syndicated or
near-duplicate stories in one split. `validate_data()` rejects exact normalized
text and event-ID overlap; near-duplicate and source/date checks still require
annotation review. Use a chronological holdout for a realistic final evaluation.
Fit features and update weights only on training data. Use validation for model
selection and tuning. Freeze the test set; do not use its results to choose epochs.

The provided `data/smoke.csv` contains 48 original fictional examples: 36 train,
6 validation, and 6 test. It tests execution, not real-world generalization.
For the final project, follow the annotation plan in `specs/003-model-and-data.md`.

## 3. Run baseline training

Run all commands from the project root:

```bash
cd /Users/zaynfuture/Developments/redwood-public-analysis
source .venv/bin/activate
python train.py --mode baseline --data data/smoke.csv
```

The baseline fits TF-IDF unigrams/bigrams and logistic regression on the training
split, then reports validation and test metrics. Outputs:
`artifacts/baseline.joblib` and `artifacts/baseline-metrics.json`.
Use the identical data and splits for the Transformer comparison.

## 4. Fine-tune DistilBERT

Reproduce the existing one-epoch smoke experiment:

```bash
python train.py --mode transformer \
  --model distilbert-base-uncased \
  --data data/smoke.csv \
  --epochs 1 --batch-size 4 --local-files-only
```

For a new real-data experiment, replace the CSV path and choose epochs using
validation results. For example, after creating your annotated dataset:

```bash
python train.py --mode transformer \
  --data data/annotated-news.csv --epochs 3 --batch-size 4
```

The second command is a template: `data/annotated-news.csv` is not provided.
Omit `--local-files-only` to allow downloading the pretrained Hugging Face model.
Training commands overwrite the corresponding local model and metrics; archive
previous runs first when comparing experiments. They do not upload automatically.

The actual implementation in `train.py` performs:

1. Load the tokenizer and pretrained encoder with a six-class classification head.
2. Tokenize batches with dynamic padding and truncate inputs to 256 tokens.
3. Set training mode and compute cross-entropy loss using the gold class labels.
4. Clear gradients, backpropagate the loss, clip gradient norm to 1.0, and update
   all weights using AdamW (learning rate `2e-5`, weight decay `0.01`).
5. At each epoch, switch to evaluation mode and disable gradients for validation.
6. Save the checkpoint with the best validation macro-F1; ties keep the earlier one.
7. Reload the selected checkpoint and evaluate the test split after selection.

Other settings: seed 453; batch size defaults to 4; default epochs 3; training uses
CUDA when available, otherwise CPU. No LoRA, encoder freezing, early stopping,
learning-rate scheduler, or mixed-precision training is used. Seeds improve
repeatability but do not guarantee identical results across hardware/library versions.

Outputs: `artifacts/transformer/` (weights, config, tokenizer) and
`artifacts/transformer-metrics.json` (configuration, data fingerprint, loss history,
selected epoch, metrics, confusion matrix, parameter counts, and total runtime).

## 5. Evaluate the saved model without retraining

Use the standalone evaluator for a checkpoint already on disk:

```bash
python evaluate.py --model-dir artifacts/transformer \
  --data data/smoke.csv --split validation

# Final held-out evaluation after model selection:
python evaluate.py --model-dir artifacts/transformer \
  --data data/smoke.csv --split test
```

The evaluator loads the saved weights, calls `model.eval()`, uses
`torch.inference_mode()`, and calculates predictions and loss without an optimizer
or weight updates. It respects the checkpoint's label mapping. The final partial
batch contributes proportionally to the sample-weighted average loss.

This serves the same evaluation purpose as the screenshot's `trainer.evaluate()`.
Our project uses an explicit PyTorch evaluation loop, so there is no `trainer`
object to call directly. Do not copy the screenshot's results into the report.

| Report field | Meaning | Interpretation |
|---|---|---|
| `eval_loss` | Mean cross-entropy across labeled examples | Lower is better; compare on the same split/task |
| `eval_accuracy` | Correct predictions / number of examples | Higher is better; can hide class imbalance |
| `eval_macro_f1` | Mean F1 across all six labels | Primary selection metric; each class has equal weight |
| `metrics.per_class` | Precision, recall, F1, and support per label | Reveals which classes fail |
| `metrics.confusion_matrix` | Gold labels in rows, predictions in columns | Ordering is in `label_order` |
| `eval_runtime` | Time for tokenization and inference | Excludes loading; not total app latency |
| `eval_samples_per_second` | Examples / evaluation runtime | Throughput, not predictive quality |
| `eval_steps_per_second` | Batches / evaluation runtime | Depends on batch size and hardware |

Reports are saved to `artifacts/evaluation-validation.json` or
`artifacts/evaluation-test.json`. Use `--output` for a separate filename.
Per-example predictions include event IDs and gold/predicted labels for error review.
The evaluator does not claim an epoch number: it evaluates saved weights. The
selected training epoch is recorded in the training report's `best_epoch`.

Repeated evaluation is useful for checking reproducibility, but must not turn the
test set into a tuning set. A six-example smoke test cannot establish superiority,
calibration, or production readiness. The current independent rerun verifies the
saved checkpoint; it is not a new independent dataset.

## 6. Interpret the actual current results

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| TF-IDF + logistic regression | 0.5000 | 0.4444 |
| DistilBERT, one epoch | 0.1667 | 0.0476 |

These values come from the saved experiment reports, not the screenshot. The
Transformer predicted the infrastructure class for all six test examples. It
underperformed the baseline. The result proves the training/evaluation pipeline
runs; it does not meet the project's real-data performance target. Do not describe
this checkpoint as an accurate or validated news classifier.

The recorded training loss (1.7967) is not evaluation loss. The original training
report's runtime (11.23 seconds) includes loading, training, validation, saving and
test evaluation. It must not be labeled `eval_runtime`.

## 7. Evaluate extraction and the business application separately

Classifier accuracy does not validate Qwen's generated impact analysis. Create a
separate human-reviewed evaluation set and record:

- Primary-event type and location correctness against adjudicated gold labels.
- Source sentence citation validity and semantic support for each extracted claim.
- Unsupported-claim rate and whether historical facts are misreported as current.
- Whether each business hypothesis is conditional, relevant, and distinguishable
  from a reported fact; use a written rubric and two reviewers on a subset.
- JSON completion rate, empty/failed extraction rate, and p50/p95 end-to-end latency
  over representative URLs, PDFs and pasted text, with hardware and warm/cold status.
- User triage time and decision quality compared with a manual workflow.

Report sample counts and failures. JSON validation and exact citations are checks,
not proof that the claims are true. Manual radius scenarios are assumptions and
must not be evaluated or presented as learned impact probabilities.

## 8. Publish and test the uploaded model

After reviewing a training run and its evaluation:

```bash
python publish_model.py \
  --repo-id zhengzhihust/redwood-public-analysis-event-classifier \
  --update-existing
```

This uploads a private revision with weights, tokenizer, model card and metrics,
then downloads that exact revision, checks SHA256 hashes, and compares pipeline
outputs with the local checkpoint. Publication does not train the model, and a
successful reload does not establish predictive quality. See
`artifacts/hub-publication.json` for the already verified smoke checkpoint.
