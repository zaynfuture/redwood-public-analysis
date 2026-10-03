# Validation record — 2026-10-03

22 automated tests passed, including Streamlit demo flow, dynamic radius update,
source-grounding checks, invalid coordinates, antimeridian geometry, URL rejection,
redirect validation, download limits, mocked PDF text extraction, and split leakage.
The PDF check mocks the parser; it is not an end-to-end arbitrary-PDF guarantee.

## Actual training runs

Both models used the same 48 original fictional examples (36/6/6 split).

| Experiment | Test macro-F1 | Test accuracy | Runtime (s) |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 0.444 | 0.500 | 0.61 |
| DistilBERT, one full fine-tuning epoch | 0.048 | 0.167 | 11.23 |

The Transformer updated 66,958,086 trainable parameters.
It underperformed the baseline and predicted one class on this tiny held-out set.
This is a successful pipeline smoke test and an unsuccessful quality result.
No real-news accuracy or business-value target is claimed achieved.

## Live inference and publication

Qwen3-0.6B extracted a schema-valid event from one fictional report in
2.9 seconds after loading. This is not a latency benchmark
or extraction-quality estimate. The model missed reported scope, returning Unknown;
its generated title was generic. Location and event type were correct on this sample.
Qwen2.5-0.5B failed schema/quotation checks during development, so it is not the default.

Private model repository: https://huggingface.co/zhengzhihust/redwood-public-analysis-event-classifier
Verified revision: `582eb8ed710670bfc200751b58d181c861adb658`.
All 6 staged files matched their downloaded SHA256 hashes;
downloaded and local pipeline outputs matched on two fictional inference samples.
Uploaded checkpoint and model card explicitly identify synthetic smoke training.

## Remaining final-project evidence

Real licensed news corpus, adjudicated annotations, held-out model comparison,
extraction/location/impact evaluation, representative URL/PDF integration testing,
resource/latency benchmark, and measured business triage benefit remain outstanding.
The screenshot's rubric section A is unavailable.

## BBC URL regression repair
The supplied BBC URL (ck1wxx8n2x3zo) now completes in the actual Streamlit browser
workflow with one natural-hazard event in Bangkok and a verified source citation.
25 automated tests pass. The repair adds JSON-schema token filtering, restricts
source IDs, handles repeated place mentions, and removes article UI boilerplate.
The lightweight model still requires semantic review; format validity is not an
accuracy guarantee. Details: artifacts/bbc-regression.json.
