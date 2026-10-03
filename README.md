# Redwood Public Analysis

An English-language Streamlit business application for turning public news into
source-grounded events and location-specific exposure scenarios. Built with a
specification-driven workflow and Hugging Face models.

## Run locally

```bash
cd /Users/zaynfuture/Developments/redwood-public-analysis
source .venv/bin/activate
streamlit run app.py
```

For a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
streamlit run app.py
```

Use **Demo scenario → Analyze source → confirm event coordinates** for a no-model
walkthrough. Live extraction focuses on the primary headline event and accepts news HTML/PDF URLs, uploaded text-based PDFs,
and pasted reports. Select a Hugging Face model in the sidebar. Downloads are enabled by default. Offline mode uses only weights already cached
on the server; a laptop cache is not transferred to Streamlit Cloud.

## Hugging Face models and pipelines

- Extraction: `Qwen/Qwen3-0.6B` is the lightweight default;
  `Qwen/Qwen2.5-3B-Instruct` is configurable for quality experiments. Neither is
  claimed to meet the accuracy targets without validation.
- Classification: actual fine-tuning of `distilbert-base-uncased` on six event labels.
- Inference: `pipeline("text-generation")` and `pipeline("text-classification")`.
- Streamlit caches pipeline resources (maximum two entries). Models are not loaded
  for the fictional demo. Classifier supports batches; choose batch size by measured
  memory/latency, since batching is not always faster. CUDA/MPS/CPU selection is
  automatic; CUDA uses float16, other backends use float32 for compatibility.
- Generation is bounded and deterministic (`do_sample=False`), followed by schema
  and source-sentence validation. JSON-schema token filtering constrains output during generation, including
  source sentence references. Post-generation evidence validation remains mandatory.
- Models use `trust_remote_code=False`. No hosted inference service or API key is
  needed. Model downloads and news URL retrieval require network access.

## Train and fine-tune

See [How to fine-tune and evaluate](MODEL_WORKFLOW.md) for the complete procedure,
actual hyperparameters, standalone evaluation, metric definitions, and limitations.


```bash
python train.py --mode baseline
python train.py --mode transformer --epochs 3 --local-files-only
python -m unittest discover -s tests -v
```

Remove `--local-files-only` to download pretrained weights if not cached.
`data/smoke.csv` contains 48 original fictional examples: 36 train / 6 validation /
6 test. It is a pipeline smoke fixture, **not a real news benchmark**. Both commands
train models; the Transformer command updates the full pretrained encoder and new
classification head. Validation macro-F1 selects the checkpoint, then test metrics
are computed. Saved outputs are in `artifacts/`.

For a real experiment, supply an annotated CSV matching the contract in
`specs/003-model-and-data.md`, then pass `--data path/to/news.csv` to both commands.
Keep the test set fixed. Do not tune hyperparameters against the reported test
results. Real data collection, annotation, and final evaluation are still required.
The classifier fine-tuning does not fine-tune the generative extraction model.

## User workflow

1. Add a report and extract events; inspect the exact supporting quotation.
2. Choose the event and inspect reported scope versus possible business effects.
3. Enter your business location (offline city choices or arbitrary coordinates).
4. Confirm event coordinates, choose a business dependency, and vary the radius.
5. Pan/zoom the world map and export source evidence and scenario assumptions.

The slider updates a hypothetical footprint, not event movement over time.
Indirect business impacts may exist far beyond a geographic radius. A point
outside the circle is not evidence that the location is safe or unaffected.
The map requires Plotly geographic assets to load in the browser.

## Specification-driven structure

```text
specs/001-product.md             Requirements, acceptance criteria, rubric scope
specs/002-design.md              Architecture and traceability
specs/003-model-and-data.md      Training, annotation, evaluation design
specs/004-evidence-context.md    Bounded local research and evidence identifiers
app.py                          English Streamlit UI and model cache
geoimpact/core.py               Contracts and geodesic scenarios
geoimpact/ingest.py             HTML/PDF ingestion and source provenance
geoimpact/extract.py            Hugging Face inference pipelines
train.py                        Baseline and deep-learning experiments
data/smoke.csv                  Original fictional smoke fixture
tests/                          Acceptance and regression checks
artifacts/                      Saved models and measured experiment reports
```

## Limitations and next research milestone

The application is a working MVP, not a completed empirical final project.
Small instruction models can produce invalid JSON or misinterpret news; quote
matching validates copied text, not the truth or semantic entailment of every
claim. Location disambiguation currently requires user confirmation. Scanned PDFs
need separate OCR. News sources may block retrieval or include irrelevant text.
Input is capped at 24,000 characters and a model token budget; warnings disclose
truncation. The prototype assumes one local user. Read deployment boundaries in
`specs/002-design.md` before exposing it to other users.

Next milestone: annotate licensed news excerpts, run baseline/deep-model comparison,
review extraction/location/impact correctness independently, measure latency and
business triage time, and prepare the presentation using `DEMO.md`.

## References

- [Hugging Face pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Sequence classification](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md)

The provided rubric screenshot covers sections B–F; section A was not supplied.

## Publish and test the uploaded model

The initial private repository is
[zhengzhihust/redwood-public-analysis-event-classifier](https://huggingface.co/zhengzhihust/redwood-public-analysis-event-classifier).
It currently contains the explicitly labeled synthetic smoke checkpoint.
Publication verification is recorded in `artifacts/hub-publication.json`.

After a new, reviewed training run:

```bash
hf auth login
python publish_model.py \
  --repo-id zhengzhihust/redwood-public-analysis-event-classifier \
  --update-existing
```

Omit `--update-existing` when creating a new repository. Repositories default to
private; `--public` is explicit and never changes an existing repo's visibility.
The script stages only weights, tokenizer, config, model card, and sanitized
metrics. It uploads, downloads the exact commit into a fresh directory, verifies
SHA256 hashes, and compares downloaded and local pipeline predictions.
`--prepare-only` creates a local reviewable package without making network writes.
Never put a Hugging Face token in a notebook, source file, or Git repository.

Source auto-reload is disabled to avoid Streamlit probing Transformers lazy modules.
Restart Streamlit after editing Python code.

## Streamlit Cloud first-run model loading
Leave **Use downloaded models only** unchecked (the default). On the first analysis,
the server downloads the selected public model, then caches it. Rebuilds may discard
that cache. Do not set HF_HUB_OFFLINE on a fresh deployment. No token is required
for the public Qwen extractor; the separately published private classifier needs
its own authorized credentials and is not bundled in Git. If downloads fail with
offline mode off, inspect Cloud logs for outbound-network or memory limits.
