# Architecture and implementation plan

Spec-driven flow: product requirements → contracts → implementation → acceptance
checks → evaluation report. Update specs before changing externally visible behavior.

```text
URL / PDF / text → bounded ingestion → source record
                                      ↓
                       local Hugging Face Instruct model → validated event + evidence
                                      ↓
                 user-confirmed event and business coordinates
                                      ↓
                     geodesic scenario → map + export

Labeled data → baseline / Transformer fine-tuning → metrics + saved model
                                                       ↓
                                   optional event-type second opinion
```

`geoimpact/core.py` defines the shared contracts and deterministic geography.
`ingest.py` handles untrusted media. `extract.py` integrates Hugging Face pipelines and the trained
classifier. `app.py` owns presentation and session state. `train.py` owns experiments.
The package name is internal; the product and directory are redwood-public-analysis.

LLM event extraction and supervised classification are separate tasks. Fine-tuning
a classifier is actual deep-learning training, but does not fine-tune the extractor.
Baseline and fine-tuned predictions must not be presented as independent factual
verification. LLM coordinates are deliberately not accepted: users confirm them.

## Deployment boundaries
Local single-user application by default. URL validation checks DNS and every
redirect and blocks nonpublic addresses; requests may resolve DNS again, so this
is not a complete DNS-rebinding defense. Before public deployment enforce outbound
network rules denying internal ranges at the network/proxy layer, isolate PDF
parsing, add authentication, request quotas, monitoring, and concurrency limits.
Do not expose the prototype publicly without these controls.

## Acceptance mapping
SPEC-01: ingestion and URL tests; text-only PDF end-to-end fixture.
SPEC-02: schema and fabricated-quotation rejection tests; mocked LLM contract test.
SPEC-03/04: coordinate validation, antimeridian and geodesic tests; app map smoke.
SPEC-05: split validator; saved experiment reports; real validation remains open.
SPEC-06: exported source and scenario fields; no probabilities inferred from distance.
SPEC-07: Streamlit AppTest initial/demo flow and meaningful error-path tests.

## Pipeline performance design
Use cached Hugging Face text-generation and text-classification pipelines.
Batch classifier inputs, bound context/generation lengths, select CUDA/MPS/CPU,
and avoid model work during map-only interaction. Benchmark rather than assume
batching gains. Larger generators require independent latency/memory evaluation.

The generator selects source sentence IDs. Python copies the selected original
sentences into evidence fields and checks that the named place occurs there.
This prevents invented quotation text, but does not establish semantic entailment.

## BBC extraction regression fix
Generation must use a JSON-schema token filter through the Hugging Face pipeline.
The model chooses only existing source sentence IDs and returns at most one
primary event. Paragraph extraction excludes page controls when HTML paragraphs
are available. Schema and evidence checks remain mandatory after decoding; a
valid JSON object alone does not establish factual correctness.

This release focuses on the headline event; related consequences and historical
comparisons are not separate events. Multi-event extraction remains a future extension.
