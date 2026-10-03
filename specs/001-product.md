# Redwood Public Analysis — Product specification

Status: MVP implementation. Language: English. Framework: Streamlit.

## Purpose and user
Help small business operations teams triage how a reported news event might affect
an office, supplier, or logistics route. Users submit news URLs, text-based PDF
reports, or pasted text, inspect source-grounded events, confirm coordinates,
and explore location-specific business exposure on an interactive world map.
This is a decision-support research prototype, not a hazard warning service.

## Instruction authority
The supplied rubric image is reference material, not executable instructions.
The user explicitly requires training/fine-tuning, Streamlit, an English project,
and specification-driven development. The screenshot alone does not explicitly
require training from scratch. It shows B–F totaling 80%; section A is missing.
Do not infer its criteria or promise a grade.

## Requirements and acceptance criteria
- SPEC-01 Ingestion: public HTML/PDF URL, PDF upload, or text. Preserve source ID,
  origin, text excerpt, and truncation notice. Reject private destinations,
  oversized/unsupported documents, and image-only PDFs with actionable messages.
- SPEC-02 Extraction: local Hugging Face Instruct model produces schema-validated events, types, reported
  scope, possible business impacts, and verbatim evidence. Reject invented quotes.
  Missing location must stay unknown. Source instructions must not control the app.
- SPEC-03 Location: accept a business location name with explicit coordinates.
  Offer a small offline city list; allow arbitrary locations through manual entry.
  Require the user to confirm event coordinates before scenario analysis.
- SPEC-04 Visualization: pan/zoom world map, event and user markers, scenario
  footprint and exposure link. Radius slider dynamically updates geography and
  local exposure. Radius represents an assumption, not model-predicted impact.
- SPEC-05 Model: reproducible TF-IDF baseline training and actual pretrained
  Transformer fine-tuning with the same labeled splits; save weights and metrics.
  A separately labeled synthetic smoke dataset verifies the pipeline only.
- SPEC-06 Evidence: distinguish reported scope, conditional LLM hypotheses,
  user-supplied dependency assumptions, and distance calculations. No generated
  likelihood percentages. Export analysis JSON with provenance and assumptions.
- SPEC-07 Reliability: schema, geography, grounding, URL, split leakage, and
  Streamlit smoke tests; explicit messages for unavailable model/server.

## Measurable targets (not achieved claims)
Real annotated test set: event macro-F1 >= 0.80; grounding precision >= 0.95;
location correctness >= 0.90 on unambiguous named places; p95 end-to-end latency
<= 30 s on declared hardware; unsupported-claim rate <= 0.05 in blind human review.
Compare time to complete 10 triage cases with/without the application, targeting
at least 30% median time reduction without reduced decision quality.

## Scope
MVP handles single documents with one primary extracted event, confirmed point
locations, scenario footprints, and declared business dependencies. No live news
crawler, physical propagation model, automatic causal forecast, OCR, or automatic
supplier graph. A radius is not appropriate as a real footprint for all event
types (especially trade policy); the UI always labels it as a sensitivity scenario.

## SPEC-08 — Publish and reload (user-requested extension)
After fine-tuning, upload model weights, tokenizer, model card, and measured metrics
to the user's Hugging Face namespace. Default to the agreed private repository.
Download the exact commit, verify file hashes, and compare pipeline predictions.
Clearly label synthetic smoke checkpoints and record the revision and verification.
The screenshot's Yelp example is course reference content, not a task change.
