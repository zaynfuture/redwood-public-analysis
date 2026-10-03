# Local research context and external references

Local knowledge health checked 2026-10-03: healthy; 6 collections, 54 documents,
1,529 passages. Search: news event geographic impact NLP transformer fine tuning
Streamlit. Context bundle capped at 1,500 estimated tokens; returned 1,322.
Observation ID: search-9e0028e8-b7bf-4107-a5eb-5a9503767be7.

Preserved evidence identifiers:
- evidence-099851f4a85905d12a8d8214e75ad919 /
  passage-4a3aeb8bf696a24ce94de3f6fb5495b2: personal embedding-model notes;
  general domain adaptation context, not current API authority.
- evidence-49259f9766b2cf7ba03b20ed32f7d71e /
  passage-bc3fdceafaaac715dbafdbba46770136: domain-specific RAG notes.
- evidence-b23aa28c8fdfc6129d34bbbf852991c9 /
  passage-06a2d258bf7c7f30e6547c65f9d37639: historical fine-tuning notes;
  old provider/model claims were not adopted as implementation facts.
- evidence-31eea0c5cf6b8d790a207d380d9c54d7 /
  passage-31eea0c5cf6b8d790a207d380d9c54d7: document NLP integration heading;
  insufficient detail to support specific architecture claims.

Gaps identified before external lookup: no task-specific labeled news data, no
geographic-impact benchmark, no Streamlit implementation guidance; bundle truncated.

Primary external references consulted after gap identification:
- https://huggingface.co/docs/transformers/en/main_classes/trainer
- https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/sequence_classification.md
- https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart
- https://huggingface.co/docs/transformers/main_classes/pipelines

Implementation uses explicit PyTorch training for transparency and Plotly's globe
projection for the world view. These are project design decisions, not claims
made by the retrieved local evidence. No personal note content is sent to the LLM.

Additional primary references for the requested Hugging Face workflow:
- https://huggingface.co/Qwen/Qwen3-0.6B
- https://huggingface.co/docs/huggingface_hub/en/guides/upload
- https://huggingface.co/docs/huggingface_hub/en/guides/repository

## BBC URL repair — 2026-10-03
Knowledge health remained healthy. A new context bundle used 552/1,500 estimated
tokens, observation search-0c82e555-c0e5-44c4-9910-eb911ebf688f. Retrieved evidence
evidence-956ff4d30affbf15549402c7dcf4139b and
evidence-4604da0952de975c15ea2e27c7a85882 concerned wiki tooling and were not
relevant to constrained decoding. This gap was identified before external lookup.
Primary implementation reference: https://github.com/noamgat/lm-format-enforcer .
The adapter uses the public core API because version 0.11.3's bundled HF adapter
imports a tokenizer symbol removed from the installed Transformers 5 release.

## Training/evaluation documentation
Health checked; bounded context 601/1,500 estimated tokens; observation
search-9074ef24-98d2-4ae0-bdf4-ff191d6d8666. Retrieved identifiers:
evidence-49259f9766b2cf7ba03b20ed32f7d71e (generic fine-tuning note),
evidence-8333938817302903ef0f1bb9fb76e048 (unrelated option-model validation).
Gap: neither documents this classifier's procedure or metrics. Documentation is
grounded in train.py and saved experiment reports; no new external claims needed.
