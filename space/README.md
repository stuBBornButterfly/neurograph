---
title: NeuroGraph assistant
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: The university student assistant checked against a knowledge graph
---

# NeuroGraph assistant

A student-support assistant for the university that answers from public university pages and from the
NeuroGraph knowledge graph (an EEG depression study plus published EEG research), and checks every answer before
it is shown.

- **Pipeline:** safety check (distress or crisis → fixed referral to university counselling and 999) → router → search over
  100 PII-redacted university passages (BGE embeddings, Qdrant) or a read-only graph query → Qwen2.5-3B fine-tuned with
  QLoRA for university questions, the base Qwen2.5-3B for research questions (the adapter was trained on university passages only;
  both Q4_K_M GGUF, llama.cpp on CPU) → verifier (every number must appear in the sources; each sentence is
  checked by an NLI model).
- **Speed:** on the free 2-vCPU hardware an university answer takes about 100 s (the prompt holds ~1,700 tokens of passages) and a research answer 40-50 s; first start about 2 minutes plus the model downloads.
- **Monitoring:** every answer and 👍/👎 vote is logged (questions PII-redacted, distress messages not stored);
  the sidebar shows counts, verification rate, latency and feedback.
- Code, evaluation and limitations: https://github.com/stuBBornButterfly/neurograph

> Research prototype. Not a diagnostic tool and not a counsellor. In an emergency in Bangladesh call 999.

## Settings (Space → Settings → Variables and secrets)

| name | kind | value |
|---|---|---|
| `GGUF_REPO` | variable | the model repo holding the fine-tuned GGUF, e.g. `<user>/neurograph-qwen2.5-3b-gguf` (the base GGUF comes from `Qwen/Qwen2.5-3B-Instruct-GGUF`) |
| `HF_TOKEN` | secret | only if the model repo is private, or for `LOG_DATASET` |
| `LOG_DATASET` | variable (optional) | a private dataset repo, e.g. `<user>/neurograph-logs`; the log is pushed there every 10 min |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | secrets (optional) | query the live graph instead of the bundled snapshot |
