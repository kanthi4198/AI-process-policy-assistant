---
title: Model Release Checklist
doc_type: sop
domain: llm
owner: AI Platform Team
effective_date: 2025-12-01
version: v1.0
sensitivity: internal_sample
---

# Model Release Checklist (v1.0)

Before releasing an LLM-enabled feature:
1. Define intended use and known failure modes.
2. Add refusal behavior for out-of-scope questions.
3. Implement citation-first responses for policy Q&A.
4. Evaluate on a fixed question set and track regressions.
5. Security review if any Restricted data could be processed.
6. Add monitoring for:
   - hallucination indicators (no-source answers)
   - latency
   - repeated user retries (confusion signal)
