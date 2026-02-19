# Enterprise policy assistant
A calm citation-first AI system to understand policies without guessing

## Project structure

- **`src/`** — CLI entry points and the `policy_assistant` package:
  - **`policy_assistant/data/`** — document loaders (PDF) and parent-child chunking
  - **`policy_assistant/embeddings/`** — local Hugging Face / Sentence-Transformers embeddings
  - **`policy_assistant/store/`** — FAISS vector store build and load
  - **`policy_assistant/retrieval/`** — retrieval logic and algorithm backends (flat, hnsw, ivf, lsh, hybrid)
  - **`policy_assistant/eval/`** — evaluation scripts (embed, chunk, retrieval) and shared eval types
- **`data/docs/`** — policy PDFs (input for ingest and evals)
- **`eval/`** — `eval_questions.json` and optional `best_retrieval.json`
- **`vector_store/`** — output of ingest (FAISS index + parent docstore)

## What problem am I trying to solve?

In large organizations, policies are everywhere — data access rules, security guidelines, incident procedures, AI usage policies, release checklists.

Yet in day-to-day work, people still struggle with questions like:

* “Which policy applies to my situation?”
* “Am I allowed to do this, or do I need approval?”
* “Who should I escalate this to, and when?”

The problem isn’t that policies don’t exist.
The problem is that they are hard to interpret when you actually need them.

Policies are:

* Written in formal language
* Spread across multiple documents and formats
* Full of conditions, exceptions, and cross-references

As a result, people either:

* Spend too much time searching and second-guessing
* Ask the same questions repeatedly to experts
* Or worse — make assumptions and hope they’re right

That’s risky, slow, and frustrating.

This project focuses on one core issue:

How do we make policy knowledge accessible and reliable at the moment decisions are made?

## Why is AI required for this?

This is not a simple search problem.

People don’t think in document titles or keywords.
They ask situational questions:

* “Can I use this dataset for my project?”
* “Does this count as a reportable incident?”
* “Are we allowed to use an external LLM here?”

Answering these requires:

* Understanding natural language
* Interpreting intent and context
* Connecting information across multiple documents

Traditional systems struggle here because they treat policies as static text.

AI, specifically language models combined with retrieval is well-suited to:

* Understand how people naturally phrase questions
* Read and summarize policy language
* Combine relevant sections from different documents into a coherent explanation

That said, AI alone is not trustworthy in this domain.
Left unchecked, it can sound confident while being wrong.

This project uses AI only where it adds value and surrounds it with guardrails.

## What is the alternative without AI?

### 1. Document portals and keyword search

SharePoint, Confluence, internal wikis.

They work if:
* You already know which document to read
* You already understand the policy structure

They break down when:
* You’re unsure which policy applies
* The answer is spread across documents
* You’re under time pressure

### 2. Rule-based workflows and decision trees

"If X, then follow policy Y"

These look good on paper, but in reality:
* They are expensive to maintain
* They don’t adapt well when policies change
* They fail on edge cases and ambiguous situations

### 3. Human experts and governance teams

This is often the most reliable option.

But it doesn't scale:
* Experts become bottlenecks
* The same questions get asked repeatedly
* Response times increase as organizations grow

## Why is AI better than these alternatives?

AI is not better because it is “smarter”.

It is better when used carefully because it reduces friction without removing accountability.

This system is designed to:
* Let users ask questions in their own words
* Pull together relevant policy sections automatically
* Explain answers in plain language
* Always show where the answer comes from

Just as importantly, it is designed to fail safely:

* If the documents don’t support an answer, the system says so
* If policies conflict, both are shown
* If confidence is low, that is made explicit

Instead of replacing experts, the assistant helps ensure that simple, low-risk questions are handled efficiently,and complex or high-risk cases are escalated earlier and with better context.

## What this project is (and is not)

This is not a generic chatbot.

It is an experiment in building an enterprise-ready AI assistant that prioritizes:

* Trust over fluency
* Evidence over confidence
* Transparency over convenience

The focus is on system design:

* Retrieval before generation
* Citations before conclusions
* Clear refusal behavior when information is missing

## Design principles I followed

A few simple ideas guided this project:
* AI should assist decisions and not make them:
The assistant explains the policy, while the responsibility stays with the user.

* Uncertainty should be visible:
Saying "I don't know" is better than being confidently wrong.

* Trust comes from evidence:
 Every answer is tied-back to a specific policy text

* Enterprise AI is a system, not a model:
Retrieval, filtering, reranking, and prompting matter as much as the LLm itself.

## Why I built this?

I'm interested in AI systems that actually work inside organizations, not just in demos.

Many AI failures don't come from weak models, they come from:
* Missing guardrails
* Poor integration with real workflows
* Overconfidence in fluent outputs

This project is my attempt to explore what responsible, practical AI looks like when accuracy, explainability, and governance matter.

## Retrieval evaluation

The project includes a **retrieval evaluation pipeline** that compares multiple retrieval algorithms on your policy corpus so you can use the best one in production.

### Algorithms evaluated

| Algorithm | Description |
|-----------|-------------|
| **flat** | Exact k-NN (FAISS flat index) — baseline |
| **hnsw** | Approximate k-NN using HNSW graph (FAISS) |
| **ivf** | Inverted file index (FAISS IVF) |
| **lsh** | Locality-sensitive hashing (FAISS LSH) |
| **hybrid** | Dense (flat) + BM25 with RRF fusion |

### Metrics

- **Hit@1** — fraction of questions where the first retrieved doc is relevant
- **Hit@k** — fraction where at least one relevant doc is in the top k
- **MRR** — mean reciprocal rank of the first relevant doc
- **Precision@k** — average over questions of (relevant docs in top-k) / k
- **Recall@k** — average over questions of (relevant docs in top-k) / (total relevant docs in corpus for that question)

### How to run

From the project root (chunking defaults match ingest):

```bash
# Evaluate all algorithms (flat, hnsw, ivf, lsh, hybrid)
python -m policy_assistant.eval.retrieval_eval --docs_dir data/docs --eval_questions_file eval/eval_questions.json

# Evaluate only some algorithms and save the best for production
python -m policy_assistant.eval.retrieval_eval --algorithms flat hnsw hybrid --top_k 5 --save_best eval/best_retrieval.json
```

The script prints a leaderboard and recommends an algorithm for production (e.g. `Recommended for production: 'hnsw'`). Use that algorithm when building your vector store so retrieval uses the same index type; the current ingest builds a flat FAISS index by default.