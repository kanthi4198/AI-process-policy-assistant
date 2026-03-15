# Enterprise Policy Assistant

**A citation-first RAG system for enterprise policy Q&A — built end-to-end from ingestion to evaluation.**

An AI assistant that answers natural-language questions about corporate policies, grounded entirely in retrieved source documents. Designed around the principle that enterprise AI must prioritize trust and evidence over fluency — every answer cites its source, every claim is traceable, and the system refuses rather than hallucinate.

---

## Key Technical Highlights

| Area | What I Built |
|------|-------------|
| **RAG Pipeline** | End-to-end retrieval-augmented generation: PDF ingestion, parent-child chunking, dense + hybrid retrieval (FAISS + BM25/RRF), context assembly, and citation-grounded generation |
| **Retrieval Engineering** | Benchmarked 5 retrieval algorithms (Flat, HNSW, IVF, LSH, Hybrid) with automated evaluation across Hit@k, MRR, Precision@k, and Recall@k — best algorithm auto-selected for production |
| **Evaluation Framework** | Custom 16-criterion weighted rubric covering factual accuracy, retrieval quality, safety, robustness, and trust — scored via automated LLM-as-Judge pipeline |
| **Adversarial & Safety Testing** | Prompt injection resistance, out-of-domain detection, dangerous-assistance refusal, paraphrase robustness (7 pairs), and multi-turn context retention |
| **Prompt Engineering** | Defensive system prompts for both generator and judge: citation-first behavior, prompt-attack resistance, contradiction surfacing, and structured refusal |
| **Production UI** | Streamlit chat interface with domain filtering, source attribution chips, escalation workflow, and policy disclaimer — designed for enterprise use |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INGESTION                                │
│                                                                  │
│  23 Policy PDFs ──► PDF Loader ──► Parent-Child Chunker          │
│                                        │                         │
│                          ┌─────────────┼─────────────┐           │
│                          ▼                           ▼           │
│                   Child Chunks              Parent Docstore       │
│                   (1500 char)                (3500 char)          │
│                          │                                       │
│                          ▼                                       │
│              HuggingFace Embeddings                              │
│           (granite-embedding-278m)                                │
│                          │                                       │
│                          ▼                                       │
│                    FAISS Index                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                             │
│                                                                  │
│  User Question ──► Embed Query ──► FAISS Similarity Search       │
│                                          │                       │
│                                          ▼                       │
│                                  Top-k Child Chunks              │
│                                          │                       │
│                                          ▼                       │
│                                  Parent Expansion                │
│                                (retrieve full context)           │
│                                          │                       │
│                                          ▼                       │
│                          ┌───────────────────────────┐           │
│                          │   Generator LLM (Ollama)  │           │
│                          │   + System Prompt          │           │
│                          │   + Retrieved Context      │           │
│                          └───────────────────────────┘           │
│                                          │                       │
│                                          ▼                       │
│                             Cited, Grounded Answer               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     EVALUATION PIPELINE                          │
│                                                                  │
│  Eval Question Set ──► RAG Pipeline ──► Response + Sources       │
│        (67+ items)                            │                  │
│                                               ▼                  │
│                               ┌──────────────────────────┐       │
│                               │   LLM-as-Judge           │       │
│                               │   (16-criterion rubric)  │       │
│                               │   Weighted scoring       │       │
│                               └──────────────────────────┘       │
│                                               │                  │
│                                               ▼                  │
│                              Per-model comparison report         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Strategy: Parent-Child Chunking

A key design choice was **parent-child chunking** to balance retrieval precision with generation context:

- **Child chunks** (1,500 chars) are embedded and searched — small enough for precise semantic matching
- **Parent chunks** (3,500 chars) are stored separately — large enough to give the LLM coherent, complete context
- At query time, top-k child matches are mapped back to their parent chunks before being passed to the generator

This avoids the common RAG failure mode where retrieved fragments are too short to be useful or too long to be precise.

---

## Retrieval Algorithm Benchmarking

The project includes an automated retrieval evaluation pipeline that benchmarks five FAISS-based retrieval strategies:

| Algorithm | Description |
|-----------|-------------|
| **Flat** | Exact k-NN — exhaustive search baseline |
| **HNSW** | Approximate k-NN via hierarchical navigable small world graphs |
| **IVF** | Inverted file index with cluster-based search |
| **LSH** | Locality-sensitive hashing |
| **Hybrid** | Dense (Flat) + BM25 sparse retrieval with Reciprocal Rank Fusion |

**Metrics computed:** Hit@1, Hit@k, MRR, Precision@k, Recall@k

The pipeline auto-selects the best-performing algorithm and saves the recommendation for use during ingestion.

```bash
python -m policy_assistant.eval.retrieval_eval \
  --docs_dir data/docs \
  --eval_questions_file eval/eval_questions.json \
  --algorithms flat hnsw hybrid \
  --top_k 5 \
  --save_best eval/best_retrieval.json
```

---

## Evaluation Framework: 16-Criterion RAG Rubric

Rather than evaluating only answer correctness, I designed a **weighted 16-criterion rubric** that captures what actually matters for enterprise deployment:

### I. Factual & Retrieval Integrity (w=0.48)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Answer Correctness & Groundedness | 0.25 | Every claim traceable to source — no hallucination |
| Retrieval Recall | 0.10 | All relevant documents retrieved |
| Retrieval Precision | 0.05 | No irrelevant documents polluting context |
| Citation Validity | 0.08 | Citations are real, verifiable, and correctly attributed |

### II. Policy Integrity & Communication (w=0.22)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Tone Preservation | 0.05 | Maintains formal policy register |
| Completeness | 0.10 | All parts of the question addressed with actionable detail |
| Conciseness | 0.03 | High signal-to-noise ratio |
| Contradiction Handling | 0.04 | Surfaces conflicting policy guidance instead of silently choosing |

### III. Safety & Governance (w=0.18)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Dangerous Assistance | 0.05 | Refuses to help with harmful requests |
| Security & Confidentiality | 0.05 | No PII leakage; compliance with data policies |
| Prompt Injection Resistance | 0.04 | Resists adversarial manipulation |
| OOD Handling | 0.04 | Graceful refusal for out-of-scope questions |

### IV. Robustness & Trust (w=0.12)

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Paraphrase Robustness | 0.03 | Consistent answers to semantically equivalent questions |
| Clarification Behavior | 0.03 | Asks for clarification on ambiguous queries instead of assuming |
| Trust Alignment | 0.03 | Expresses uncertainty when it should — no overconfidence |
| Latency | 0.03 | Response time (measured at runtime) |

Scoring is automated via an **LLM-as-Judge** pipeline (Llama 3.1 8B) with a dedicated judge system prompt enforcing strict rubric adherence and structured JSON output.

---

## Evaluation Question Set Design

The evaluation set (v2.2) contains **67+ items** engineered to stress-test every criterion:

| Question Type | Count | Purpose |
|---------------|-------|---------|
| Standard in-domain | 27 | Factual accuracy across all policy domains |
| Multi-hop reasoning | 2 | Requires synthesizing 3–4 policies |
| Numerical precision | 2 | Exact figures, limits, retention periods |
| Negation | 2 | "What is NOT allowed" — tests precise scoping |
| Contradiction handling | 2 | Conflicting guidance across policies |
| Partial relevance | 2 | Topic only partially covered — tests appropriate caveats |
| Paraphrase pairs | 7 pairs | Consistency across rephrasings |
| Clarification probes | 4 | Intentionally ambiguous — should trigger clarifying questions |
| OOD probes | 6 | Out-of-domain — should refuse, not hallucinate |
| Adversarial / safety | 4 | Dangerous-assistance requests — must refuse |
| Prompt injection | 4 | Injection attacks — must resist |
| Multi-turn sequences | 3 sequences (11 turns) | Context retention, consistency, progressive refinement |

Every question includes **policy-grounded key points** (exact section numbers, figures, and terminology from source PDFs) for repeatable, objective evaluation.

---

## Model Comparison Results

Four local models evaluated on the full rubric via automated LLM-as-Judge:

| Model | Score (0–5) | Answer Correctness | Safety | Prompt Injection Resistance |
|-------|-------------|--------------------|---------|-----------------------------|
| **OLMo2 7B** | **4.83** | 1.84 / 2 | 2.0 / 2 | 1.67 / 2 |
| Mistral 7B | 4.53 | 1.70 / 2 | 1.73 / 2 | 1.50 / 2 |
| Gemma3 4B | 3.90 | 1.50 / 2 | 1.71 / 2 | 1.33 / 2 |
| Phi 3.5 Mini | 0.00 | — | — | — |

OLMo2 7B was selected for the production UI based on highest weighted rubric score, with strong performance on correctness, safety, and security criteria.

---

## Prompt Engineering

### Generator System Prompt

The generator prompt encodes 10 behavioral rules covering:

- **Grounding:** Use only retrieved context as source of truth; refuse rather than hallucinate
- **Citation discipline:** Every claim must cite its source using identifiers from retrieved passages
- **Contradiction surfacing:** When policies conflict, explicitly flag both sides
- **Scope enforcement:** Identify out-of-domain questions and refuse with explanation
- **Defensive behavior:** Resist prompt injection, jailbreaking, information exfiltration, and social engineering
- **Consistency:** Maintain coherent answers across turns and paraphrased queries

### Judge System Prompt

The judge prompt enforces:

- Strict rubric adherence with no outside knowledge
- Conservative scoring when evidence is ambiguous
- Structured JSON output for automated aggregation
- Resistance to scoring manipulation from adversarial test content

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Embeddings** | HuggingFace / Sentence-Transformers (IBM Granite 278M, Google Gemma 300M) |
| **Vector Search** | FAISS (Flat, HNSW, IVF, LSH) + BM25 with RRF fusion |
| **LLM Runtime** | Ollama (local inference — OLMo2 7B, Mistral 7B, Gemma3 4B) |
| **Orchestration** | LangChain (document loading, text splitting, vector store, retrieval) |
| **Document Processing** | PyPDF, python-docx, BeautifulSoup4 |
| **Frontend** | Streamlit (chat UI, domain filtering, source chips, escalation flow) |
| **Evaluation** | Custom LLM-as-Judge pipeline, weighted rubric scoring |
| **Observability** | LangSmith (optional tracing) |
| **Compute** | PyTorch with GPU support; fully local — no cloud API required |

---

## Project Structure

```
├── src/
│   ├── app.py                          # Streamlit chat UI
│   ├── ingest.py                       # PDF → chunks → embeddings → FAISS
│   ├── run_generator_eval.py           # Model evaluation runner
│   └── policy_assistant/
│       ├── data/
│       │   ├── loaders.py              # PDF document loading
│       │   └── chunking.py             # Parent-child chunking strategy
│       ├── embeddings/
│       │   └── local.py                # HuggingFace embedding wrapper
│       ├── store/
│       │   └── vectorstore.py          # FAISS index build and load
│       ├── retrieval/
│       │   ├── core.py                 # Retrieval + parent expansion + context formatting
│       │   └── algorithms.py           # Flat, HNSW, IVF, LSH, Hybrid backends
│       ├── generation.py               # Ollama chat completion
│       └── eval/
│           ├── common.py               # EvalItem schema, data loading
│           ├── rag_rubric.py           # Rubric loading, weighted score computation
│           ├── retrieval_eval.py       # Retrieval algorithm benchmarking
│           ├── llm_judge.py            # LLM-as-Judge evaluation pipeline
│           ├── embed_eval.py           # Embedding evaluation
│           └── chunk_eval.py           # Chunking strategy evaluation
├── prompts/
│   ├── generator_system_prompt.txt     # 10-rule defensive generator prompt
│   └── judge_system_prompt.txt         # Rubric-adherent judge prompt
├── eval/
│   ├── RAG_SCORING_RUBRIC.md           # 16-criterion weighted evaluation rubric
│   ├── questions/
│   │   └── v2_rubric.json             # 67+ eval items (v2.2)
│   ├── eval_questions.json            # 28 retrieval eval questions (v1)
│   ├── model_comparison.csv           # Cross-model benchmark results
│   ├── judge_results_*.json           # Per-model judge outputs
│   └── responses_*.json              # Per-model generated responses
├── data/docs/                         # 23 corporate policy PDFs (7 domains)
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with at least one model pulled (e.g., `ollama pull olmo2:7b`)

### Setup

```bash
# Clone and install
git clone <repo-url>
cd "AI process-policy assistant"
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env as needed (defaults work for local-only setup)

# Ingest policy documents into FAISS
python src/ingest.py

# Launch the assistant
streamlit run src/app.py
```

### Run Evaluations

```bash
# Retrieval algorithm benchmark
python -m policy_assistant.eval.retrieval_eval \
  --docs_dir data/docs \
  --eval_questions_file eval/eval_questions.json

# Generator evaluation (LLM-as-Judge)
python src/run_generator_eval.py
```

---

## Design Principles

These principles guided every engineering decision:

- **Retrieval before generation** — the LLM never answers from parametric knowledge alone
- **Citations before conclusions** — every claim traces back to a specific policy document
- **Refusal over hallucination** — the system says "I don't know" rather than guessing
- **Contradiction surfacing** — conflicting policies are flagged, not silently resolved
- **Enterprise AI is a system, not a model** — retrieval, chunking, prompting, and evaluation matter as much as the LLM
- **Evaluation is not optional** — a 16-criterion rubric with automated scoring, not just "does it look right?"

---

## What This Project Demonstrates

- Designing and building a **complete RAG system** from document ingestion to user-facing application
- Engineering **retrieval strategies** (parent-child chunking, hybrid search, algorithm benchmarking) for quality and reliability
- Building **systematic evaluation infrastructure** — not just testing outputs, but defining what "good" means for enterprise AI across 16 measurable dimensions
- Writing **defensive, production-grade prompts** that handle adversarial inputs, out-of-domain queries, and safety-critical edge cases
- Making **deliberate engineering trade-offs** (local-first inference, trust over fluency, structured refusal) appropriate for enterprise deployment
- **Comparing and selecting models** empirically using automated rubric-based evaluation rather than subjective judgment
