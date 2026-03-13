"""
Generator LLM Evaluation Pipeline
==================================
Evaluates candidate generator LLMs for the corporate-policy RAG system.
All generator models and the judge run locally via Ollama.

Generator models under test
---------------------------
  1. Qwen 3.5 4B         — Ollama (local)
  2. Qwen 2.5 7B         — Ollama (local)
  3. Phi 3.5 Mini 3.8B   — Ollama (local)
  4. Mistral 7B          — Ollama (local)
  5. Gemma 3 4B          — Ollama (local)
  6. Olmo 2 7B           — Ollama (local)

Judge model
-----------
  - Llama 3.1 8B         — Ollama (local)  (model id: llama3.1:8b)

Prerequisites
-------------
  - Ingested vector store          (python src/ingest.py)
  - Ollama running locally with all models pulled:
        ollama pull qwen3.5:4b
        ollama pull qwen2.5:7b
        ollama pull phi3.5
        ollama pull mistral:7b
        ollama pull gemma3:4b
        ollama pull olmo2:7b
        ollama pull llama3.1:8b

Usage
-----
  python src/run_generator_eval.py                     # full run
  python src/run_generator_eval.py --skip_generate     # reuse cached responses
  python src/run_generator_eval.py --models qwen3.5-4b phi3.5-mini  # subset
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ── project-root on sys.path ─────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
for _p in (_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass


# =====================================================================
# Model registry
# =====================================================================
MODELS: list[dict[str, str]] = [
    {
        "name": "Qwen 3.5 4B",
        "provider": "ollama",
        "model_id": "qwen3.5:4b",
        "slug": "qwen3.5-4b",
    },
    {
        "name": "Qwen 2.5 7B",
        "provider": "ollama",
        "model_id": "qwen2.5:7b",
        "slug": "qwen2.5-7b",
    },
    {
        "name": "Phi 3.5 Mini 3.8B",
        "provider": "ollama",
        "model_id": "phi3.5:latest",
        "slug": "phi3.5-mini",
    },
    {
        "name": "Mistral 7B",
        "provider": "ollama",
        "model_id": "mistral:7b",
        "slug": "mistral-7b",
    },
    {
        "name": "Gemma 3 4B",
        "provider": "ollama",
        "model_id": "gemma3:4b",
        "slug": "gemma3-4b",
    },
    {
        "name": "Olmo 2 7B",
        "provider": "ollama",
        "model_id": "olmo2:7b",
        "slug": "olmo2-7b",
    },
]


# =====================================================================
# Provider wrappers (lazy-initialised)
# =====================================================================
_groq_client = None
_hf_clients: dict[str, Any] = {}


def _groq(model_id: str, messages: list[dict]) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = _groq_client.chat.completions.create(
        model=model_id, messages=messages,
        temperature=0.0, max_tokens=2048,
    )
    return r.choices[0].message.content or ""


def _huggingface(model_id: str, messages: list[dict]) -> str:
    from huggingface_hub import InferenceClient
    if model_id not in _hf_clients:
        _hf_clients[model_id] = InferenceClient(
            model=model_id, token=os.environ.get("HF_TOKEN"),
        )
    r = _hf_clients[model_id].chat_completion(
        messages=messages, temperature=0.01, max_tokens=2048,
    )
    return r.choices[0].message.content or ""


def _ollama(model_id: str, messages: list[dict]) -> str:
    import ollama
    r = ollama.chat(
        model=model_id, messages=messages,
        options={"temperature": 0.0},
    )
    return r["message"]["content"]


_PROVIDERS = {"groq": _groq, "huggingface": _huggingface, "ollama": _ollama}


def chat_completion(provider: str, model_id: str, msgs: list[dict]) -> str:
    return _PROVIDERS[provider](model_id, msgs)


# =====================================================================
# Helpers
# =====================================================================
_SRC_ID_RE = re.compile(r"^([A-Z]+-(?:[A-Z]+-)?[0-9]+)")


def _source_id(source: str) -> str:
    """'…/CORP-01 – Code of Conduct & Ethics.pdf' -> 'CORP-01'"""
    name = Path(source).stem
    m = _SRC_ID_RE.match(name)
    if m:
        return m.group(1)
    return name.split(" ")[0].split("_")[0].strip()


def _retrieve(query: str, vs, pds, k: int = 5):
    """Return (context_string, [source_ids])."""
    from policy_assistant.retrieval.core import get_context_for_llm, get_relevant_chunks
    chunks = get_relevant_chunks(query, vs, k=k)
    context = get_context_for_llm(
        query, vs, pds, k=k, use_parent_content=True,
    )
    ids = list({
        _source_id(d.metadata["source"])
        for d in chunks if d.metadata.get("source")
    })
    return context, ids


# =====================================================================
# Phase 1 — generate responses for every eval item
# =====================================================================
def generate_responses(
    cfg: dict, items: list[dict], vs, pds,
    sys_prompt: str, k: int = 5, rate: float = 1.0,
) -> list[dict]:
    prov, mid, name = cfg["provider"], cfg["model_id"], cfg["name"]
    out: list[dict] = []
    mt_hist: dict[str, list[dict]] = {}
    total = len(items)

    for i, item in enumerate(items, 1):
        qid = item["id"]

        t0 = time.perf_counter()
        ctx, src_ids = _retrieve(item["question"], vs, pds, k)
        t_ret = time.perf_counter() - t0

        user_msg = (
            f"Context:\n{ctx}\n\nUser question:\n{item['question']}"
            if ctx else
            f"Context:\n(No relevant context found)\n\nUser question:\n{item['question']}"
        )

        msgs: list[dict] = [{"role": "system", "content": sys_prompt}]

        if item.get("question_type") == "multi_turn":
            seq = item["multi_turn"]["sequence_id"]
            mt_hist.setdefault(seq, [])
            for prev in mt_hist[seq]:
                msgs.append({"role": "user", "content": prev["u"]})
                msgs.append({"role": "assistant", "content": prev["a"]})

        msgs.append({"role": "user", "content": user_msg})

        print(f"  [{i:>3}/{total}] {qid:<22s}", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            answer = chat_completion(prov, mid, msgs)
            t_gen = time.perf_counter() - t0
            latency = t_ret + t_gen
            print(f"OK  ({latency:.1f}s)")
        except Exception as exc:
            t_gen = time.perf_counter() - t0
            latency = t_ret + t_gen
            answer = f"[ERROR] {exc}"
            print(f"FAIL ({exc})")

        out.append({
            "id": qid,
            "response": answer,
            "retrieved_sources": src_ids,
            "latency_seconds": round(latency, 2),
        })

        if item.get("question_type") == "multi_turn":
            seq = item["multi_turn"]["sequence_id"]
            mt_hist[seq].append({"u": user_msg, "a": answer})

        time.sleep(rate)

    return out


# =====================================================================
# Phase 2 — LLM judge (local via Ollama)
# =====================================================================
def judge_all(
    items_by_id: dict[str, dict],
    responses: list[dict],
    rubric: dict,
    judge_prompt: str,
    rate: float = 0.0,
) -> dict[str, Any]:
    import ollama
    from policy_assistant.eval.llm_judge import (
        MULTI_TURN_PROMPT, SINGLE_TURN_PROMPT,
        _format_criteria_definitions, _format_key_points,
    )
    from policy_assistant.eval.rag_rubric import compute_weighted_score

    crit_defs = _format_criteria_definitions(rubric)
    resp_map = {r["id"]: r for r in responses}
    results: list[dict] = []
    agg: dict[str, list[int]] = {}
    mt_hist: dict[str, list[dict]] = {}
    total = len(items_by_id)

    for idx, (qid, item) in enumerate(items_by_id.items(), 1):
        entry = resp_map.get(qid)
        if not entry:
            continue

        resp_text = entry.get("response", "")
        retrieved = entry.get("retrieved_sources", [])

        conv_hist = None
        if item.get("question_type") == "multi_turn":
            seq = item["multi_turn"]["sequence_id"]
            mt_hist.setdefault(seq, [])
            if mt_hist[seq]:
                lines = []
                for ti, t in enumerate(mt_hist[seq], 1):
                    lines += [f"Turn {ti} Q: {t['q']}", f"Turn {ti} A: {t['a']}"]
                conv_hist = "\n".join(lines)

        crit_focus = item.get("criteria_focus", [])
        cts = (
            ", ".join(f"`{c}`" for c in crit_focus)
            if crit_focus else "(all applicable criteria)"
        )

        fmt = dict(
            question=item["question"],
            response=resp_text,
            retrieved_sources=", ".join(retrieved) if retrieved else "(none)",
            expected_sources=", ".join(item.get("expected_sources", [])),
            key_points=_format_key_points(item.get("key_points", [])),
            criteria_to_score=cts,
            criteria_definitions=crit_defs,
        )

        if item.get("question_type") == "multi_turn":
            mt = item.get("multi_turn", {})
            prompt = MULTI_TURN_PROMPT.format(
                turn_number=mt.get("turn", "?"),
                conversation_history=conv_hist or "(first turn \u2014 no prior history)",
                **fmt,
            )
        else:
            prompt = SINGLE_TURN_PROMPT.format(**fmt)

        print(f"  [Judge {idx:>3}/{total}] {qid:<22s}", end=" ", flush=True)

        result: dict = {"scores": {}}
        for attempt in range(1, 4):
            try:
                raw = ollama.chat(
                    model="llama3.1:8b",
                    messages=[
                        {"role": "system", "content": judge_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.0},
                )
                txt = raw["message"]["content"].strip()
                # Strip optional markdown fences the model might add
                txt = re.sub(r"^```(?:json)?\s*", "", txt)
                txt = re.sub(r"\s*```$", "", txt)

                # First try: parse the whole response as JSON
                try:
                    result = json.loads(txt)
                except json.JSONDecodeError:
                    # Fallback: extract the first JSON-looking substring
                    start = txt.find("{")
                    end = txt.rfind("}") + 1
                    if start != -1 and end > start:
                        candidate = txt[start:end]
                        result = json.loads(candidate)
                    else:
                        # Re-raise to trigger retry logic
                        raise

                print("OK")
                break
            except Exception as exc:
                if attempt == 3:
                    print(f"FAIL ({exc})")
                else:
                    time.sleep(2.0 * attempt)

        scores = result.get("scores", {})

        if item.get("measure_latency") and "latency_seconds" in entry:
            t = entry["latency_seconds"]
            scores["latency"] = {
                "score": 2 if t < 2 else (1 if t <= 5 else 0),
                "justification": f"Measured latency: {t:.2f}s",
            }

        result.update(item_id=qid, scores=scores)
        results.append(result)

        for cid, sc in scores.items():
            v = sc.get("score", 0) if isinstance(sc, dict) else 0
            agg.setdefault(cid, []).append(v)

        if item.get("question_type") == "multi_turn":
            seq = item["multi_turn"]["sequence_id"]
            mt_hist[seq].append({"q": item["question"], "a": resp_text})

        time.sleep(rate)

    mean_scores = {c: sum(v) / len(v) for c, v in agg.items()}
    rounded = {c: round(m) for c, m in mean_scores.items()}
    raw, n01, n05 = compute_weighted_score(rubric, rounded)

    return {
        "item_results": results,
        "aggregate_scores": {
            c: {"mean": sum(v) / len(v), "n": len(v)}
            for c, v in agg.items()
        },
        "mean_scores": mean_scores,
        "weighted_score": {"raw": raw, "normalized_0_1": n01, "normalized_0_5": n05},
    }


# =====================================================================
# Phase 3 — comparison table + winner
# =====================================================================
def show_comparison(all_results: dict[str, dict], rubric: dict) -> None:
    from policy_assistant.eval.rag_rubric import get_criterion_weights, list_criterion_ids

    cids = list_criterion_ids(rubric)
    weights = get_criterion_weights(rubric)
    names = list(all_results.keys())

    CW = 38
    MW = max(18, *(len(n) + 2 for n in names))
    W = CW + 7 + (MW + 3) * len(names)

    def _row(label: str, wt: str, vals: list[str]) -> str:
        r = f"  {label:<{CW}} {wt:>5}"
        for v in vals:
            r += f" | {v:^{MW}}"
        return r

    print(f"\n{'=' * W}")
    print("   GENERATOR LLM EVALUATION  -  COMPARISON TABLE")
    print(f"{'=' * W}")
    print(_row("Criterion", "Wt", names))
    print(f"  {'-' * (W - 2)}")

    for cid in cids:
        w = weights.get(cid, 0)
        vals = []
        for n in names:
            ms = all_results[n].get("mean_scores", {})
            v = ms.get(cid)
            vals.append(f"{v:.2f}" if v is not None else "-")
        print(_row(cid, f"{w:.2f}", vals))

    print(f"  {'-' * (W - 2)}")

    scores_05: dict[str, float] = {}
    vals = []
    for n in names:
        s = all_results[n]["weighted_score"]["normalized_0_5"]
        scores_05[n] = s
        vals.append(f"{s:.2f}")
    print(_row("WEIGHTED SCORE (0-5)", "", vals))

    vals = []
    for n in names:
        s = all_results[n]["weighted_score"]["normalized_0_1"]
        vals.append(f"{s:.3f}")
    print(_row("NORMALIZED (0-1)", "", vals))

    print(f"  {'-' * (W - 2)}")

    ranked = sorted(scores_05.items(), key=lambda x: x[1], reverse=True)
    rank_map = {n: i + 1 for i, (n, _) in enumerate(ranked)}

    vals = []
    for n in names:
        r = rank_map[n]
        vals.append(f"#{r} WINNER" if r == 1 else f"#{r}")
    print(_row("RANK", "", vals))
    print(f"{'=' * W}")

    winner, ws = ranked[0]
    print(f"\n  >>> WINNER: {winner}  -  {ws:.2f} / 5.00 <<<\n")


# =====================================================================
# Pre-flight checks
# =====================================================================
def _preflight(configs: list[dict], skip_gen: bool) -> list[str]:
    warns: list[str] = []
    if not skip_gen:
        provs = {c["provider"] for c in configs}
        if "groq" in provs and not os.environ.get("GROQ_API_KEY"):
            warns.append("GROQ_API_KEY not set - Groq models will fail")
        if "huggingface" in provs and not os.environ.get("HF_TOKEN"):
            warns.append("HF_TOKEN not set - HuggingFace models may fail")
        if "ollama" in provs:
            try:
                import ollama as _ol
                local = _ol.list()
                local_names = {m.model for m in local.models}
                needed = {c["model_id"] for c in configs if c["provider"] == "ollama"} | {
                    "llama3.1:8b"
                }
                missing = needed - local_names
                if missing:
                    cmds = ", ".join(f"ollama pull {m}" for m in sorted(missing))
                    warns.append(f"Ollama models not pulled: {', '.join(sorted(missing))}. Run: {cmds}")
            except Exception:
                warns.append(
                    "Ollama not reachable - ensure it is running (ollama serve)"
                )
    return warns


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate generator LLMs for the corporate-policy RAG system.",
    )
    ap.add_argument("--k", type=int, default=5,
                    help="Chunks to retrieve per query (default: 5)")
    ap.add_argument("--skip_generate", action="store_true",
                    help="Reuse cached response files in eval/")
    ap.add_argument("--skip_judge", action="store_true",
                    help="Reuse cached judge-result files in eval/")
    ap.add_argument("--models", nargs="*",
                    help="Slug(s) of models to evaluate (default: all). "
                         "Available: " + ", ".join(c["slug"] for c in MODELS))
    args = ap.parse_args()

    base = _ROOT
    eval_dir = base / "eval"

    # ── select models ──────────────────────────────────────────────
    cfgs = MODELS
    if args.models:
        cfgs = [c for c in cfgs if c["slug"] in args.models]
    if not cfgs:
        print("No models selected. Available slugs:",
              ", ".join(c["slug"] for c in MODELS))
        return

    # ── preflight ──────────────────────────────────────────────────
    warns = _preflight(cfgs, args.skip_generate)
    if warns:
        print("\n  *** Preflight warnings ***")
        for w in warns:
            print(f"    - {w}")
        print()

    # ── shared resources ───────────────────────────────────────────
    print("Loading evaluation items ...")
    with (eval_dir / "questions" / "v2_rubric.json").open(encoding="utf-8") as f:
        items: list[dict] = json.load(f)["items"]
    items_by_id: dict[str, dict] = {it["id"]: it for it in items}
    print(f"  {len(items)} items loaded ({sum(1 for i in items if i.get('question_type') == 'multi_turn')} multi-turn)")

    from policy_assistant.eval.rag_rubric import load_rubric
    rubric = load_rubric(eval_dir / "rag_rubric.json")

    gen_prompt = (base / "prompts" / "generator_system_prompt.txt").read_text(encoding="utf-8")
    judge_prompt = (base / "prompts" / "judge_system_prompt.txt").read_text(encoding="utf-8")

    vs = pds = None
    if not args.skip_generate:
        print("Loading vector store ...")
        from policy_assistant.retrieval.core import load_retrieval_artifacts
        hf_model = os.environ.get("HF_MODEL", "ibm-granite/granite-embedding-278m-multilingual")
        vs, pds, _ = load_retrieval_artifacts(Path("vector_store"), hf_model)
        print("  Vector store ready.\n")

    # ── per-model loop ─────────────────────────────────────────────
    all_judge: dict[str, dict] = {}

    for cfg in cfgs:
        name, slug = cfg["name"], cfg["slug"]
        resp_path = eval_dir / f"responses_{slug}.json"
        judge_path = eval_dir / f"judge_results_{slug}.json"

        # -- generate --
        if not args.skip_generate:
            print(f"{'=' * 60}")
            print(f"  Generating responses: {name}")
            print(f"  Provider: {cfg['provider']}   Model: {cfg['model_id']}")
            print(f"{'=' * 60}")
            rate = 2.0 if cfg["provider"] == "groq" else 0.0
            try:
                resps = generate_responses(
                    cfg, items, vs, pds, gen_prompt, k=args.k, rate=rate,
                )
            except Exception as exc:
                print(f"\n  *** Generation failed for {name}: {exc} ***")
                print(f"  Skipping this model.\n")
                continue
            resp_path.parent.mkdir(parents=True, exist_ok=True)
            with resp_path.open("w", encoding="utf-8") as f:
                json.dump(resps, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(resps)} responses -> {resp_path.relative_to(base)}\n")
        else:
            if not resp_path.exists():
                print(f"  *** No cached responses for {name} ({resp_path.name}), skipping. ***\n")
                continue
            print(f"  Loading cached responses: {name}")
            with resp_path.open(encoding="utf-8") as f:
                resps = json.load(f)
            print(f"  {len(resps)} responses loaded.\n")

        # -- judge --
        if not args.skip_judge:
            print(f"  Judging: {name}")
            print("  (Judge: llama3.1:8b via Ollama; ETA depends on hardware and context size)")
            try:
                jr = judge_all(items_by_id, resps, rubric, judge_prompt)
            except Exception as exc:
                print(f"\n  *** Judging failed for {name}: {exc} ***")
                print(f"  Skipping this model.\n")
                continue
            with judge_path.open("w", encoding="utf-8") as f:
                json.dump(jr, f, indent=2, ensure_ascii=False)
            ws = jr["weighted_score"]
            print(f"\n  {name}:  {ws['normalized_0_5']:.2f} / 5.00  "
                  f"(raw {ws['raw']:.3f}, norm {ws['normalized_0_1']:.3f})\n")
        else:
            if not judge_path.exists():
                print(f"  *** No cached judge results for {name} ({judge_path.name}), skipping. ***\n")
                continue
            print(f"  Loading cached judge results: {name}")
            with judge_path.open(encoding="utf-8") as f:
                jr = json.load(f)

        all_judge[name] = jr

    # ── final comparison ───────────────────────────────────────────
    if len(all_judge) > 1:
        show_comparison(all_judge, rubric)
    elif len(all_judge) == 1:
        n, jr = next(iter(all_judge.items()))
        ws = jr["weighted_score"]
        print(f"\n  Single model evaluated: {n} -> {ws['normalized_0_5']:.2f} / 5.00\n")
    else:
        print("\n  No models were successfully evaluated.\n")


if __name__ == "__main__":
    main()
