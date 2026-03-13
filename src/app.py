"""
FA Automotive -- Intelligent Policy Assistant (Streamlit UI)

Launch:
    streamlit run src/app.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

# ── path setup ----------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
for _p in (_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from policy_assistant.generation import chat_completion  # noqa: E402
from policy_assistant.retrieval.core import (             # noqa: E402
    get_context_for_llm,
    get_relevant_chunks,
    load_retrieval_artifacts,
)

# ── constants -----------------------------------------------------------------
HF_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
GENERATOR_MODEL = "olmo2:7b"
VECTOR_STORE_DIR = Path("vector_store")
SYSTEM_PROMPT_PATH = _ROOT / "prompts" / "generator_system_prompt.txt"

_SRC_ID_RE = re.compile(r"^([A-Z]+-(?:[A-Z]+-)?[0-9]+)")

SOURCE_LABELS: dict[str, str] = {
    "CORP-01": "Code of Conduct & Ethics",
    "CORP-02": "Data Classification & Handling",
    "CORP-03": "Information Security & Access Control",
    "CORP-04": "Incident Response & Escalation",
    "CORP-05": "Records Retention & Deletion",
    "CORP-06": "Third-Party & Vendor Engagement",
    "CORP-Q-01": "Quality Management System (QMS)",
    "CORP-Q-02": "Audit Readiness & Compliance",
    "CORP-Q-03": "Non-Conformance & CAPA",
    "CORP-Q-04": "Incident Response & Escalation (Quality)",
    "CORP-Q-05": "Supplier Quality Management",
    "HR-01": "Employee Data & Privacy",
    "HR-02": "Remote Work & Device Usage",
    "HR-03": "Expense & Reimbursement",
    "IT-01": "Acceptable Use of IT Systems",
    "IT-02": "Logging, Monitoring & Telemetry",
    "DATA-01": "Data Access & Approval",
    "RND-01": "Research Data Handling",
    "RND-02": "Research Data Handling (v2)",
    "OPS-01": "Supplier Data Sharing",
    "OPS-02": "Quality Incident Reporting",
    "SM-01": "Customer Data Usage",
    "SM-02": "External Communications & Claims",
}

POLICY_DOMAINS: dict[str, list[str]] = {
    "All Domains": [],
    "Corporate Governance": ["CORP-01", "CORP-02", "CORP-03", "CORP-04", "CORP-05", "CORP-06"],
    "Production & Quality": ["CORP-Q-", "OPS-"],
    "IT & Data Governance": ["IT-", "DATA-"],
    "HR & Culture": ["HR-"],
    "R&D": ["RND-"],
    "Sales & Marketing": ["SM-"],
}

DOMAIN_ICONS: dict[str, str] = {
    "All Domains": "📋",
    "Corporate Governance": "⚖️",
    "Production & Quality": "🏭",
    "IT & Data Governance": "🔒",
    "HR & Culture": "👥",
    "R&D": "🔬",
    "Sales & Marketing": "📣",
}

# ── page config (must be first Streamlit call) --------------------------------
st.set_page_config(
    page_title="FA Policy Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers -------------------------------------------------------------------

def _source_id(source: str) -> str:
    """Extract a clean source ID like 'CORP-01' from a file path."""
    name = Path(source).stem
    m = _SRC_ID_RE.match(name)
    if m:
        return m.group(1)
    return name.split(" ")[0].split("_")[0].strip()


def _source_matches_domain(src_id: str, prefixes: list[str]) -> bool:
    """Check if a source ID matches the selected domain's prefixes."""
    if not prefixes:
        return True
    return any(src_id.startswith(p) for p in prefixes)


def _human_source_label(src_id: str) -> str:
    """Return a human-readable label for a source ID."""
    return SOURCE_LABELS.get(src_id, src_id)


def render_sources_and_disclaimer(source_ids: list[str], key_suffix: str) -> None:
    """Render source chips, disclaimer, and action buttons below an answer."""
    st.markdown('<div class="sources-label">Sources Found</div>', unsafe_allow_html=True)
    chips_html = '<div class="source-chips">'
    for sid in source_ids:
        label = _human_source_label(sid)
        chips_html += (
            f'<span class="source-chip">'
            f'<span class="chip-icon">📄</span>{label}'
            f"</span>"
        )
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="disclaimer-box">
            <div class="disc-title">⚠ Automated Policy Disclaimer</div>
            <div class="disc-body">
                I can only provide answers based on documented policies.
                I cannot provide legal advice or approve deviations from these standards.
                For binding legal interpretations, please use the escalation button below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        if st.button("🔁 Escalate to Expert", key=f"esc_{key_suffix}"):
            st.toast("Escalation request sent to the Policy & Compliance team.", icon="✅")
    with col2:
        if st.button("📄 Generate Report", key=f"rep_{key_suffix}"):
            st.toast("Report generation is coming soon.", icon="📄")


# ── cached resources ----------------------------------------------------------

@st.cache_resource(show_spinner="Loading policy knowledge base...")
def load_resources():
    """Load vector store, parent docstore, and system prompt once."""
    import os
    os.chdir(str(_ROOT))
    vs, pds, _ = load_retrieval_artifacts(VECTOR_STORE_DIR, HF_MODEL)
    sys_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return vs, pds, sys_prompt


# ── inject custom CSS ---------------------------------------------------------
st.markdown(
    """
<style>
/* ---------- LIVE badge ---------- */
.live-badge {
    display: inline-block;
    background: #22c55e;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 12px;
    margin-left: 10px;
    vertical-align: middle;
    letter-spacing: 0.05em;
}

/* ---------- header row ---------- */
.header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.header-title {
    font-size: 1.25rem;
    font-weight: 600;
}

/* ---------- source chips ---------- */
.source-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
    margin-bottom: 4px;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.85);
}
.source-chip .chip-icon {
    font-size: 0.9rem;
}

/* ---------- disclaimer box ---------- */
.disclaimer-box {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 12px;
    margin-bottom: 8px;
}
.disclaimer-box .disc-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #f59e0b;
    margin-bottom: 4px;
}
.disclaimer-box .disc-body {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.45;
}

/* ---------- confidence badge ---------- */
.confidence-badge {
    float: right;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    margin-top: 2px;
}

/* ---------- sidebar branding ---------- */
.sidebar-brand {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.sidebar-brand .brand-name {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.sidebar-brand .brand-sub {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.45);
    margin-top: 2px;
}

/* ---------- sidebar section labels ---------- */
.sidebar-section {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.4);
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}

/* ---------- recent-query links ---------- */
.recent-query {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    padding: 4px 0;
}

/* ---------- sidebar user ---------- */
.sidebar-user {
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 0.8rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sidebar-user .user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #3b82f6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}
.sidebar-user .user-name {
    font-size: 0.88rem;
    font-weight: 600;
}
.sidebar-user .user-role {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.45);
}

/* ---------- sources label ---------- */
.sources-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.4);
    margin-top: 14px;
    margin-bottom: 2px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── session state init --------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = "All Domains"

# ── load backend resources ----------------------------------------------------
vs, pds, sys_prompt = load_resources()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-name">⚡ FA Automotive</div>
            <div class="brand-sub">Policy Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Policy Domains</div>',
        unsafe_allow_html=True,
    )

    domain_options = list(POLICY_DOMAINS.keys())
    selected = st.radio(
        "Domain",
        domain_options,
        index=domain_options.index(st.session_state.selected_domain),
        label_visibility="collapsed",
        format_func=lambda d: f"{DOMAIN_ICONS.get(d, '')}  {d}",
    )
    st.session_state.selected_domain = selected

    if st.session_state.recent_queries:
        st.markdown(
            '<div class="sidebar-section">Recent Queries</div>',
            unsafe_allow_html=True,
        )
        for rq in st.session_state.recent_queries[-5:][::-1]:
            truncated = (rq[:50] + "...") if len(rq) > 50 else rq
            st.markdown(
                f'<div class="recent-query">{truncated}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="sidebar-user">
            <div class="user-avatar">JS</div>
            <div>
                <div class="user-name">Julian Schmidt</div>
                <div class="user-role">Policy Analyst</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="header-row">
        <div class="header-title">
            ⚡ Intelligent Policy Assistant <span class="live-badge">LIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── replay chat history -------------------------------------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            if sources:
                st.markdown(
                    '<span class="confidence-badge">'
                    "Response based on documented policies</span>",
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if sources:
                render_sources_and_disclaimer(sources, key_suffix=f"hist_{idx}")
        else:
            st.markdown(msg["content"])

# ── chat input ----------------------------------------------------------------
user_input = st.chat_input(
    "Ask about corporate policy, sustainability, or legal frameworks..."
)

if user_input:
    if user_input not in st.session_state.recent_queries:
        st.session_state.recent_queries.append(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # -- retrieval --
    domain_prefixes = POLICY_DOMAINS.get(st.session_state.selected_domain, [])
    chunks = get_relevant_chunks(user_input, vs, k=8)

    source_ids: list[str] = []
    seen: set[str] = set()
    for doc in chunks:
        src = doc.metadata.get("source", "")
        sid = _source_id(src)
        if sid and sid not in seen and _source_matches_domain(sid, domain_prefixes):
            source_ids.append(sid)
            seen.add(sid)

    context = get_context_for_llm(
        user_input, vs, pds, k=8, use_parent_content=True,
    )
    if not context:
        context = "(No relevant context found)"

    user_msg = f"Context:\n{context}\n\nUser question:\n{user_input}"
    llm_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]

    # -- generation --
    with st.chat_message("assistant"):
        if source_ids:
            st.markdown(
                '<span class="confidence-badge">'
                "Response based on documented policies</span>",
                unsafe_allow_html=True,
            )

        stream = chat_completion(GENERATOR_MODEL, llm_messages, stream=True)
        response = st.write_stream(stream)

        if source_ids:
            render_sources_and_disclaimer(source_ids, key_suffix="new")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": source_ids,
    })
    st.rerun()
