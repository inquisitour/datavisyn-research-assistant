"""
app.py - Streamlit UI for the Gene Research Assistant.

Connects to the running FastAPI backend at http://localhost:8000
and streams responses directly into the chat interface.

Run with:
    streamlit run app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gene Research Assistant",
    page_icon="🧬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Header */
.gene-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.gene-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 2px;
    margin-bottom: 1.5rem;
}

/* Stat cards */
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.stat-label {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}

/* Chat messages */
.stChatMessage {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* Input box */
.stChatInputContainer {
    border-top: 1px solid #30363d;
    background-color: #0d1117;
}

/* Example query pills */
.example-pill {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #9ca3af;
    margin: 3px;
    cursor: pointer;
    font-family: 'IBM Plex Mono', monospace;
}

/* Divider */
hr { border-color: #30363d; }

/* Streamlit button override */
.stButton > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    padding: 4px 12px;
    width: 100%;
    text-align: left;
}
.stButton > button:hover {
    background: #30363d;
    border-color: #58a6ff;
    color: #58a6ff;
}

/* Chat text brightness */
[data-testid="stChatMessageContent"] {
    color: #e6edf3 !important;
}

/* Hide top and bottom streamlit toolbars/decorations */
header[data-testid="stHeader"] {
    background-color: #0d1117 !important;
}
[data-testid="stBottom"] {
    background-color: #0d1117 !important;
}

[data-testid="stBottom"] > div {
    background-color: #0d1117 !important;
}

</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_stats() -> dict:
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        return r.json()
    except Exception:
        return {}


def stream_query(question: str):
    """Generator: yields text chunks from the SSE stream, stripping `data:` prefix."""
    with requests.post(
        f"{API_URL}/query",
        json={"question": question},
        stream=True,
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
            if text == "[DONE]":
                break
            yield text

def pretty_biotype(bt: str) -> str:
    result = bt.replace("_r_n_a", " RNA").replace("_", " ")
    return result.title().replace("Rna", "RNA")

def fix_formatting(text: str) -> str:
    """Ensure bullet points and sections render properly in Streamlit markdown."""
    import re
    text = re.sub(r'(?<!\n)(- \*\*)', r'\n\1', text)
    text = re.sub(r'(?<!\n)(- \*\()', r'\n\1', text)
    text = re.sub(r'(\*\*[A-Z][^*]+Genes[^*]*\*\*:)', r'\n\n\1\n', text)
    text = re.sub(r'(\w)\]([A-Z])', r'\1]\n\n\2', text)
    text = re.sub(r'(\w)(Chromosome \d)', r'\1\n\n\2', text)
    text = re.sub(r'(\w)(The dataset|This dataset|Overall)', r'\1\n\n\2', text)
    text = re.sub(r'(\))(Chromosome|The |These |This )', r'\1\n\n\2', text)
    text = re.sub(r'(\d)(The |These |This )', r'\1\n\n\2', text)
    text = re.sub(r'(gene)(The |These |This )', r'\1\n\n\2', text)
    text = re.sub(r'(tools\.)([A-Z])', r'\1\n\n\2', text)
    text = re.sub(r'(\.)([A-Z][a-z])', r'\1\n\n\2', text)
    text = re.sub(r'(\.)([A-Z])', r'\1\n\n\2', text)
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1\n\n\2', text)
    text = re.sub(r'(\d)([A-Z][a-z])', r'\1\n\n\2', text)
    text = re.sub(r'(\.)(\d)', r'\1 \2', text)
    text = re.sub(r'(\.)(\*\*[A-Z])', r'\1\n\n\2', text)
    return text


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="gene-header">🧬 GeneQuery</div>', unsafe_allow_html=True)
    st.markdown('<div class="gene-sub">Human Gene Research Assistant</div>', unsafe_allow_html=True)

    stats = fetch_stats()
    if stats:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Genes</div>
            <div class="stat-value">{stats.get('total_genes', '—'):,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Chromosomes</div>
            <div class="stat-value">{len(stats.get('chromosomes', []))}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">With Symbol</div>
            <div class="stat-value">{stats.get('genes_with_symbol', '—'):,}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<p style='color:#c9d1d9;font-weight:600;margin-bottom:8px'>Biotype distribution</p>", unsafe_allow_html=True)
        biotypes = stats.get("biotypes", {})
        total = sum(biotypes.values()) or 1
        for bt, count in sorted(biotypes.items(), key=lambda x: -x[1])[:8]:
            pct = count / total * 100
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<small style='color:#c9d1d9;font-family:IBM Plex Mono,monospace'>{pretty_biotype(bt)}</small>"
                f"<small style='color:#8b949e'>{count:,}</small>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.progress(pct / 100)

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown('<div class="gene-header" style="font-size:1.3rem">Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="gene-sub">Ask anything about the human gene dataset</div>', unsafe_allow_html=True)

# Example queries
EXAMPLES = [
    "Protein coding genes on chromosome 17",
    "How many genes per chromosome?",
    "Find all glutathione peroxidase genes",
    "List pseudogenes on chromosome X",
    "What biotypes exist in the dataset?",
]

cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    with col:
        if st.button(ex, key=f"ex_{i}"):
            st.session_state["prefill"] = ex

st.markdown("---")

# ── Chat state ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(fix_formatting(msg["content"]))
        else:
            st.markdown(msg["content"])

# Handle prefill from example buttons
prefill = st.session_state.pop("prefill", None)

# Chat input
prompt = st.chat_input("Ask about genes, chromosomes, biotypes…") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            for chunk in stream_query(prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(fix_formatting(full_response))
        except requests.exceptions.ConnectionError:
            full_response = "⚠️ Cannot reach the API at `localhost:8000`. Is uvicorn running?"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})