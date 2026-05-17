import streamlit as st
import os
import datetime
from ingest import save_uploaded_file, ingest_file, reset_vectorstore, DOCS_FOLDER
from retriever import (
    retrieve_and_answer, retrieve_and_stream,
    check_hallucination, list_ingested_docs,
    AVAILABLE_MODELS
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralDoc · RAG Assistant",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c10 !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(99,102,241,0.08) 0%, transparent 55%),
        #080c10 !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stSidebar"] {
    background: rgba(8,12,18,0.98) !important;
    border-right: 1px solid rgba(56,189,248,0.1) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

.main .block-container {
    max-width: 880px !important;
    padding: 2rem 2rem 6rem !important;
    margin: 0 auto !important;
}

.nd-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 1.8rem; padding-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.nd-logo {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%);
    clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    flex-shrink: 0;
    animation: pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { filter: drop-shadow(0 0 4px rgba(56,189,248,0.4)); }
    50%      { filter: drop-shadow(0 0 12px rgba(56,189,248,0.7)); }
}
.nd-title {
    font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #f0f9ff 30%, #7dd3fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1;
}
.nd-subtitle {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.66rem; color: #475569;
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px;
}
.nd-msg-wrap {
    display: flex; gap: 12px; margin-bottom: 1.2rem;
    animation: fadeUp 0.3s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.nd-msg-wrap.user { flex-direction: row-reverse; }
.nd-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}
.nd-avatar.user-av { background: linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; }
.nd-avatar.ai-av   { background: linear-gradient(135deg,#0ea5e9,#38bdf8); color:#fff; }
.nd-bubble {
    max-width: 76%; padding: 12px 16px; border-radius: 16px;
    font-size: 0.9rem; line-height: 1.65; color: #e2e8f0;
}
.nd-bubble.user-bubble {
    background: linear-gradient(135deg,rgba(99,102,241,0.2),rgba(139,92,246,0.13));
    border: 1px solid rgba(99,102,241,0.28); border-top-right-radius: 4px;
}
.nd-bubble.ai-bubble {
    background: rgba(12,18,28,0.85); border: 1px solid rgba(56,189,248,0.13);
    border-top-left-radius: 4px; backdrop-filter: blur(8px);
}
.nd-conf-wrap {
    margin-top: 10px; display: flex; align-items: center; gap: 8px;
}
.nd-conf-label { font-family:'DM Mono',monospace; font-size:0.65rem; color:#475569; white-space:nowrap; }
.nd-conf-bar   { flex:1; height:4px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; }
.nd-conf-fill  { height:100%; border-radius:4px; transition:width 0.6s ease; }
.nd-conf-pct   { font-family:'DM Mono',monospace; font-size:0.65rem; color:#7dd3fc; min-width:30px; }
.nd-verdict {
    display: inline-flex; align-items: center; gap: 5px;
    margin-top: 8px; padding: 4px 11px; border-radius: 20px;
    font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.04em;
}
.nd-verdict.grounded     { background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.28); color:#34d399; }
.nd-verdict.not-grounded { background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.28); color:#fbbf24; }
.nd-src-label {
    font-family:'DM Mono',monospace; font-size:0.64rem;
    letter-spacing:0.1em; text-transform:uppercase; color:#334155; margin:10px 0 6px;
}
.nd-src-card {
    background:rgba(10,16,26,0.7); border:1px solid rgba(255,255,255,0.05);
    border-left:3px solid #38bdf8; border-radius:7px; padding:9px 12px;
    margin-bottom:6px; font-size:0.78rem; color:#94a3b8; line-height:1.5;
}
.nd-src-meta {
    font-family:'DM Mono',monospace; font-size:0.63rem;
    color:#38bdf8; margin-bottom:4px; opacity:0.75;
}
.nd-doc-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:4px 10px; border-radius:20px; margin:3px;
    background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.18);
    font-family:'DM Mono',monospace; font-size:0.68rem; color:#7dd3fc;
}
.nd-stat-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:1.2rem; }
.nd-stat-card {
    flex:1; min-width:70px; padding:10px 14px;
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:10px; text-align:center;
}
.nd-stat-val { font-size:1.3rem; font-weight:800; }
.nd-stat-key { font-family:'DM Mono',monospace; font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:0.07em; margin-top:2px; }
.nd-empty { text-align:center; padding:4rem 2rem; }
.nd-empty-icon { font-size:2.5rem; margin-bottom:1rem; opacity:0.35; }
.nd-empty-text { font-size:0.95rem; font-weight:600; color:#475569; margin-bottom:0.4rem; }
.nd-empty-sub  { font-family:'DM Mono',monospace; font-size:0.7rem; color:#334155; }
[data-testid="stChatInput"] {
    background:rgba(12,18,28,0.95) !important; border:1px solid rgba(56,189,248,0.2) !important; border-radius:14px !important;
}
[data-testid="stChatInput"]:focus-within { border-color:rgba(56,189,248,0.5) !important; box-shadow:0 0 0 3px rgba(56,189,248,0.07) !important; }
[data-testid="stChatInput"] textarea { font-family:'Syne',sans-serif !important; font-size:0.88rem !important; color:#e2e8f0 !important; }
.stButton button { font-family:'Syne',sans-serif !important; font-weight:600 !important; border-radius:9px !important; }
.stSelectbox > div > div { background:rgba(12,18,28,0.9) !important; border:1px solid rgba(255,255,255,0.08) !important; color:#e2e8f0 !important; border-radius:9px !important; }
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(56,189,248,0.18); border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "messages": [], "query_history": [], "total_queries": 0,
    "grounded_count": 0, "selected_model": "llama-3.3-70b-versatile",
    "use_reranker": True, "selected_docs": [], "streaming": True,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

os.makedirs(DOCS_FOLDER, exist_ok=True)

def conf_color(pct):
    if pct >= 70: return "#34d399"
    if pct >= 40: return "#fbbf24"
    return "#f87171"

def render_message(msg):
    role, content = msg["role"], msg.get("content", "")
    verdict, sources, conf = msg.get("verdict",""), msg.get("sources",[]), msg.get("confidence",0)
    if role == "user":
        st.markdown(f"""
        <div class="nd-msg-wrap user">
            <div class="nd-avatar user-av">U</div>
            <div class="nd-bubble user-bubble">{content}</div>
        </div>""", unsafe_allow_html=True)
    else:
        is_g = "Not" not in verdict and "Grounded" in verdict
        vc   = "grounded" if is_g else "not-grounded"
        vi   = "✦" if is_g else "⚠"
        srcs = ""
        if sources:
            srcs = '<div class="nd-src-label">Source Chunks</div>'
            for i, s in enumerate(sources):
                fname = os.path.basename(s["source"]) if s["source"] != "Unknown" else "Unknown"
                srcs += f'<div class="nd-src-card"><div class="nd-src-meta">[{i+1}] {fname} · pg {s["page"]} · dist {s["score"]}</div>{s["content"]}</div>'
        conf_html = f"""<div class="nd-conf-wrap"><span class="nd-conf-label">Confidence</span>
            <div class="nd-conf-bar"><div class="nd-conf-fill" style="width:{conf}%;background:{conf_color(conf)};"></div></div>
            <span class="nd-conf-pct">{conf}%</span></div>""" if verdict else ""
        st.markdown(f"""
        <div class="nd-msg-wrap">
            <div class="nd-avatar ai-av">AI</div>
            <div class="nd-bubble ai-bubble">{content}{conf_html}
                <div class="nd-verdict {vc}">{vi} {verdict}</div>{srcs}
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════ SIDEBAR ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style='margin-bottom:1.2rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,0.05)'>
        <div style='font-size:1.05rem;font-weight:800;background:linear-gradient(90deg,#f0f9ff,#7dd3fc);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>NeuralDoc</div>
        <div style='font-family:"DM Mono",monospace;font-size:0.62rem;color:#334155;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;'>RAG · v2.0</div>
    </div>""", unsafe_allow_html=True)

    q, g = st.session_state.total_queries, st.session_state.grounded_count
    ac, nd = (f"{int(g/q*100)}%" if q>0 else "—"), len(list_ingested_docs())
    st.markdown(f"""<div class="nd-stat-row">
        <div class="nd-stat-card"><div class="nd-stat-val" style="color:#7dd3fc">{q}</div><div class="nd-stat-key">Queries</div></div>
        <div class="nd-stat-card"><div class="nd-stat-val" style="color:#34d399">{g}</div><div class="nd-stat-key">Grounded</div></div>
        <div class="nd-stat-card"><div class="nd-stat-val" style="color:#818cf8">{ac}</div><div class="nd-stat-key">Accuracy</div></div>
        <div class="nd-stat-card"><div class="nd-stat-val" style="color:#f472b6">{nd}</div><div class="nd-stat-key">Docs</div></div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Model</div>', unsafe_allow_html=True)
    model_labels = list(AVAILABLE_MODELS.values())
    model_keys   = list(AVAILABLE_MODELS.keys())
    sel_label = st.selectbox("model", model_labels, index=0, label_visibility="collapsed")
    st.session_state.selected_model = model_keys[model_labels.index(sel_label)]

    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 6px;">Settings</div>', unsafe_allow_html=True)
    k_chunks = st.slider("Chunks to retrieve", 2, 8, 4, 1)
    c1, c2 = st.columns(2)
    with c1: st.session_state.use_reranker = st.toggle("Re-ranker", value=True)
    with c2: st.session_state.streaming    = st.toggle("Streaming",  value=True)

    docs = list_ingested_docs()
    if docs:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 6px;">Search in Docs</div>', unsafe_allow_html=True)
        selected = st.multiselect("docs", docs, default=docs, label_visibility="collapsed")
        st.session_state.selected_docs = selected

    st.divider()

    if st.session_state.query_history:
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">History</div>', unsafe_allow_html=True)
        for i, pq in enumerate(reversed(st.session_state.query_history[-8:])):
            label = f"↩ {pq[:36]}…" if len(pq) > 36 else f"↩ {pq}"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state["rerun_query"] = pq
                st.rerun()
        st.divider()

    if st.session_state.messages:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.total_queries  = 0
            st.session_state.grounded_count = 0
            st.rerun()
        try:
            from exporter import export_chat_to_pdf
            if st.button("📄 Export as PDF", use_container_width=True):
                path = export_chat_to_pdf(
                    st.session_state.messages,
                    filepath=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
                with open(path, "rb") as f:
                    st.download_button("⬇ Download", f.read(), file_name=os.path.basename(path),
                                       mime="application/pdf", use_container_width=True)
        except ImportError:
            st.caption("pip install fpdf2 for PDF export")

# ══════════════════════════ MAIN ══════════════════════════════════════════════
st.markdown("""<div class="nd-header">
    <div class="nd-logo"></div>
    <div><div class="nd-title">NeuralDoc</div>
    <div class="nd-subtitle">Knowledge-Grounded RAG Assistant · v2.0</div></div>
</div>""", unsafe_allow_html=True)

tab_chat, tab_docs, tab_compare = st.tabs(["💬 Chat", "📂 Documents", "⚖ Compare Models"])

# ─────────────── TAB 1: CHAT ──────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.messages:
        st.markdown("""<div class="nd-empty">
            <div class="nd-empty-icon">⬡</div>
            <div class="nd-empty-text">Ready to answer from your documents</div>
            <div class="nd-empty-sub">Upload PDFs in the Documents tab, then ask anything here</div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg)

    rerun_q = st.session_state.pop("rerun_query", None)
    query   = st.chat_input("Ask anything from your documents…") or rerun_q

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.query_history.append(query)
        st.session_state.total_queries += 1
        render_message({"role": "user", "content": query})

        if st.session_state.streaming:
            with st.chat_message("assistant"):
                placeholder  = st.empty()
                full_answer  = ""
                for token in retrieve_and_stream(
                    query, k=k_chunks, model_name=st.session_state.selected_model,
                    history=st.session_state.messages[:-1],
                    use_reranker=st.session_state.use_reranker,
                    selected_docs=st.session_state.selected_docs or None
                ):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)

            last    = retrieve_and_stream.last_results
            context = last.get("context", "")
            sources = last.get("sources", [])
            conf    = last.get("confidence", 0)
            verdict = ""
            if context:
                verdict = check_hallucination(full_answer, context, st.session_state.selected_model)
                if "Not" not in verdict and "Grounded" in verdict:
                    st.session_state.grounded_count += 1

            st.session_state.messages.append({
                "role": "assistant", "content": full_answer,
                "verdict": verdict, "sources": sources, "confidence": conf
            })
            st.rerun()
        else:
            with st.spinner("Searching and reasoning…"):
                result = retrieve_and_answer(
                    query, k=k_chunks, model_name=st.session_state.selected_model,
                    history=st.session_state.messages[:-1],
                    use_reranker=st.session_state.use_reranker,
                    selected_docs=st.session_state.selected_docs or None
                )
            verdict = result["hallucination_check"]
            if "Not" not in verdict and "Grounded" in verdict:
                st.session_state.grounded_count += 1
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"],
                "verdict": verdict, "sources": result["sources"], "confidence": result["confidence"]
            })
            st.rerun()

# ─────────────── TAB 2: DOCUMENTS ─────────────────────────────────────────────
with tab_docs:
    st.markdown("### Upload Documents")
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">Drop PDFs here — chunked, embedded and indexed automatically.</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"],
                                       accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        for uf in uploaded_files:
            prog_txt = st.empty()
            prog_bar = st.progress(0)
            step = [0]
            def progress_cb(msg, _txt=prog_txt, _bar=prog_bar, _step=step):
                _txt.markdown(f'<span style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#7dd3fc;">{msg}</span>', unsafe_allow_html=True)
                _step[0] = min(_step[0] + 33, 99)
                _bar.progress(_step[0])
            saved = save_uploaded_file(uf)
            res   = ingest_file(saved, progress_cb=progress_cb)
            prog_bar.progress(100)
            if res["success"]:
                prog_txt.markdown(f'<span style="color:#34d399;font-family:\'DM Mono\',monospace;font-size:0.75rem;">✓ {res["filename"]} — {res["pages"]} pages, {res["chunks"]} chunks</span>', unsafe_allow_html=True)
            else:
                prog_txt.markdown(f'<span style="color:#f87171;font-family:\'DM Mono\',monospace;font-size:0.75rem;">✗ {res["filename"]} — {res["error"]}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Indexed Documents")
    docs = list_ingested_docs()
    if not docs:
        st.markdown('<p style="color:#334155;font-size:0.85rem;">No documents indexed yet.</p>', unsafe_allow_html=True)
    else:
        st.markdown("".join([f'<span class="nd-doc-pill">📄 {d}</span>' for d in docs]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Reset All Documents", type="secondary"):
            reset_vectorstore()
            st.success("Vector store cleared.")
            st.rerun()

# ─────────────── TAB 3: COMPARE ───────────────────────────────────────────────
with tab_compare:
    st.markdown("### Compare Model Responses")
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">Same question, different models, side by side.</p>', unsafe_allow_html=True)

    compare_models = st.multiselect(
        "Select models", options=list(AVAILABLE_MODELS.keys()),
        default=list(AVAILABLE_MODELS.keys())[:2],
        format_func=lambda x: AVAILABLE_MODELS[x]
    )
    compare_query = st.text_input("Your question", placeholder="e.g. What is the main contribution of this paper?")

    if st.button("⚖ Run Comparison", type="primary") and compare_query and compare_models:
        cols = st.columns(len(compare_models))
        for i, mk in enumerate(compare_models):
            with cols[i]:
                st.markdown(f"""<div style='padding:8px 12px;background:rgba(56,189,248,0.08);
                    border:1px solid rgba(56,189,248,0.18);border-radius:8px;margin-bottom:12px;
                    font-family:"DM Mono",monospace;font-size:0.72rem;color:#38bdf8;'>
                    {AVAILABLE_MODELS[mk]}</div>""", unsafe_allow_html=True)
                with st.spinner(f"Querying…"):
                    res = retrieve_and_answer(compare_query, k=k_chunks, model_name=mk, history=[],
                                             use_reranker=st.session_state.use_reranker)
                vd = res["hallucination_check"]
                cf = res["confidence"]
                ig = "Not" not in vd and "Grounded" in vd
                st.markdown(f"""
                <div style='background:rgba(12,18,28,0.85);border:1px solid rgba(56,189,248,0.12);
                    border-radius:12px;padding:14px;font-size:0.85rem;color:#e2e8f0;line-height:1.6;margin-bottom:10px;'>
                    {res["answer"]}</div>
                <div class="nd-verdict {'grounded' if ig else 'not-grounded'}">{'✦' if ig else '⚠'} {vd}</div>
                <div class="nd-conf-wrap" style="margin-top:8px;">
                    <span class="nd-conf-label">Confidence</span>
                    <div class="nd-conf-bar"><div class="nd-conf-fill" style="width:{cf}%;background:{conf_color(cf)};"></div></div>
                    <span class="nd-conf-pct">{cf}%</span>
                </div>""", unsafe_allow_html=True)
