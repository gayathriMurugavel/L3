import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime

from config.settings import DOCUMENTS_DIR
from src.rag.document_processor import load_documents, chunk_documents
from src.rag.vector_store import build_vector_store, retrieve_context
from src.agents.crew_setup import build_crew
from src.evaluation.ragas_evaluator import run_ragas_evaluation
from src.mcp.sqlite_mcp_server import (
    init_db,
    register_document,
    get_recent_queries,
    get_stats,
    add_knowledge_entry,
    save_query,
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Init DB on startup
init_db()

# ── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f, #2980b9);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-left: 4px solid #2980b9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .agent-step {
        background: #e8f4f8;
        border: 1px solid #2980b9;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    .score-good  { color: #27ae60; font-weight: bold; }
    .score-avg   { color: #f39c12; font-weight: bold; }
    .score-poor  { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🧠 Enterprise Knowledge Assistant</h1>
    <p style="margin:0.3rem 0 0 0; opacity:0.85;">
        CrewAI (5 Agents) • RAG • RAGAS Evaluation • MCP Servers • Ollama
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Control Panel")

    sidebar_tab = st.radio(
        "Section",
        ["📚 Knowledge Base", "🔢 Statistics", "📜 History", "➕ Add Entry"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── KNOWLEDGE BASE ───────────────────────────────────────────────
    if sidebar_tab == "📚 Knowledge Base":
        st.subheader("📚 Knowledge Base")
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

        uploaded = st.file_uploader(
            "Upload PDF / TXT / MD",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded:
            for f in uploaded:
                dest = os.path.join(DOCUMENTS_DIR, f.name)
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
            st.success(f"✅ Saved {len(uploaded)} file(s)")

        if st.button("🔄 Build / Rebuild Vector Index", use_container_width=True):
            with st.spinner("Processing documents…"):
                docs = load_documents()
                if not docs:
                    st.error("No documents found. Upload files first.")
                else:
                    chunks = chunk_documents(docs)
                    build_vector_store(chunks)

                    # Register in SQLite MCP
                    seen = {}
                    for ch in chunks:
                        src = ch.metadata.get("source", "unknown")
                        bn = os.path.basename(src)
                        seen[bn] = seen.get(bn, 0) + 1
                    for fname, cnt in seen.items():
                        register_document(
                            fname,
                            f"Auto-indexed: {cnt} chunks",
                            cnt,
                        )
                    st.success(
                        f"✅ Indexed {len(chunks)} chunks "
                        f"from {len(docs)} pages"
                    )

        st.divider()
        st.subheader("📄 Files in KB")
        files = os.listdir(DOCUMENTS_DIR) if os.path.exists(DOCUMENTS_DIR) else []
        if files:
            for f in files:
                st.write(f"📄 {f}")
        else:
            st.info("No files uploaded yet.")

    # ── STATISTICS ───────────────────────────────────────────────────
    elif sidebar_tab == "🔢 Statistics":
        st.subheader("🔢 KB Statistics")
        try:
            stats = get_stats()
            st.metric("Documents Indexed", stats["total_documents"])
            st.metric("Queries Processed", stats["total_queries"])
            st.metric("Knowledge Entries", stats["total_entries"])
            st.divider()
            st.caption("📊 Average RAGAS Scores")
            if stats["avg_faithfulness"]:
                st.progress(
                    stats["avg_faithfulness"],
                    text=f"Faithfulness: {stats['avg_faithfulness']}",
                )
                st.progress(
                    stats["avg_answer_relevancy"],
                    text=f"Ans Relevancy: {stats['avg_answer_relevancy']}",
                )
                st.progress(
                    stats["avg_context_precision"],
                    text=f"Ctx Precision: {stats['avg_context_precision']}",
                )
                st.progress(
                    stats["avg_context_recall"],
                    text=f"Ctx Recall: {stats['avg_context_recall']}",
                )
            else:
                st.info("No RAGAS scores yet.")
        except Exception as e:
            st.error(f"Stats error: {e}")

    # ── QUERY HISTORY ────────────────────────────────────────────────
    elif sidebar_tab == "📜 History":
        st.subheader("📜 Recent Queries")
        try:
            history = get_recent_queries(limit=10)
            if history:
                for h in history:
                    with st.expander(f"🔍 {h['query'][:60]}…"):
                        st.write(h["answer"][:300] + "…")
                        st.caption(h["created_at"])
            else:
                st.info("No queries yet.")
        except Exception as e:
            st.error(f"History error: {e}")

    # ── ADD KNOWLEDGE ENTRY ──────────────────────────────────────────
    elif sidebar_tab == "➕ Add Entry":
        st.subheader("➕ Add Knowledge Entry")
        with st.form("add_entry_form"):
            category = st.text_input("Category", placeholder="HR, Policy, IT…")
            title    = st.text_input("Title")
            content  = st.text_area("Content", height=120)
            source   = st.text_input("Source (optional)")
            submitted = st.form_submit_button("💾 Save Entry")
            if submitted and title and content:
                add_knowledge_entry(category, title, content, source)
                st.success("✅ Entry saved to SQLite MCP!")

# ══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

# ── MODEL INFO BADGES ────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.info("🤖 LLM: llama3.2:1b")
col_m2.info("🔢 Embed: nomic-embed-text")
col_m3.info("🗄️ DB: ChromaDB")
col_m4.info("🗃️ MCP: Filesystem + SQLite")

st.divider()

# ── QUERY INPUT ──────────────────────────────────────────────────────
st.subheader("💬 Ask Your Enterprise Knowledge Base")
query = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g. What is the leave policy for new employees?",
)

# ── OPTIONS ──────────────────────────────────────────────────────────
opt1, opt2, opt3 = st.columns(3)
with opt1:
    run_btn = st.button(
        "🚀 Run Agentic Workflow",
        type="primary",
        use_container_width=True,
    )
with opt2:
    run_ragas = st.checkbox("📊 Run RAGAS Evaluation", value=True)
with opt3:
    show_context = st.checkbox("📑 Show Retrieved Context", value=True)

# ══════════════════════════════════════════════════════════════════════
#  WORKFLOW EXECUTION
# ══════════════════════════════════════════════════════════════════════
if run_btn and query.strip():

    st.divider()
    st.subheader("🤖 Agent Execution Workflow")

    agent_status = st.empty()
    with agent_status.container():
        st.markdown("""
        <div class="agent-step">⏳ <b>Step 1/5:</b> Planner Agent – Analyzing query…</div>
        <div class="agent-step">⏸ <b>Step 2/5:</b> Research Agent – Waiting…</div>
        <div class="agent-step">⏸ <b>Step 3/5:</b> Response Agent – Waiting…</div>
        <div class="agent-step">⏸ <b>Step 4/5:</b> Reviewer Agent – Waiting…</div>
        <div class="agent-step">⏸ <b>Step 5/5:</b> Evaluation Agent – Waiting…</div>
        """, unsafe_allow_html=True)

    try:
        with st.spinner("🤖 CrewAI agents are working… (this may take a minute)"):
            crew, research_task, response_task = build_crew(query)
            result = crew.kickoff(inputs={"query": query})
            result_str = str(result)

        # Update status to all complete
        agent_status.empty()
        with agent_status.container():
            st.markdown("""
            <div class="agent-step">✅ <b>Step 1/5:</b> Planner Agent – Complete</div>
            <div class="agent-step">✅ <b>Step 2/5:</b> Research Agent – Complete</div>
            <div class="agent-step">✅ <b>Step 3/5:</b> Response Agent – Complete</div>
            <div class="agent-step">✅ <b>Step 4/5:</b> Reviewer Agent – Complete</div>
            <div class="agent-step">✅ <b>Step 5/5:</b> Evaluation Agent – Complete</div>
            """, unsafe_allow_html=True)

        # ── FINAL ANSWER ────────────────────────────────────────────
        st.divider()
        st.subheader("✅ Final Answer")
        st.markdown(result_str)

        # ── RETRIEVED CONTEXT ────────────────────────────────────────
        ctx_docs = retrieve_context(query, k=4)
        contexts = [d.page_content for d in ctx_docs]

        if show_context and contexts:
            with st.expander("📑 Retrieved Context from Vector DB", expanded=False):
                for i, (doc, ctx) in enumerate(zip(ctx_docs, contexts), 1):
                    src = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Chunk {i}** | Source: `{os.path.basename(src)}`")
                    st.text(ctx[:600] + ("…" if len(ctx) > 600 else ""))
                    st.divider()

        # ── RAGAS EVALUATION ────────────────────────────────────────
        if run_ragas and contexts:
            st.divider()
            st.subheader("📊 RAGAS Evaluation Results")

            with st.spinner("Running RAGAS evaluation…"):
                df = run_ragas_evaluation(
                    question=query,
                    answer=result_str,
                    contexts=contexts,
                )

            metrics = {
                "faithfulness":      df["faithfulness"].iloc[0]      if "faithfulness"      in df.columns else None,
                "answer_relevancy":  df["answer_relevancy"].iloc[0]  if "answer_relevancy"  in df.columns else None,
                "context_precision": df["context_precision"].iloc[0] if "context_precision" in df.columns else None,
                "context_recall":    df["context_recall"].iloc[0]    if "context_recall"    in df.columns else None,
            }

            # Save scores to SQLite
            save_query(
                query,
                result_str,
                {k: float(v) for k, v in metrics.items() if v is not None},
            )

            # ── SCORE DISPLAY ────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)

            def fmt(v):
                return f"{float(v):.3f}" if v is not None else "N/A"

            def delta(v):
                if v is None:
                    return None
                fv = float(v)
                return "🟢 Good" if fv >= 0.7 else ("🟡 Average" if fv >= 0.4 else "🔴 Poor")

            m1.metric("Faithfulness",      fmt(metrics["faithfulness"]),      delta(metrics["faithfulness"]))
            m2.metric("Answer Relevancy",  fmt(metrics["answer_relevancy"]),  delta(metrics["answer_relevancy"]))
            m3.metric("Context Precision", fmt(metrics["context_precision"]), delta(metrics["context_precision"]))
            m4.metric("Context Recall",    fmt(metrics["context_recall"]),    delta(metrics["context_recall"]))

            # ── INTERPRETATION ───────────────────────────────────────
            with st.expander("📖 Metric Interpretation", expanded=True):
                st.markdown("""
| Metric | What it measures | Good score |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | ≥ 0.7 |
| **Answer Relevancy** | Does the answer address the question? | ≥ 0.7 |
| **Context Precision** | Are retrieved chunks signal (not noise)? | ≥ 0.7 |
| **Context Recall** | Was enough relevant context retrieved? | ≥ 0.7 |

**Overall Assessment:**
""")
                valid = [float(v) for v in metrics.values() if v is not None]
                avg_score = sum(valid) / len(valid) if valid else 0

                if avg_score >= 0.7:
                    st.success(
                        f"🟢 Strong performance (avg: {avg_score:.3f}) — "
                        "The system is producing high-quality, grounded answers."
                    )
                elif avg_score >= 0.4:
                    st.warning(
                        f"🟡 Average performance (avg: {avg_score:.3f}) — "
                        "Consider adding more relevant documents to the knowledge base."
                    )
                else:
                    st.error(
                        f"🔴 Low performance (avg: {avg_score:.3f}) — "
                        "Knowledge base may be insufficient for this query."
                    )

            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error during execution: {e}")
        st.exception(e)

elif run_btn and not query.strip():
    st.warning("⚠️ Please enter a question first.")