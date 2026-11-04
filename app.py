"""
app.py - Academic RAG Assistant Streamlit Interface (High-Performance Streaming)

Subject-wise university academic assistant with a polished, modern executive UI.
Allows students to select a subject, ask conceptual and exam questions,
and receive real-time streaming answers grounded in course documents
with exact page/slide citations and zero reload latency.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

# Set page layout & metadata
st.set_page_config(
    page_title="Academic RAG Assistant | University Knowledge System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Custom CSS for Modern Executive UI (Dark/Light Responsive)
st.markdown("""
<style>
    /* Global Typography & Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
        --card-bg: rgba(255, 255, 255, 0.04);
        --card-border: rgba(255, 255, 255, 0.1);
        --badge-bg: rgba(59, 130, 246, 0.15);
        --badge-text: #60A5FA;
    }

    /* Main Hero Banner */
    .hero-container {
        padding: 1.25rem 1.5rem;
        background: linear-gradient(180deg, rgba(30, 58, 138, 0.25) 0%, rgba(15, 23, 42, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.75rem;
        margin-top: 0.4rem;
        font-size: 0.95rem;
    }

    /* Badges & Pills */
    .pill-category {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(139, 92, 246, 0.2);
        color: #C084FC;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }
    .pill-status-indexed {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .pill-status-unindexed {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-top: 0.15rem;
    }

    /* Source Citation Card */
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #60A5FA;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }
    .source-title {
        font-weight: 600;
        color: #93C5FD;
    }
    .source-match {
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        background: rgba(34, 197, 94, 0.18);
        color: #86EFAC;
        font-weight: 600;
    }
    .source-snippet {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        color: #CBD5E1;
        background: rgba(0, 0, 0, 0.25);
        padding: 0.5rem;
        border-radius: 6px;
        white-space: pre-wrap;
        max-height: 140px;
        overflow-y: auto;
    }

    /* Suggestion Chips Section */
    .chips-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_catalog() -> Dict[str, Any]:
    catalog_path = ROOT_DIR / "config" / "subjects_catalog.json"
    if not catalog_path.exists():
        return {}
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_vectorstore_meta() -> Optional[Dict[str, Any]]:
    meta_path = ROOT_DIR / "vectorstore" / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Pre-curated high-yield academic test questions by subject
CURATED_SUGGESTIONS: Dict[str, List[str]] = {
    "generative_ai": [
        "What is the difference between VAE and GAN architectures?",
        "How does self-attention work in Transformer models?",
        "Explain Diffusion Models forward and reverse process"
    ],
    "operating_systems": [
        "What is a Process Control Block (PCB) and what does it store?",
        "What are the 4 necessary conditions for Deadlock?",
        "Explain Round Robin CPU Scheduling vs FCFS"
    ],
    "computer_networks": [
        "What are the 7 layers of the OSI reference model?",
        "What is an IP address and how does routing work?",
        "Explain Longitudinal Redundancy Check (LRC) error detection"
    ]
}


def get_subject_suggestions(subj_id: str, catalog: Dict[str, Any]) -> List[str]:
    """Returns tailored question suggestions for the active subject."""
    if subj_id in CURATED_SUGGESTIONS:
        return CURATED_SUGGESTIONS[subj_id]
    info = catalog.get(subj_id, {})
    title = info.get("title", subj_id)
    return [
        f"Summarize the key core concepts of {title}",
        f"What are the most common exam questions in {title}?",
        f"Explain the fundamental theories and architectures in {title}"
    ]


def get_detected_api_key() -> Optional[str]:
    """Detects Gemini API key from environment or Streamlit Cloud Secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    return key


# Cache the RAG pipeline as a persistent singleton resource
@st.cache_resource(show_spinner="⏳ Initializing AI models & search engine...")
def get_cached_pipeline(api_key: Optional[str] = None):
    import importlib
    import src.rag_pipeline
    importlib.reload(src.rag_pipeline)
    return src.rag_pipeline.AcademicRAGPipeline(api_key=api_key)


# Load Metadata
catalog = load_catalog()
vector_meta = load_vectorstore_meta()

# Ensure pipeline has latest methods and active key
detected_key = get_detected_api_key()
pipeline = get_cached_pipeline(api_key=detected_key)
if not hasattr(pipeline, "generate_response_stream") or not hasattr(pipeline, "ingest_file"):
    st.cache_resource.clear()
    pipeline = get_cached_pipeline(api_key=detected_key)

if detected_key and not pipeline.client:
    pipeline.set_api_key(detected_key)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_prompt_input" not in st.session_state:
    st.session_state.active_prompt_input = None


# ==========================================
# SIDEBAR: Subject Navigation & Controls
# ==========================================
with st.sidebar:
    st.markdown("### 🎓 Academic Navigator")
    st.caption("Select your university course subject to query lecture slides, textbooks, and notes.")

    # Identify indexed subjects
    indexed_subjects = set(vector_meta.get("subjects", {}).keys()) if vector_meta else set()
    all_subject_ids = list(catalog.keys())

    # Sort: Indexed first, then Core subjects, then alphabetical
    def subject_sort_key(s_id):
        is_ind = s_id in indexed_subjects
        is_core = catalog.get(s_id, {}).get("is_core", False)
        title = catalog.get(s_id, {}).get("title", s_id)
        return (not is_ind, not is_core, title)

    sorted_subject_ids = sorted(all_subject_ids, key=subject_sort_key)

    def format_subject_label(s_id):
        info = catalog.get(s_id, {})
        title = info.get("title", s_id)
        status = "🟢" if s_id in indexed_subjects else "⚪"
        return f"{status} {title}"

    selected_subject_id = st.selectbox(
        "Select Course Subject:",
        options=sorted_subject_ids,
        format_func=format_subject_label,
        index=0 if sorted_subject_ids else None,
        key="main_subject_selector"
    )

    st.markdown("---")

    # API Key Configuration
    has_api_key = bool(pipeline.client)
    with st.expander(
        "🟢 Gemini API Connected" if has_api_key else "🔑 Gemini API Key Required",
        expanded=not has_api_key
    ):
        if not has_api_key:
            st.warning("A free Gemini API key is needed to synthesize grounded answers.")
        user_key = st.text_input(
            "Gemini API Key:",
            value=pipeline.api_key or "",
            type="password",
            help="Get your free key from https://aistudio.google.com/app/apikey"
        )
        if user_key and (user_key.strip() != (pipeline.api_key or "") or not pipeline.client):
            pipeline.set_api_key(user_key.strip())
            os.environ["GEMINI_API_KEY"] = user_key.strip()
            st.success("✅ Gemini API connected!")
            st.rerun()

        st.caption("☁️ On Streamlit Cloud: paste `GEMINI_API_KEY = \"...\"` under **Settings ➔ Secrets**.")

    st.markdown("---")

    # Retrieval Tuning Controls
    st.markdown("#### ⚙️ Retrieval Settings")
    top_k = st.slider("Sources to Retrieve (Top-K):", min_value=1, max_value=10, value=5)
    allow_gen_knowledge = st.checkbox(
        "Allow General AI Explanation",
        value=True,
        help="Appends supplementary knowledge under a distinct section if course notes are concise."
    )
    cross_subject = st.checkbox(
        "Search Across All Subjects",
        value=False,
        help="Queries entire university knowledge base across all 40 subjects."
    )

    st.markdown("---")

    # File Ingestion Drawer
    st.markdown("#### 📤 Upload Course Materials")
    with st.expander("Upload & Index New File", expanded=False):
        st.caption("Upload PDFs, PPTX slides, Word docs, or TXT notes to index directly into the vector database.")
        upload_target_subj = st.selectbox(
            "Target Subject:",
            options=sorted_subject_ids,
            format_func=format_subject_label,
            index=sorted_subject_ids.index(selected_subject_id) if selected_subject_id in sorted_subject_ids else 0,
            key="sidebar_upload_subject_select"
        )
        uploaded_files = st.file_uploader(
            "Select documents:",
            type=["pdf", "pptx", "docx", "txt"],
            accept_multiple_files=True,
            key="sidebar_academic_uploader",
            help="PDF preserves page citations; PPTX preserves slide numbers & slide titles."
        )
        if uploaded_files:
            if st.button("⚡ Index Uploaded Files", type="primary", use_container_width=True, key="sidebar_index_btn"):
                target_info = catalog.get(upload_target_subj, {})
                target_title = target_info.get("title", upload_target_subj)
                dest_dir = ROOT_DIR / "data" / "subjects" / upload_target_subj
                dest_dir.mkdir(parents=True, exist_ok=True)

                total_new_chunks = 0
                sidebar_prog = st.progress(0.0)
                sidebar_status = st.empty()

                for idx, uf in enumerate(uploaded_files):
                    sidebar_status.info(f"Indexing ({idx+1}/{len(uploaded_files)}): `{uf.name}`...")
                    save_path = dest_dir / uf.name
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())

                    added = pipeline.ingest_file(
                        str(save_path),
                        subject_id=upload_target_subj,
                        subject_title=target_title
                    )
                    total_new_chunks += added
                    sidebar_prog.progress((idx + 1) / len(uploaded_files))

                sidebar_status.empty()
                sidebar_prog.empty()

                if total_new_chunks > 0:
                    st.success(f"✅ Indexed {len(uploaded_files)} file(s) ({total_new_chunks} chunks) into **{target_title}**!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("⚠️ No extractable text found in uploaded file(s).")

    st.markdown("---")

    # Session Utilities
    st.markdown("#### 🛠️ Session Utilities")
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_export:
        if st.session_state.messages:
            # Build clean markdown transcript for study notes
            transcript_lines = [f"# Academic Study Notes: {selected_subject_id.replace('_', ' ').title()}\n"]
            transcript_lines.append(f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n---\n")
            for m in st.session_state.messages:
                role_label = "### 🧑 Student Question" if m["role"] == "user" else "### 🤖 Assistant Answer"
                transcript_lines.append(f"{role_label}\n{m['content']}\n")
                if m.get("sources"):
                    transcript_lines.append("\n**Sources Cited:**")
                    for s in m["sources"]:
                        loc = f"Page {s.get('page')}" if s.get('page') else f"Slide {s.get('slide')}"
                        transcript_lines.append(f"- `{s.get('source_file')}` ({loc})")
                    transcript_lines.append("\n---\n")

            st.download_button(
                label="📥 Export Notes",
                data="\n".join(transcript_lines),
                file_name=f"{selected_subject_id}_study_notes.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.button("📥 Export Notes", disabled=True, use_container_width=True)

    # Index Summary Footer
    if vector_meta:
        st.caption(f"📊 **Global Index**: {vector_meta.get('total_chunks', 0):,} chunks across {len(vector_meta.get('subjects', {}))} subjects.")
        st.caption(f"🧠 **Model**: `{vector_meta.get('embedding_model', 'all-MiniLM-L6-v2')}`")
    st.caption("🔒 *University Academic Grounding Active*")


# ==========================================
# MAIN APP: Hero Banner & Subject Context
# ==========================================
current_subject_info = catalog.get(selected_subject_id, {})
subject_title = current_subject_info.get("title", selected_subject_id)
subject_category = current_subject_info.get("category", "General Core")
subject_desc = current_subject_info.get("description", "University syllabus and lecture material.")

# Fetch subject specific stats
subject_meta = vector_meta.get("subjects", {}).get(selected_subject_id, {}) if vector_meta else {}
subj_file_count = subject_meta.get("files_count", 0)
subj_chunk_count = subject_meta.get("chunks_count", 0)
is_indexed = selected_subject_id in indexed_subjects if vector_meta else False

# Hero Banner
status_html = (
    f'<span class="pill-status-indexed">🟢 Grounded ({subj_chunk_count:,} chunks)</span>'
    if is_indexed else
    '<span class="pill-status-unindexed">⚪ Cataloged (Upload notes to index)</span>'
)

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">{subject_title}</div>
    <div class="hero-meta">
        <span class="pill-category">📚 {subject_category}</span>
        {status_html}
    </div>
    <p style="color: #94A3B8; margin-top: 0.6rem; margin-bottom: 0; font-size: 0.95rem;">{subject_desc}</p>
</div>
""", unsafe_allow_html=True)

# Metrics Strip
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{subject_category.split()[0]}</div>
        <div class="metric-label">Subject Domain</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{subj_file_count if is_indexed else '0'} Files</div>
        <div class="metric-label">Course Documents</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{subj_chunk_count:,}</div>
        <div class="metric-label">Knowledge Vectors</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">Gemini 2.5</div>
        <div class="metric-label">Reasoning Engine</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

# API Key Missing Alert
if not pipeline.client:
    st.error(
        "⚠️ **Gemini API Key Required**: "
        "Please enter your Gemini API Key in the left sidebar under **🔑 Gemini API Key Required**, "
        "or on Streamlit Cloud add `GEMINI_API_KEY = \"your_key\"` under **App Settings ➔ Secrets**."
    )


# Quick Ingestion for Unindexed / Additional Materials
with st.expander(
    f"📤 Add Course Notes / Slides for {subject_title}" if is_indexed else f"⚠️ No Notes Indexed Yet — Click to Upload for {subject_title}",
    expanded=not is_indexed
):
    if not is_indexed:
        st.info(f"💡 No course notes are currently indexed for **{subject_title}**. Upload lecture slides (PPTX), textbooks (PDF), or notes (DOCX/TXT) below to start asking questions immediately!")
    else:
        st.markdown(f"Upload supplementary lecture materials or exam questions directly for **{subject_title}**.")

    main_uploaded_files = st.file_uploader(
        f"Drop PDF, PPTX, DOCX, or TXT for {subject_title}:",
        type=["pdf", "pptx", "docx", "txt"],
        accept_multiple_files=True,
        key=f"main_uploader_{selected_subject_id}"
    )
    if main_uploaded_files:
        if st.button(f"⚡ Index Files for {subject_title}", type="primary", key=f"main_index_btn_{selected_subject_id}"):
            dest_dir = ROOT_DIR / "data" / "subjects" / selected_subject_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            total_chunks = 0
            main_prog = st.progress(0.0)
            main_status = st.empty()

            for idx, uf in enumerate(main_uploaded_files):
                main_status.info(f"Processing ({idx+1}/{len(main_uploaded_files)}): `{uf.name}`...")
                save_path = dest_dir / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                added = pipeline.ingest_file(
                    str(save_path),
                    subject_id=selected_subject_id,
                    subject_title=subject_title
                )
                total_chunks += added
                main_prog.progress((idx + 1) / len(main_uploaded_files))

            main_status.empty()
            main_prog.empty()

            if total_chunks > 0:
                st.success(f"✅ Successfully indexed {len(main_uploaded_files)} file(s) ({total_chunks} chunks)! Ready to query.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("⚠️ No extractable text found in uploaded file(s).")


# ==========================================
# SUGGESTED PROMPT CHIPS (Click to Ask)
# ==========================================
suggestions = get_subject_suggestions(selected_subject_id, catalog)
st.markdown('<div class="chips-label">💡 Suggested Questions (Click to Ask):</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(suggestions))
for col, suggestion_text in zip(chip_cols, suggestions):
    if col.button(f"💬 {suggestion_text}", use_container_width=True, key=f"btn_chip_{hash(suggestion_text)}"):
        st.session_state.active_prompt_input = suggestion_text
        st.rerun()

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)


# ==========================================
# CHAT TIMELINE
# ==========================================
def render_sources_accordion(sources: List[Dict[str, Any]], elapsed_sec: Optional[float] = None):
    """Renders sleek source citation cards with match relevance."""
    if not sources:
        return
    title_text = f"📄 View {len(sources)} Verified Academic Sources"
    if elapsed_sec:
        title_text += f" ({elapsed_sec:.1f}s)"

    with st.expander(title_text, expanded=False):
        for idx, src in enumerate(sources, start=1):
            file_type = src.get("file_type", "pdf").upper()
            loc_label = f"Page {src['page']}" if src.get("page") else f"Slide {src.get('slide')}"
            title_part = f" — *{src['slide_title']}*" if src.get("slide_title") else ""
            dist = src.get("distance", 1.0)
            # Normalize match score percentage
            match_pct = max(10, min(99, int((1.0 - (dist / 2.0)) * 100)))

            st.markdown(f"""
            <div class="source-card">
                <div class="source-header">
                    <span class="source-title">[{file_type}] {src.get('source_file', 'Document')} ({loc_label}{title_part})</span>
                    <span class="source-match">Relevance: {match_pct}%</span>
                </div>
                <div class="source-snippet">{src.get('text', '')[:400] + ('...' if len(src.get('text', '')) > 400 else '')}</div>
            </div>
            """, unsafe_allow_html=True)


# Render previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources_accordion(msg["sources"])


# ==========================================
# QUERY DISPATCH & STREAMING
# ==========================================
chat_input_val = st.chat_input(f"Ask an academic or exam question about {subject_title}...")

# Check if a suggestion chip was clicked
incoming_query = None
if st.session_state.active_prompt_input:
    incoming_query = st.session_state.active_prompt_input
    st.session_state.active_prompt_input = None
elif chat_input_val:
    incoming_query = chat_input_val

if incoming_query:
    # 1. Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": incoming_query})
    with st.chat_message("user"):
        st.markdown(incoming_query)

    # 2. Stream assistant answer
    with st.chat_message("assistant"):
        search_subj = None if cross_subject else selected_subject_id
        t_start = time.time()

        if hasattr(pipeline, "generate_response_stream"):
            stream_generator, sources, display_subj = pipeline.generate_response_stream(
                query=incoming_query,
                subject_id=search_subj,
                top_k=top_k,
                allow_general_knowledge=allow_gen_knowledge
            )
            # Real-time token streaming directly to UI
            full_response = st.write_stream(stream_generator)
        else:
            resp_data = pipeline.generate_response(
                query=incoming_query,
                subject_id=search_subj,
                top_k=top_k,
                allow_general_knowledge=allow_gen_knowledge
            )
            full_response = resp_data["answer"]
            sources = resp_data.get("sources", [])
            st.markdown(full_response)

        elapsed_sec = time.time() - t_start

        # Render rich source provenance cards
        if sources:
            render_sources_accordion(sources, elapsed_sec=elapsed_sec)

        # Append to message history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
