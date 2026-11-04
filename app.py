"""
app.py - Academic RAG Assistant Streamlit Interface (High-Performance Streaming)

Subject-wise university academic assistant.
Allows students to select a subject, ask conceptual and exam questions,
and receive real-time streaming answers grounded in course documents
with exact page/slide citations and zero reload latency.
"""

import os
import sys
import json
import time
from pathlib import Path

import streamlit as st

# Set page layout
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Custom CSS for academic polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        margin-top: 0.6rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_catalog():
    catalog_path = ROOT_DIR / "config" / "subjects_catalog.json"
    if not catalog_path.exists():
        return {}
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_vectorstore_meta():
    meta_path = ROOT_DIR / "vectorstore" / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# CACHE THE RAG PIPELINE AS A SINGLETON RESOURCE
# This eliminates the model loading penalty on user queries!
@st.cache_resource(show_spinner="⏳ Warming up AI models and search engine (one-time initialization)...")
def get_cached_pipeline():
    import importlib
    import src.rag_pipeline
    importlib.reload(src.rag_pipeline)
    return src.rag_pipeline.AcademicRAGPipeline()


# Load Subject Catalog & Metadata
catalog = load_catalog()
vector_meta = load_vectorstore_meta()

# Pre-warm pipeline so it's instant when user asks a question
pipeline = get_cached_pipeline()
if not hasattr(pipeline, "generate_response_stream") or not hasattr(pipeline, "ingest_file"):
    st.cache_resource.clear()
    pipeline = get_cached_pipeline()

# Sidebar: Subject Selection & Settings
with st.sidebar:
    st.title("🎓 Academic Navigator")
    st.markdown("Select a **Subject** to query course notes, lecture slides, and textbooks.")

    # Get available subjects
    indexed_subjects = set(vector_meta.get("subjects", {}).keys()) if vector_meta else set()

    # Sort subjects: indexed ones first
    all_subject_ids = list(catalog.keys())

    def subject_sort_key(s_id):
        is_indexed = s_id in indexed_subjects
        is_core = catalog[s_id].get("is_core", False)
        return (not is_indexed, not is_core, catalog[s_id].get("title", s_id))

    sorted_subject_ids = sorted(all_subject_ids, key=subject_sort_key)

    def format_subject_label(s_id):
        info = catalog.get(s_id, {})
        title = info.get("title", s_id)
        status = "🟢" if s_id in indexed_subjects else "⚪"
        return f"{status} {title}"

    selected_subject_id = st.selectbox(
        "Choose Subject:",
        options=sorted_subject_ids,
        format_func=format_subject_label,
        index=0 if sorted_subject_ids else None
    )

    st.divider()

    # Search options
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Top Sources to Retrieve (Top-K):", min_value=1, max_value=10, value=5)
    allow_gen_knowledge = st.checkbox(
        "Allow General AI Explanation",
        value=True,
        help="If checked, Gemini adds supplementary general knowledge under a clearly labeled section if course notes are brief."
    )
    cross_subject = st.checkbox(
        "Search Across All Subjects",
        value=False,
        help="If checked, searches the entire knowledge base instead of strictly isolating to the selected subject."
    )

    st.divider()

    # File Upload Section in Sidebar
    st.subheader("📤 Upload Material")
    with st.expander("Upload & Index New File", expanded=False):
        upload_target_subj = st.selectbox(
            "Assign to Subject:",
            options=sorted_subject_ids,
            format_func=format_subject_label,
            index=sorted_subject_ids.index(selected_subject_id) if selected_subject_id in sorted_subject_ids else 0,
            key="sidebar_upload_subject_select"
        )
        uploaded_files = st.file_uploader(
            "Select files:",
            type=["pdf", "pptx", "docx", "txt"],
            accept_multiple_files=True,
            key="sidebar_academic_uploader",
            help="Supported: PDF (page citations), PPTX (slide citations), DOCX, TXT"
        )
        if uploaded_files:
            if st.button("⚡ Index Uploaded Material", type="primary", use_container_width=True, key="sidebar_index_btn"):
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

    st.divider()

    # Indexing status card
    if vector_meta:
        st.caption(f"**Index Status**: {vector_meta.get('total_chunks', 0):,} chunks indexed.")
        st.caption(f"**Model**: `{vector_meta.get('embedding_model', 'all-MiniLM-L6-v2')}`")
        st.caption(f"**Updated**: {vector_meta.get('created_at', 'N/A')}")
    else:
        st.warning("⚠️ Vectorstore index is building or not found.")


# Main App Area
current_subject_info = catalog.get(selected_subject_id, {})
subject_title = current_subject_info.get("title", selected_subject_id)
subject_category = current_subject_info.get("category", "General")
subject_desc = current_subject_info.get("description", "")

st.markdown(f'<div class="main-header">📚 {subject_title}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Category: <b>{subject_category}</b> &nbsp;|&nbsp; <i>{subject_desc}</i></div>', unsafe_allow_html=True)

# Warning & Quick Upload if subject not indexed yet
is_indexed = selected_subject_id in indexed_subjects if vector_meta else False

with st.expander(
    f"📤 Upload Course Notes / Slides for {subject_title}" if is_indexed else f"⚠️ No Notes Indexed Yet — Click to Upload for {subject_title}",
    expanded=not is_indexed
):
    if not is_indexed:
        st.info(f"💡 No course materials are currently indexed for **{subject_title}**. Drop your lecture slides (PPTX), textbooks (PDF), or notes (DOCX/TXT) below to start asking questions immediately!")
    else:
        st.markdown(f"Add extra PDFs, PPTX presentations, or documents directly to **{subject_title}**.")

    main_uploaded_files = st.file_uploader(
        f"Choose files for {subject_title}:",
        type=["pdf", "pptx", "docx", "txt"],
        accept_multiple_files=True,
        key=f"main_uploader_{selected_subject_id}"
    )
    if main_uploaded_files:
        if st.button(f"⚡ Index Material for {subject_title}", type="primary", key=f"main_index_btn_{selected_subject_id}"):
            dest_dir = ROOT_DIR / "data" / "subjects" / selected_subject_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            total_chunks = 0
            main_prog = st.progress(0.0)
            main_status = st.empty()

            for idx, uf in enumerate(main_uploaded_files):
                main_status.info(f"Indexing ({idx+1}/{len(main_uploaded_files)}): `{uf.name}`...")
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

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📄 View {len(msg['sources'])} Retrieved Sources & Provenance"):
                for src in msg["sources"]:
                    loc = f"Page {src['page']}" if src.get("page") else f"Slide {src.get('slide')}"
                    title_info = f" — *{src['slide_title']}*" if src.get("slide_title") else ""
                    st.markdown(f"**[{src['source_file']}]** ({loc}{title_info}) &nbsp; `Distance: {src['distance']:.4f}`")
                    st.text(src["text"][:350] + ("..." if len(src["text"]) > 350 else ""))
                    st.markdown("---")

# User Query Input
user_question = st.chat_input(f"Ask a question about {subject_title}...")

if user_question:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant response with real-time token streaming
    with st.chat_message("assistant"):
        search_subj = None if cross_subject else selected_subject_id

        # 1. Retrieve & generate response (streaming or fallback)
        t_start = time.time()
        if hasattr(pipeline, "generate_response_stream"):
            stream_generator, sources, display_subj = pipeline.generate_response_stream(
                query=user_question,
                subject_id=search_subj,
                top_k=top_k,
                allow_general_knowledge=allow_gen_knowledge
            )
            # Stream tokens in real time directly to the UI
            full_response = st.write_stream(stream_generator)
        else:
            resp_data = pipeline.generate_response(
                query=user_question,
                subject_id=search_subj,
                top_k=top_k,
                allow_general_knowledge=allow_gen_knowledge
            )
            full_response = resp_data["answer"]
            sources = resp_data.get("sources", [])
            st.markdown(full_response)
        elapsed_sec = time.time() - t_start

        # 3. Display sources accordion
        if sources:
            with st.expander(f"📄 View {len(sources)} Retrieved Sources & Provenance ({elapsed_sec:.1f}s)"):
                for src in sources:
                    loc = f"Page {src['page']}" if src.get("page") else f"Slide {src.get('slide')}"
                    title_info = f" — *{src['slide_title']}*" if src.get("slide_title") else ""
                    st.markdown(f"**[{src['source_file']}]** ({loc}{title_info}) &nbsp; `Distance: {src['distance']:.4f}`")
                    st.text(src["text"][:350] + ("..." if len(src["text"]) > 350 else ""))
                    st.markdown("---")

        # 4. Append assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
