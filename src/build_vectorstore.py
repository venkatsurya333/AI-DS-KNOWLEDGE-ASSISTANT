"""
build_vectorstore.py - Vector Database Generator with Subject-Aware Indexing

Extracts documents from data/subjects/, generates chunks with rich academic metadata,
computes 384-dimensional embeddings using all-MiniLM-L6-v2, and builds a FAISS index.
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure standard output supports UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.document_loader import load_document, create_chunks


def build_vectorstore(
    subjects_to_index: Optional[List[str]] = None,
    catalog_path: str = "config/subjects_catalog.json",
    data_dir: str = "data/subjects",
    output_dir: str = "vectorstore",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Builds or updates the FAISS vectorstore for specified subjects.
    If subjects_to_index is None, indexes all subjects marked with is_core: true.
    """
    start_time = time.time()
    catalog_file = ROOT_DIR / catalog_path
    subjects_root = ROOT_DIR / data_dir
    vectorstore_path = ROOT_DIR / output_dir

    if not catalog_file.exists():
        raise FileNotFoundError(f"Subjects catalog not found at {catalog_file}")

    with open(catalog_file, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Determine subjects to index
    if subjects_to_index:
        target_subjects = {s: catalog[s] for s in subjects_to_index if s in catalog}
    else:
        target_subjects = {s: info for s, info in catalog.items() if info.get("is_core", False)}

    print("=" * 60)
    print(f"BUILDING VECTORSTORE FOR {len(target_subjects)} SUBJECTS")
    print("=" * 60)

    all_chunks = []
    subject_stats = {}

    for subj_id, subj_info in target_subjects.items():
        subj_dir = subjects_root / subj_id
        if not subj_dir.exists():
            print(f"[Skip] Directory not found for {subj_id}: {subj_dir}")
            continue

        subj_title = subj_info.get("title", subj_id)
        files = [
            f for f in subj_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".pptx", ".docx", ".txt"}
        ]

        print(f"\n[*] Processing Subject: '{subj_title}' ({subj_id})")
        print(f"    Found {len(files)} document files")

        subj_chunks = []
        for file_path in files:
            try:
                docs = load_document(str(file_path), subj_id, subj_title)
                chunks = create_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                subj_chunks.extend(chunks)
                print(f"    - {file_path.name[:45]:45s} -> {len(docs):3d} units, {len(chunks):3d} chunks")
            except Exception as e:
                print(f"    [Error] reading {file_path.name}: {e}")

        all_chunks.extend(subj_chunks)
        subject_stats[subj_id] = {
            "title": subj_title,
            "files_count": len(files),
            "chunks_count": len(subj_chunks)
        }
        print(f"    -> Total chunks for {subj_id}: {len(subj_chunks)}")

    print("\n" + "=" * 60)
    print(f"Total Chunks Across All Target Subjects: {len(all_chunks)}")
    print("=" * 60)

    if not all_chunks:
        print("[Warning] No chunks generated! Aborting vectorstore build.")
        return {}

    # Encode embeddings
    print(f"\nLoading embedding model: '{embedding_model_name}'...")
    try:
        embedder = SentenceTransformer(embedding_model_name, local_files_only=True)
    except Exception:
        embedder = SentenceTransformer(embedding_model_name)

    texts = [chunk["text"] for chunk in all_chunks]
    print(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = np.asarray(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    print(f"Embeddings computed: shape={embeddings.shape}, dimension={dimension}")

    # Build FAISS IndexFlatL2
    print("Creating FAISS IndexFlatL2 index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS total indexed vectors: {index.ntotal}")

    # Save to disk
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    faiss_file = vectorstore_path / "index.faiss"
    chunks_file = vectorstore_path / "chunks.pkl"
    meta_file = vectorstore_path / "metadata.json"

    faiss.write_index(index, str(faiss_file))

    with open(chunks_file, "wb") as f:
        pickle.dump(all_chunks, f)

    meta_info = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": len(all_chunks),
        "embedding_model": embedding_model_name,
        "dimension": dimension,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "subjects": subject_stats,
        "build_duration_seconds": round(time.time() - start_time, 2)
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    print(f"\n[OK] Vectorstore saved successfully to {vectorstore_path}")
    print(f"     - Index:  {faiss_file.name} ({index.ntotal} vectors)")
    print(f"     - Chunks: {chunks_file.name} ({len(all_chunks)} chunks)")
    print(f"     - Meta:   {meta_file.name}")
    print(f"     - Time:   {meta_info['build_duration_seconds']}s")

    return meta_info


if __name__ == "__main__":
    subjs = sys.argv[1:] if len(sys.argv) > 1 else None
    build_vectorstore(subjs)
