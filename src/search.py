"""
search.py - Subject-Filtered Vector Retrieval Engine

Enforces strict subject-wise filtering:
- Searches vector embeddings for relevant chunks
- Isolates results to the user-selected subject
- Preserves document name, page, slide, and text provenance
- Rejects chunks from other subjects unless cross-subject mode is enabled
"""

import os
import sys
import json
import pickle
from datetime import datetime
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


class SubjectVectorSearch:
    def __init__(
        self,
        vectorstore_dir: str = "vectorstore",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.vectorstore_path = ROOT_DIR / vectorstore_dir
        self.index_file = self.vectorstore_path / "index.faiss"
        self.chunks_file = self.vectorstore_path / "chunks.pkl"
        self.meta_file = self.vectorstore_path / "metadata.json"
        self.embedding_model_name = embedding_model_name

        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

        # Load embedding model (prefer local cache to avoid HuggingFace ping)
        try:
            self.model = SentenceTransformer(embedding_model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(embedding_model_name)

        # Load FAISS index and chunks
        if self.index_file.exists() and self.chunks_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            dim = self.model.get_sentence_embedding_dimension() or 384
            self.index = faiss.IndexFlatL2(dim)
            self.chunks = []

        # Index chunk positions by subject for fast filtering
        self.subject_indices = {}
        for idx, chunk in enumerate(self.chunks):
            s_id = chunk.get("subject_id", "unknown")
            if s_id not in self.subject_indices:
                self.subject_indices[s_id] = []
            self.subject_indices[s_id].append(idx)

    def add_documents(
        self,
        new_chunks: List[Dict[str, Any]],
        persist: bool = True
    ) -> int:
        """
        Incrementally embeds and adds new document chunks into the live FAISS index
        and optionally persists changes to disk (index.faiss, chunks.pkl, metadata.json).
        Returns the number of chunks added.
        """
        if not new_chunks:
            return 0

        # 1. Embed text chunks
        texts = [chunk["text"] for chunk in new_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        embeddings = np.asarray(embeddings).astype("float32")

        # 2. Add embeddings to FAISS index
        self.index.add(embeddings)

        # 3. Update chunk list and subject indices
        start_idx = len(self.chunks)
        self.chunks.extend(new_chunks)
        for i, chunk in enumerate(new_chunks):
            s_id = chunk.get("subject_id", "unknown")
            if s_id not in self.subject_indices:
                self.subject_indices[s_id] = []
            self.subject_indices[s_id].append(start_idx + i)

        # 4. Persist to disk
        if persist:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.chunks_file, "wb") as f:
                pickle.dump(self.chunks, f)

            meta = {}
            if self.meta_file.exists():
                try:
                    with open(self.meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}

            meta["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta["total_chunks"] = len(self.chunks)
            meta["embedding_model"] = self.embedding_model_name
            meta["dimension"] = int(self.index.d)
            if "subjects" not in meta:
                meta["subjects"] = {}

            for chunk in new_chunks:
                s_id = chunk.get("subject_id", "unknown")
                s_title = chunk.get("subject_title", s_id)
                if s_id not in meta["subjects"]:
                    meta["subjects"][s_id] = {
                        "title": s_title,
                        "files_count": 0,
                        "chunks_count": 0
                    }
                meta["subjects"][s_id]["chunks_count"] += 1

            for s_id in meta["subjects"]:
                subj_files = {
                    c.get("source_file")
                    for c in self.chunks
                    if c.get("subject_id") == s_id and c.get("source_file")
                }
                meta["subjects"][s_id]["files_count"] = len(subj_files)

            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return len(new_chunks)

    def get_stats(self) -> Dict[str, Any]:
        """Returns current index statistics."""
        return {
            "total_chunks": len(self.chunks),
            "faiss_total": int(self.index.ntotal),
            "subjects_count": len(self.subject_indices),
            "indexed_subjects": list(self.subject_indices.keys())
        }

    def search(
        self,
        query: str,
        subject_id: Optional[str] = None,
        top_k: int = 5,
        max_candidates: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Executes query embedding and retrieves top-k chunks.
        If subject_id is specified (and not 'all'), strictly filters chunks to that subject.
        """
        if not query or not query.strip():
            return []

        # Convert question to embedding
        query_embedding = self.model.encode([query])
        query_embedding = np.asarray(query_embedding).astype("float32")

        # Determine how many candidates to search in FAISS
        total_vectors = self.index.ntotal
        search_k = min(total_vectors, max_candidates if subject_id and subject_id != "all" else top_k)

        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            # Enforce subject filter
            if subject_id and subject_id != "all":
                if chunk.get("subject_id") != subject_id:
                    continue

            # Support both 'source_file' and legacy 'source'
            src_file = chunk.get("source_file") or Path(chunk.get("source", "")).name

            results.append({
                "rank": len(results) + 1,
                "distance": float(dist),
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "subject_id": chunk.get("subject_id", "general"),
                "subject_title": chunk.get("subject_title", "Course Material"),
                "source_file": src_file,
                "file_type": chunk.get("file_type", "pdf"),
                "page": chunk.get("page"),
                "slide": chunk.get("slide"),
                "slide_title": chunk.get("slide_title"),
                "text": chunk.get("text", "")
            })

            if len(results) >= top_k:
                break

        return results


# Global singleton cache for performance
_search_engine_instance = None


def get_search_engine() -> SubjectVectorSearch:
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = SubjectVectorSearch()
    return _search_engine_instance


if __name__ == "__main__":
    engine = SubjectVectorSearch()
    sample_subj = "operating_systems"
    sample_query = "What is a process control block?"
    print(f"Searching for '{sample_query}' in subject '{sample_subj}'...")
    matches = engine.search(sample_query, subject_id=sample_subj, top_k=3)
    for m in matches:
        loc = f"Page {m['page']}" if m.get('page') else f"Slide {m.get('slide')}"
        print(f"\n[Rank {m['rank']}] Distance: {m['distance']:.4f} | {m['source_file']} ({loc})")
        print(m['text'][:200])
