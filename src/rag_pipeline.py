"""
rag_pipeline.py - Academic RAG Reasoning & Generation Engine

Integrates SubjectVectorSearch with Gemini to generate grounded,
verifiable academic answers with exact page/slide citations and
strict separation between course materials and general AI knowledge.
Supports both buffered generation and fast real-time token streaming.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator

# Ensure standard output supports UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from google import genai

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Load .env from project root
load_dotenv(dotenv_path=ROOT_DIR / ".env")

from src.search import SubjectVectorSearch, get_search_engine


class AcademicRAGPipeline:
    def __init__(
        self,
        gemini_model: str = "gemini-2.5-flash",
        search_engine: Optional[SubjectVectorSearch] = None
    ):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("[Warning] GEMINI_API_KEY not found in environment or .env file.")

        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.gemini_model = gemini_model
        self.search_engine = search_engine or get_search_engine()

    def build_prompt_and_context(
        self,
        query: str,
        subject_id: Optional[str] = None,
        top_k: int = 5,
        allow_general_knowledge: bool = True
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Retrieves context chunks and builds the grounded academic prompt.
        """
        retrieved_chunks = self.search_engine.search(
            query=query,
            subject_id=subject_id,
            top_k=top_k
        )

        subject_display = subject_id.replace("_", " ").title() if subject_id else "All Subjects"
        if retrieved_chunks and retrieved_chunks[0].get("subject_title"):
            subject_display = retrieved_chunks[0]["subject_title"]

        context_blocks = []
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.get("file_type") == "pdf" or chunk.get("page") is not None:
                loc_str = f"Page {chunk.get('page')}"
            elif chunk.get("file_type") == "pptx" or chunk.get("slide") is not None:
                title_str = f" (Title: {chunk['slide_title']})" if chunk.get('slide_title') else ""
                loc_str = f"Slide {chunk.get('slide')}{title_str}"
            else:
                loc_str = "Document Section"

            context_blocks.append(
                f"[Source {rank}]\n"
                f"Subject: {chunk['subject_title']}\n"
                f"Document: {chunk['source_file']}\n"
                f"Location: {loc_str}\n"
                f"Content:\n{chunk['text']}\n"
            )

        context_text = "\n---\n".join(context_blocks) if context_blocks else "No relevant documents found in course materials."

        gk_instruction = (
            "- You MAY provide supplementary academic background or intuition from your general knowledge, but you MUST place it under a clearly separated heading: '### Supplementary General Knowledge (Not from course documents)'."
            if allow_general_knowledge else
            "- Do NOT use any outside general knowledge. Restrict your answer solely to the provided course excerpts."
        )

        sample_source = retrieved_chunks[0]['source_file'] if retrieved_chunks else 'Document.pdf'

        prompt = f"""You are an authoritative Academic Assistant for university students studying {subject_display}.

You have been provided with excerpts from the university course materials for {subject_display}.

=== RETRIEVED COURSE EXCERPTS ===
{context_text}
=================================

=== STUDENT QUESTION ===
{query}
========================

=== INSTRUCTIONS & GROUNDING RULES ===
1. PRIMARY COURSE MATERIAL ANSWER:
   - Base your core answer strictly on the provided course excerpts.
   - For every factual claim, definition, formula, or concept derived from the course notes, attach an explicit in-text citation pointing to the exact document and page/slide (e.g. `[{sample_source}, Page X]` or `[Slide Y]`).
   - If the retrieved course excerpts do NOT contain information to answer the question, explicitly state:
     "The uploaded course material for {subject_display} does not contain a supported answer to this question."

2. GENERAL AI KNOWLEDGE (Clear Distinction):
   {gk_instruction}
   - NEVER attribute general knowledge statements to the uploaded course files.
   - NEVER fabricate page numbers, slide numbers, or document names.

3. STRUCTURE OF YOUR RESPONSE:
   - **Course Material Answer**: Grounded explanation citing specific files and pages/slides.
   - **Supplementary General Knowledge** (optional/clearly labeled if needed).
   - **Sources Referenced**: Bulleted list of documents with exact page/slide numbers used.
"""
        return prompt, retrieved_chunks, subject_display

    def generate_response(
        self,
        query: str,
        subject_id: Optional[str] = None,
        top_k: int = 5,
        allow_general_knowledge: bool = True
    ) -> Dict[str, Any]:
        """
        Buffered response generation.
        """
        if not query or not query.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "subject_id": subject_id,
                "found_in_materials": False
            }

        prompt, retrieved_chunks, subject_display = self.build_prompt_and_context(
            query=query,
            subject_id=subject_id,
            top_k=top_k,
            allow_general_knowledge=allow_general_knowledge
        )

        if not self.client:
            return {
                "answer": "Error: Gemini client not initialized. Please verify GEMINI_API_KEY.",
                "sources": retrieved_chunks,
                "subject_id": subject_id,
                "found_in_materials": False
            }

        try:
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            answer_text = response.text
        except Exception as e:
            answer_text = f"Error during Gemini generation: {e}"

        return {
            "answer": answer_text,
            "sources": retrieved_chunks,
            "subject_id": subject_id,
            "subject_title": subject_display,
            "query": query
        }

    def generate_response_stream(
        self,
        query: str,
        subject_id: Optional[str] = None,
        top_k: int = 5,
        allow_general_knowledge: bool = True
    ) -> Tuple[Iterator[str], List[Dict[str, Any]], str]:
        """
        Real-time streaming generation. Returns an iterator of text chunks
        along with the retrieved source chunks.
        """
        prompt, retrieved_chunks, subject_display = self.build_prompt_and_context(
            query=query,
            subject_id=subject_id,
            top_k=top_k,
            allow_general_knowledge=allow_general_knowledge
        )

        def stream_generator():
            if not self.client:
                yield "Error: Gemini client not initialized. Please verify GEMINI_API_KEY."
                return
            try:
                response_stream = self.client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=prompt
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield f"\n\n[Error during generation: {e}]"

        return stream_generator(), retrieved_chunks, subject_display


_cached_pipeline: Optional[AcademicRAGPipeline] = None


def get_rag_pipeline(
    gemini_model: str = "gemini-2.5-flash",
    search_engine: Optional[SubjectVectorSearch] = None,
    force_reload: bool = False
) -> AcademicRAGPipeline:
    """Returns a singleton AcademicRAGPipeline instance."""
    global _cached_pipeline
    if _cached_pipeline is None or force_reload:
        _cached_pipeline = AcademicRAGPipeline(
            gemini_model=gemini_model,
            search_engine=search_engine
        )
    return _cached_pipeline


if __name__ == "__main__":
    pipeline = AcademicRAGPipeline()
    test_subj = "computer_networks"
    test_q = "what is ip address"
    print(f"Streaming query: '{test_q}' for Subject: {test_subj}\n")
    stream, sources, subj_title = pipeline.generate_response_stream(test_q, subject_id=test_subj, top_k=3)
    for token in stream:
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n\nRetrieved sources:", len(sources))
