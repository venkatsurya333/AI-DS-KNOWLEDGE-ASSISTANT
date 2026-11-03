"""
document_loader.py - Multi-format Document Ingestion Engine

Extracts clean text and granular source citations from:
- PDF files (PyMuPDF) -> page number metadata
- PPTX files (python-pptx) -> slide number & slide title metadata
- DOCX files (python-docx) -> paragraph/section metadata
- TXT files -> line/file metadata

Generates chunked representations preserving full academic metadata:
- subject_id
- subject_title
- source_file
- file_type
- page (for PDFs)
- slide & slide_title (for PPTXs)
- chunk_id
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf  # Fast PyMuPDF import
from pptx import Presentation

try:
    import docx
except ImportError:
    docx = None


def load_pdf(pdf_path: str, subject_id: str, subject_title: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF.
    Preserves exact 1-indexed page numbers.
    """
    file_name = Path(pdf_path).name
    pages = []

    try:
        doc = pymupdf.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "subject_id": subject_id,
                    "subject_title": subject_title,
                    "source_file": file_name,
                    "file_path": str(pdf_path),
                    "file_type": "pdf",
                    "page": page_num + 1,
                    "slide": None,
                    "slide_title": None,
                })
        doc.close()
    except Exception as e:
        print(f"[Warning] Error reading PDF {pdf_path}: {e}")

    return pages


def load_pptx(pptx_path: str, subject_id: str, subject_title: str) -> List[Dict[str, Any]]:
    """
    Extracts text slide-by-slide from a PPTX using python-pptx.
    Preserves exact 1-indexed slide numbers and slide titles.
    """
    file_name = Path(pptx_path).name
    slides = []

    try:
        prs = Presentation(pptx_path)
        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_title = ""
            try:
                if slide.shapes.title and slide.shapes.title.text:
                    slide_title = slide.shapes.title.text.strip()
            except Exception:
                pass

            slide_texts = []
            if slide_title:
                slide_texts.append(f"Title: {slide_title}")

            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        p_text = paragraph.text.strip()
                        if p_text and p_text != slide_title:
                            slide_texts.append(p_text)
                elif shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            slide_texts.append(row_text)

            full_text = "\n".join(slide_texts).strip()
            if full_text:
                slides.append({
                    "text": full_text,
                    "subject_id": subject_id,
                    "subject_title": subject_title,
                    "source_file": file_name,
                    "file_path": str(pptx_path),
                    "file_type": "pptx",
                    "page": None,
                    "slide": slide_num,
                    "slide_title": slide_title or None,
                })
    except Exception as e:
        print(f"[Warning] Error reading PPTX {pptx_path}: {e}")

    return slides


def load_docx(docx_path: str, subject_id: str, subject_title: str) -> List[Dict[str, Any]]:
    """
    Extracts text from DOCX files using python-docx.
    """
    if docx is None:
        return []

    file_name = Path(docx_path).name
    docs = []

    try:
        doc = docx.Document(docx_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs).strip()

        if full_text:
            docs.append({
                "text": full_text,
                "subject_id": subject_id,
                "subject_title": subject_title,
                "source_file": file_name,
                "file_path": str(docx_path),
                "file_type": "docx",
                "page": None,
                "slide": None,
                "slide_title": None,
            })
    except Exception as e:
        print(f"[Warning] Error reading DOCX {docx_path}: {e}")

    return docs


def load_txt(txt_path: str, subject_id: str, subject_title: str) -> List[Dict[str, Any]]:
    """
    Extracts text from plain text files.
    """
    file_name = Path(txt_path).name
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            return [{
                "text": text,
                "subject_id": subject_id,
                "subject_title": subject_title,
                "source_file": file_name,
                "file_path": str(txt_path),
                "file_type": "txt",
                "page": None,
                "slide": None,
                "slide_title": None,
            }]
    except Exception as e:
        print(f"[Warning] Error reading TXT {txt_path}: {e}")
    return []


def load_document(file_path: str, subject_id: str, subject_title: str) -> List[Dict[str, Any]]:
    """
    Dispatches to appropriate loader based on file extension.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path, subject_id, subject_title)
    elif ext == ".pptx":
        return load_pptx(file_path, subject_id, subject_title)
    elif ext == ".docx":
        return load_docx(file_path, subject_id, subject_title)
    elif ext == ".txt":
        return load_txt(file_path, subject_id, subject_title)
    else:
        return []


def recursive_split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Pure Python recursive character text splitter.
    Splits along natural boundaries (paragraphs, lines, words, chars)
    ensuring each chunk is within chunk_size with chunk_overlap.
    Zero external heavy dependencies, instant execution.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the first separator present in text
    separator = separators[-1]
    new_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = ""
            break
        if sep in text:
            separator = sep
            new_separators = separators[i + 1:]
            break

    # Split text using separator
    splits = text.split(separator) if separator else list(text)

    # Merge splits into chunks of size <= chunk_size
    chunks = []
    current_chunk = []
    current_len = 0

    for piece in splits:
        piece_len = len(piece) + (len(separator) if current_chunk else 0)

        if piece_len > chunk_size:
            # If current chunk has content, flush it
            if current_chunk:
                merged = separator.join(current_chunk).strip()
                if merged:
                    chunks.append(merged)
                current_chunk = []
                current_len = 0

            # Sub-split the oversized piece using remaining separators
            if new_separators:
                sub_chunks = recursive_split_text(piece, chunk_size, chunk_overlap, new_separators)
                chunks.extend(sub_chunks)
            else:
                # Hard cut if no more separators
                for start in range(0, len(piece), chunk_size - chunk_overlap):
                    chunks.append(piece[start:start + chunk_size])
            continue

        if current_len + piece_len > chunk_size:
            # Current chunk is full
            merged = separator.join(current_chunk).strip()
            if merged:
                chunks.append(merged)

            # Keep overlapping tokens
            overlap_pieces = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) <= chunk_overlap:
                    overlap_pieces.insert(0, p)
                    overlap_len += len(p) + len(separator)
                else:
                    break

            current_chunk = overlap_pieces + [piece]
            current_len = sum(len(p) for p in current_chunk) + len(separator) * (len(current_chunk) - 1)
        else:
            current_chunk.append(piece)
            current_len += piece_len

    if current_chunk:
        merged = separator.join(current_chunk).strip()
        if merged:
            chunks.append(merged)

    return chunks


def create_chunks(
    documents: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Splits documents into overlapping chunks preserving all provenance metadata
    for fine-grained citations.
    """
    chunks = []
    for doc in documents:
        split_texts = recursive_split_text(
            doc["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        for idx, text in enumerate(split_texts, start=1):
            loc_tag = (
                f"p{doc['page']}" if doc["page"] is not None
                else f"s{doc['slide']}" if doc["slide"] is not None
                else "sec"
            )
            # Safe chunk_id without spaces
            safe_source = doc["source_file"].replace(" ", "_")
            chunk_id = f"{doc['subject_id']}_{safe_source}_{loc_tag}_c{idx}"

            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "subject_id": doc["subject_id"],
                "subject_title": doc["subject_title"],
                "source_file": doc["source_file"],
                "file_path": doc["file_path"],
                "file_type": doc["file_type"],
                "page": doc["page"],
                "slide": doc["slide"],
                "slide_title": doc["slide_title"],
            })

    return chunks
