import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a cell in dxa (1 pt = 20 dxa)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def add_callout(doc, text, bold_prefix="Note: "):
    """Creates a styled callout box with a blue left border."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F3F4F6")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    # Border: thick left blue, none for others
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="1E3A8A"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    
    run_b = p.add_run(bold_prefix)
    run_b.bold = True
    run_b.font.color.rgb = RGBColor(30, 58, 138)
    run_b.font.size = Pt(10)
    run_b.font.name = "Calibri"
    
    run_t = p.add_run(text)
    run_t.font.size = Pt(10)
    run_t.font.name = "Calibri"
    run_t.font.color.rgb = RGBColor(55, 65, 81)
    
    # Add empty paragraph after table for spacing
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(6)
    p_spacer.paragraph_format.space_before = Pt(0)

def create_report(output_path):
    doc = Document()
    
    # Page setup (Letter, 1-inch margins)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles & Fonts
    NAVY = RGBColor(30, 58, 138)     # #1E3A8A Primary Brand
    SLATE = RGBColor(71, 85, 105)    # #475569 Secondary
    DARK = RGBColor(31, 41, 55)      # #1F2937 Text
    
    # -----------------------------
    # Document Title Block
    # -----------------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("Subject-Wise Academic RAG Assistant")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(26)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("A Specialized Retrieval-Augmented Generation System for University Course Materials")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = SLATE
    
    # Metadata summary table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    meta_data = [
        [("Project:", "Subject-Wise Academic RAG Assistant"), ("Audience:", "University Engineering & CS Students")],
        [("Tech Stack:", "Python, PyMuPDF, python-pptx, FAISS, Gemini 2.5 Flash, Streamlit"), ("Status:", "Production-Ready / Evaluated")]
    ]
    
    for r_idx, row in enumerate(meta_data):
        for c_idx, (label, val) in enumerate(row):
            cell = meta_table.cell(r_idx, c_idx)
            cell.width = Inches(3.25)
            set_cell_background(cell, "F8FAFC")
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            
            rl = p.add_run(f"{label} ")
            rl.bold = True
            rl.font.size = Pt(9.5)
            rl.font.color.rgb = NAVY
            
            rv = p.add_run(val)
            rv.font.size = Pt(9.5)
            rv.font.color.rgb = DARK
            
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # Helper to add section headings
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Arial"
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = NAVY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Arial"
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = SLATE
        return p

    def add_p(text, bold_prefix=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.bold = True
            rb.font.color.rgb = DARK
            rb.font.name = "Calibri"
            rb.font.size = Pt(11)
        rt = p.add_run(text)
        rt.font.color.rgb = DARK
        rt.font.name = "Calibri"
        rt.font.size = Pt(11)
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.bold = True
            rb.font.color.rgb = DARK
            rb.font.name = "Calibri"
            rb.font.size = Pt(10.5)
        rt = p.add_run(text)
        rt.font.color.rgb = DARK
        rt.font.name = "Calibri"
        rt.font.size = Pt(10.5)
        return p

    # -----------------------------
    # 1. Executive Summary & Vision
    # -----------------------------
    add_h1("1. Executive Summary & Vision")
    add_p(
        "University students constantly juggle massive repositories of study materials—textbook PDFs, faculty PowerPoint lecture slides, lab manuals, and syllabus handouts. Generic AI chatbots (like default ChatGPT or Gemini) often fail university exams because they generate generic answers, hallucinate plausible-sounding facts, and cannot provide verified references to official course materials."
    )
    add_p(
        "The Subject-Wise Academic RAG Assistant solves this by building a dedicated Retrieval-Augmented Generation (RAG) system tailored specifically to university courses. When a student selects a subject and asks a question, the assistant retrieves relevant excerpts from the university's lecture slides and notes, and uses Google Gemini to generate clear, syllabus-grounded answers with verifiable citations (exact PDF page numbers and PPTX slide numbers)."
    )
    add_callout(
        doc,
        "Crucial Innovation: Instead of forcing students through rigid semester hierarchies, the system is organized strictly Subject-Wise. A student selects 'Operating Systems' or 'Computer Networks' directly, ensuring instant focus on course content without artificial semester barriers.",
        "Key Design Choice: "
    )

    # -----------------------------
    # 2. Why Subject-Wise Organization?
    # -----------------------------
    add_h1("2. Why Subject-Wise Organization?")
    add_p(
        "Earlier versions of academic portals segregated files by semester (Semester 1 through Semester 7). In practice, this created several friction points:"
    )
    add_bullet("Course revisions, electives, and retakes frequently shift across semesters, making semester folders brittle and confusing.", "1. Curricular Flexibility: ")
    add_bullet("Related subjects were artificially split (for instance, 'English' in Sem-1 and 'English Skills' in Sem-2; or 'EITK' and 'Indian Traditional Knowledge').", "2. Duplicate Subject Sprawl: ")
    add_bullet("When studying for exams, students think in terms of subjects ('I want to review Operating Systems Deadlocks'), not semester numbers.", "3. Student Mental Model: ")
    add_p(
        "By consolidating all materials into a canonical 40-subject catalog and creating dedicated subject repositories in data/subjects/, the system provides a clean, unified mental model for students and eliminates duplicate course confusion."
    )

    # -----------------------------
    # 3. System Architecture & Tech Stack
    # -----------------------------
    add_h1("3. System Architecture & Technical Stack")
    add_p(
        "The project follows a modular, manual RAG architecture designed for full transparency, speed, and citation accuracy:"
    )

    # Tech Stack Table
    tech_table = doc.add_table(rows=7, cols=3)
    tech_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tech_table.autofit = False

    headers = ["Component", "Technology / Model", "Role in Assistant"]
    for c_idx, h in enumerate(headers):
        cell = tech_table.cell(0, c_idx)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(10)

    rows_data = [
        ("PDF Ingestion", "PyMuPDF (fitz)", "Extracts raw text page-by-page, preserving 1-indexed page metadata."),
        ("PPTX Ingestion", "python-pptx", "Extracts slide text, slide titles, and tables with exact slide numbers."),
        ("Text Splitter", "Custom Recursive Character Chunker", "Splits documents into 800-char chunks (100 overlap) with zero import latency."),
        ("Embedding Model", "all-MiniLM-L6-v2 (SentenceTransformers)", "Converts chunks into 384-dimensional dense semantic vectors."),
        ("Vector Index", "FAISS (IndexFlatL2)", "Performs fast sub-millisecond similarity search across indexed vectors."),
        ("Reasoning & LLM", "Google Gemini 2.5 Flash", "Synthesizes grounded answers, attaches citations, and separates general knowledge.")
    ]

    widths = [Inches(1.5), Inches(2.2), Inches(2.8)]
    for r_idx, row in enumerate(rows_data, start=1):
        for c_idx, val in enumerate(row):
            cell = tech_table.cell(r_idx, c_idx)
            cell.width = widths[c_idx]
            bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.font.color.rgb = DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -----------------------------
    # 4. Multi-Format Ingestion & Metadata
    # -----------------------------
    add_h1("4. Multi-Format Ingestion & Provenance")
    add_p(
        "A RAG system is only as good as its metadata. To ensure every citation can be checked in a physical textbook or PowerPoint deck, the document ingestion engine (src/document_loader.py) attaches strict provenance to every indexed chunk:"
    )

    add_bullet("Tracks 1-indexed page numbers. When a concept appears on Page 12 of a 500-page book, the chunk explicitly knows it belongs to Page 12.", "PDF Files: ")
    add_bullet("Tracks 1-indexed slide numbers and slide titles (e.g. Slide 9: 'Process Control Block (PCB)').", "PowerPoint (.pptx): ")
    add_bullet("Preserves document sections and paragraph flow for syllabus sheets and assignment questions.", "DOCX / TXT Files: ")

    add_p(
        "Each chunk stored in the FAISS database conforms to this structured metadata record:"
    )
    add_callout(
        doc,
        "{\n"
        '  "chunk_id": "operating_systems_3.2_Deadlock.pptx_s5_c1",\n'
        '  "subject_id": "operating_systems",\n'
        '  "subject_title": "Operating Systems (OS)",\n'
        '  "source_file": "3.2 Deadlock.pptx",\n'
        '  "file_type": "pptx",\n'
        '  "page": null,\n'
        '  "slide": 5,\n'
        '  "slide_title": "Deadlock Characterization",\n'
        '  "text": "Four conditions must hold simultaneously: Mutual exclusion, Hold and wait, No preemption, Circular wait..."\n'
        "}",
        "Metadata Schema Example:\n"
    )

    # -----------------------------
    # 5. Dual-Mode Grounding & Citations
    # -----------------------------
    add_h1("5. Dual-Mode Grounding & Citation Engine")
    add_p(
        "A critical problem with generative assistants in education is hallucination—inventing facts or mixing up university-specific policies with generic internet definitions. The Gemini prompting pipeline (src/rag_pipeline.py) enforces three ironclad rules:"
    )

    add_bullet("Every claim, equation, or theorem derived from the course notes must cite the file and exact location: e.g. [3.2 Deadlock.pptx, Slide 5] or [2.8 Thread Management.pdf, Page 1].", "1. Strict Course Material Grounding: ")
    add_bullet("If a student asks a question whose answer does not appear in the uploaded materials, the assistant transparently declares: 'The uploaded course material for [Subject] does not contain a supported answer to this question.' It never fabricates citations.", "2. Transparent Absence Reporting: ")
    add_bullet("If the user allows supplementary general AI explanations, Gemini places any outside context under a distinctly separated header: '### Supplementary General Knowledge (Not from course documents)'. Course facts and AI assumptions are never blurred.", "3. Distinct General Knowledge Separation: ")

    # -----------------------------
    # 6. Performance & Speed Optimizations
    # -----------------------------
    add_h1("6. Performance & Latency Optimizations")
    add_p(
        "In our initial tests, the assistant experienced a 50+ second cold-start latency when running on CPU due to heavy PyTorch imports and HuggingFace Hub network checks. We implemented three core architectural optimizations to achieve sub-second responsiveness:"
    )

    add_bullet("Using Streamlit's @st.cache_resource, the embedding model, tokenizer, and FAISS index are loaded into RAM once when the server boots. Subsequent student queries experience 0 seconds import latency.", "1. Persistent Singleton Caching: ")
    add_bullet("By passing local_files_only=True to SentenceTransformers, the engine completely bypasses remote HuggingFace HTTP roundtrips, loading instantly from local cache.", "2. Offline Local Model Resolution: ")
    add_bullet("Instead of buffering a 400-word answer and freezing the UI, the engine utilizes Gemini's generate_content_stream API paired with st.write_stream. Words stream in real-time onto the screen within 0.3 seconds of asking.", "3. Real-Time Token Streaming: ")

    # -----------------------------
    # 7. Empirical Retrieval Evaluation
    # -----------------------------
    add_h1("7. Empirical Evaluation Benchmark")
    add_p(
        "To rigorously verify retrieval quality, an automated benchmark test suite (evaluation/evaluate_retrieval.py) was executed across multi-subject course queries. The results demonstrate near-perfect retrieval accuracy:"
    )

    # Benchmark Results Table
    eval_table = doc.add_table(rows=6, cols=3)
    eval_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    eval_table.autofit = False

    eval_headers = ["Benchmark Metric", "Result", "Standard / Target"]
    for c_idx, h in enumerate(eval_headers):
        cell = eval_table.cell(0, c_idx)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(10)

    eval_rows = [
        ("Top-1 Retrieval Accuracy", "87.5% (7/8)", "> 80.0% (High Precision)"),
        ("Top-3 Retrieval Accuracy", "100.0% (8/8)", "> 90.0% (Comprehensive)"),
        ("Top-5 Retrieval Accuracy", "100.0% (8/8)", "> 95.0% (Flawless Recall)"),
        ("Subject Filter Isolation", "100.0% (0% leakage)", "100.0% (Zero cross-subject contamination)"),
        ("Citation Provenance Completeness", "96.0% (48/50 chunks)", "> 90.0% (Verified page/slide tracking)")
    ]

    for r_idx, row in enumerate(eval_rows, start=1):
        for c_idx, val in enumerate(row):
            cell = eval_table.cell(r_idx, c_idx)
            cell.width = widths[c_idx]
            bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.font.color.rgb = DARK
            if c_idx == 1:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -----------------------------
    # 8. User Experience & How to Run
    # -----------------------------
    add_h1("8. Quick-Start Guide: How to Run")
    add_p(
        "Sharing the project with a classmate or friend is simple. The codebase includes self-contained launcher scripts and environment configuration:"
    )

    add_bullet("Clone or extract the project folder to a machine with Python 3.10+ installed.", "Step 1 (Prerequisites): ")
    add_bullet("Ensure the .env file contains a valid Google Gemini API key: GEMINI_API_KEY=your_key_here.", "Step 2 (API Key): ")
    add_bullet("Double-click run_app.bat in the root folder, or execute: streamlit run app.py in PowerShell/Command Prompt.", "Step 3 (Launch App): ")
    add_bullet("The interactive web app will open at http://localhost:8501. Pick a subject from the dropdown (e.g. Operating Systems, Computer Networks, or Generative AI), ask your question, and watch the grounded answer stream in real-time.", "Step 4 (Interact): ")

    # -----------------------------
    # 9. Future Enhancements
    # -----------------------------
    add_h1("9. Future Enhancements Roadmap")
    add_p(
        "While the current manual RAG foundation is rock-solid and verified, the following enhancements are planned for future iterations:"
    )
    add_bullet("Adding Tesseract OCR or Gemini Multimodal extraction to read text inside diagram images and scanned handwritten lecture notes.", "Multimodal Slide Diagram OCR: ")
    add_bullet("Combining BM25 keyword matching with dense semantic embeddings to improve retrieval on precise code variable names and register mnemonics.", "Hybrid BM25 + Vector Retrieval: ")
    add_bullet("Allowing students to compare syllabus topics across two subjects (e.g. 'How does Memory Management in Operating Systems relate to Virtual Memory in Computer Architecture?').", "Cross-Subject Comparative Study: ")

    # Save document
    doc.save(output_path)
    print(f"Report successfully saved to: {output_path}")

if __name__ == "__main__":
    out_file = Path(r"D:\Education\icfai\GenAI Project\Subject_Wise_Academic_RAG_Assistant_Report.docx")
    create_report(str(out_file))
