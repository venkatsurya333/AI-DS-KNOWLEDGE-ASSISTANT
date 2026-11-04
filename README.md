# Subject-Wise Academic RAG Assistant

An AI-powered academic assistant for university students. Allows students to query their course materials (lecture slides, notes, textbooks) on a per-subject basis, retrieving grounded explanations with exact PDF page and PPTX slide citations.

---

## 1. Key Capabilities & Design

- **Purely Subject-Wise (No Semester Segregation)**:
  Students select their course directly from a canonical 40-subject academic catalog. Retrieval is strictly isolated to the chosen subject (with an optional cross-subject search toggle).
- **Multi-Format Ingestion with Granular Citations**:
  - **PDFs**: Extracted via PyMuPDF with 1-indexed page numbers.
  - **PPTXs**: Extracted via `python-pptx` with slide numbers and slide titles.
  - **DOCX / TXT**: Paragraph and document-level metadata tracking.
  - **Live Dynamic File Upload**: Upload course notes/slides directly through the web UI and query them immediately without rebuilding or restarting.
- **Modern Executive User Interface**:
  - **Glassmorphic Hero & KPI Strip**: Displays active subject domain, indexed document counts, knowledge vectors, and AI reasoning engine.
  - **Clickable Suggested Questions**: 1-click prompt starter chips for instant conceptual and exam review queries.
  - **Real-Time Token Streaming**: Instant response output with sub-second time-to-first-token.
  - **Rich Source Citations**: Format badges (`[PDF]`, `[PPTX]`), exact slide/page locations, and relevance match scores.
  - **Study Notes Export**: Download full Q&A sessions with verified citations as formatted Markdown (`.md`) files.
- **Dual-Mode Gemini Grounding**:
  - Grounded claims cite exact files and page/slide numbers (e.g. `[3.2 Deadlock.pptx, Slide 5]`, `[2.8 Thread Management.pdf, Page 1]`).
  - Missing course information is clearly acknowledged (*"The uploaded course material does not contain..."*).
  - Outside general AI knowledge is strictly separated under a distinct heading.
- **Reproducible Evaluation Benchmark**:
  Multi-subject test suite measuring Top-k retrieval accuracy, subject filter isolation (0% cross-subject leakage), and citation completeness.

---

## 2. Directory Structure

```text
GenAI Project/
├── config/
│   └── subjects_catalog.json          # Canonical 40-subject registry and descriptions
├── data/
│   └── subjects/                      # Clean subject-wise document folders
│       ├── generative_ai/
│       ├── operating_systems/
│       ├── computer_networks/
│       ├── database_management_systems/
│       ├── deep_learning/
│       ├── data_structures/
│       └── ...
├── src/
│   ├── document_loader.py             # Multi-format extractor (PDF, PPTX, DOCX) & chunker
│   ├── build_vectorstore.py           # Embeddings generator & FAISS indexer
│   ├── search.py                      # Subject-filtered vector retrieval engine
│   └── rag_pipeline.py                # End-to-end Gemini generation with citations
├── evaluation/
│   ├── retrieval_questions.json       # Benchmark questions & expected sources
│   ├── evaluate_retrieval.py          # Benchmark runner
│   └── retrieval_report.json          # Benchmark evaluation outputs
├── vectorstore/
│   ├── index.faiss                    # FAISS IndexFlatL2
│   ├── chunks.pkl                     # Chunks metadata store
│   └── metadata.json                  # Index statistics
├── app.py                             # Streamlit Web Application
├── run_app.bat                        # One-click launcher
├── requirements.txt
└── .env
```

---

## 3. Evaluation Results

Run evaluation via:
```bash
python evaluation/evaluate_retrieval.py
```

### Benchmark Summary (Current Index)
- **Top-1 Retrieval Accuracy**: **87.5%**
- **Top-3 Retrieval Accuracy**: **100.0%**
- **Top-5 Retrieval Accuracy**: **100.0%**
- **Subject Filter Isolation**: **100.0%** (zero cross-subject leakage)
- **Citation Provenance Completeness**: **96.0%** (chunks carry verified page/slide provenance)

---

## 4. Running the Web Application

Launch the Streamlit interactive assistant:
```bash
streamlit run app.py
```
Or double-click `run_app.bat` on Windows.
