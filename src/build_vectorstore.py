from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_processor import load_pdf, create_chunks


# -----------------------------
# 1. Find all PDF files
# -----------------------------

PDF_DIRECTORY = Path("data/documents")
VECTORSTORE_DIR = Path("vectorstore")

pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))

print("PDF files found:", len(pdf_files))


# -----------------------------
# 2. Load and chunk all PDFs
# -----------------------------

all_chunks = []

for pdf_file in pdf_files:

    print(f"\nProcessing: {pdf_file.name}")

    pages = load_pdf(str(pdf_file))

    chunks = create_chunks(pages)

    print("Chunks:", len(chunks))

    all_chunks.extend(chunks)


print("\nTotal chunks:", len(all_chunks))


# -----------------------------
# 3. Create embeddings
# -----------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    chunk["text"]
    for chunk in all_chunks
]

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings
).astype("float32")


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

print(
    "FAISS vectors:",
    index.ntotal
)


# -----------------------------
# 5. Save vectorstore
# -----------------------------

VECTORSTORE_DIR.mkdir(
    exist_ok=True
)

faiss.write_index(
    index,
    str(VECTORSTORE_DIR / "index.faiss")
)

with open(
    VECTORSTORE_DIR / "chunks.pkl",
    "wb"
) as file:

    pickle.dump(
        all_chunks,
        file
    )


print("\nVectorstore saved successfully.")