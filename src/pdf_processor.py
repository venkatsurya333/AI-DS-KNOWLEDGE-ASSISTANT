import pymupdf as fitz
#from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_no,page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text":text,
                "source":pdf_path,
                "page":page_no+1
            })
    doc.close()
    return pages

def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50 
    )
    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk in page_chunks:
            chunks.append({
                "text":chunk,
                "source":page["source"],
                "page":page["page"]
            })
    return chunks

pdf_path = "data/documents/David Foster - Generative Deep Learning.pdf"

pages  =load_pdf(pdf_path)
chunks = create_chunks(pages)

print("pages:",len(pages))
print("chunks:",len(chunks))

for i,chunk in enumerate(chunks[:5]):
    print(f"\n----CHUNK {i+1} ----")
    print("Source:",chunk['source'])
    print("pages:",chunk["page"])
    print("Text:",chunk['text'][:300])
