from sentence_transformers import SentenceTransformer
from pdf_processor import load_pdf,create_chunks

pdf_path = "data/documents/David Foster - Generative Deep Learning.pdf"

#Load pdf
pages = load_pdf(pdf_path)

#cerate chunks
chunks = create_chunks(pages)

print("no.of chunks",len(chunks))

#Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Create embeddings
texts = [chunk["text"] for chunk in chunks]

#create embeddings
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

print("no.of embeddings:", len(embeddings))
print("Embedding dimenstion:", embeddings.shape[1])