import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# -----------------------------
# 1. Load environment variables
# -----------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# -----------------------------
# 2. Load FAISS vectorstore
# -----------------------------

index = faiss.read_index(
    "vectorstore/index.faiss"
)

with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


# -----------------------------
# 3. Load embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Ask a question
# -----------------------------

query = "What is a Variational Autoencoder?"


# -----------------------------
# 5. Convert question to vector
# -----------------------------

query_embedding = embedding_model.encode([query])

query_embedding = np.asarray(
    query_embedding
).astype("float32")


# -----------------------------
# 6. Search FAISS
# -----------------------------

k = 5

distances, indices = index.search(
    query_embedding,
    k
)


# -----------------------------
# 7. Collect retrieved chunks
# -----------------------------

retrieved_chunks = []

for index_number in indices[0]:
    retrieved_chunks.append(
        chunks[index_number]
    )


# -----------------------------
# 8. Build context
# -----------------------------

context_parts = []

for rank, chunk in enumerate(retrieved_chunks, start=1):

    source_name = os.path.basename(chunk["source"])

    context_parts.append(
        f"""
[Source {rank}]
Document: {source_name}
Page: {chunk['page']}

Text:
{chunk['text']}
"""
    )

context = "\n\n".join(context_parts)


# -----------------------------
# 9. Create RAG prompt
# -----------------------------

prompt = f"""
You are an AI assistant answering questions about the provided documents.

Use ONLY the provided context to answer the question.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I could not find the answer in the provided documents."
3. Give a clear and concise answer.
4. At the end, list the supporting sources using document name and page number.

Context:
{context}

Question:
{query}
"""


# -----------------------------
# 10. Ask Gemini
# -----------------------------

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


# -----------------------------
# 11. Display answer
# -----------------------------

print("\n========== ANSWER ==========\n")
print(response.text)


# -----------------------------
# 12. Display sources
# -----------------------------

print("\n========== SOURCES ==========\n")

for rank, chunk in enumerate(
    retrieved_chunks,
    start=1
):
    print(
        f"{rank}. Page {chunk['page']}"
    )