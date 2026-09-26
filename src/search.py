import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load saved FAISS index
index = faiss.read_index("vectorstore/index.faiss")


# 2. Load chunks and metadata
with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. User question
query = "What is a Large Language Model?"


# 5. Convert question into embedding
query_embedding = model.encode([query])
query_embedding = np.asarray(query_embedding).astype("float32")


# 6. Search FAISS
k = 5

distances, indices = index.search(query_embedding, k)


# 7. Display results
for rank, (distance, index_number) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):
    chunk = chunks[index_number]

    print(f"\n--- Result {rank} ---")
    print(f"Distance: {distance:.4f}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Text:\n{chunk['text'][:500]}")