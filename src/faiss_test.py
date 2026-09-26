import faiss
import numpy as np 
from sentence_transformers import SentenceTransformer

texts = [
    "Machine learning learns patterns from data.",
    "Deep learning uses neural networks.",
    "Python is a programming language.",
    "Artificial intelligence allows machines to perform intelligent tasks."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("no.of vectors in FAISS:", index.ntotal)

query = "what is deep learning."

query_embedding = model.encode([query]).astype('float32')

K = 2

distances, indices = index.search(query_embedding, K)

print("\nSearch results:")
for distance,index_number in zip(distances[0],indices[0]):
    print(
        f"index:{index_number} |"
        f"Distance:{distance:.4f} |"
        f"Text:{texts[index_number]}"
    )