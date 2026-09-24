from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Exploring the VAE",
    "We’ll now take a look at the output from the VAE and each submodel and then see",
    "how the VAE can be used to generate completely new track observations."
]

query = "VAE belongs to which submodel"

text_embeddings = model.encode(texts)
query_embedding = model.encode([query])

similarityes = cosine_similarity(query_embedding,text_embeddings)[0]

for text,score in zip(texts, similarityes):
    print(f"{score:4f}->{text}")