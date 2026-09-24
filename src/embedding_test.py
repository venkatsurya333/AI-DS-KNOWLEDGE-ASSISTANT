from sentence_transformers import SentenceTransformer

# 1. Fixed model name typo
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Wrapped in a list
texts = [
    "Exploring the VAE",
    "We’ll now take a look at the output from the VAE and each submodel and then see",
    "how the VAE can be used to generate completely new track observations."
]

embeddings = model.encode(texts)

print("no.of embeddings:", len(embeddings))       # Outputs: 1
print("Vector size:", len(embeddings[0]))         # Outputs: 384

print("\nFirst Embedding:")
print(embeddings[0])
