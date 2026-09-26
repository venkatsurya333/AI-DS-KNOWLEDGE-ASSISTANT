import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. File paths
# -----------------------------

INDEX_PATH = "vectorstore/index.faiss"
CHUNKS_PATH = "vectorstore/chunks.pkl"
QUESTIONS_PATH = "evaluation/retrieval_questions.json"


# -----------------------------
# 2. Source mapping
# -----------------------------

SOURCE_MAP = {
    "foster": "david foster - generative deep learning.pdf",
    "llm": "hands-on large language models"
}


def get_source_id(source_path):
    filename = Path(source_path).name.lower()

    if "david foster" in filename:
        return "foster"

    if "hands-on large language models" in filename:
        return "llm"

    return "unknown"


# -----------------------------
# 3. Load vectorstore
# -----------------------------

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)


# -----------------------------
# 4. Load questions
# -----------------------------

with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)


# -----------------------------
# 5. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 6. Evaluation
# -----------------------------

top1_correct = 0
top3_correct = 0
top5_correct = 0

print("=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)


for number, item in enumerate(questions, start=1):

    question = item["question"]
    expected_source = item["expected_source"]

    query_embedding = model.encode([question])
    query_embedding = np.asarray(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, 5)

    retrieved_sources = []

    for index_number in indices[0]:
        source_id = get_source_id(chunks[index_number]["source"])
        retrieved_sources.append(source_id)

    top1 = retrieved_sources[:1]
    top3 = retrieved_sources[:3]
    top5 = retrieved_sources[:5]

    if expected_source not in top5:
        print("\n❌ FAILED RETRIEVAL")
        print("Expected:", expected_source)
        print("Retrieved:", retrieved_sources)

        for rank, index_number in enumerate(indices[0], start=1):
            chunk = chunks[index_number]
            print(f"\nRank {rank}")
            print("Source:", chunk["source"])
            print("Page:", chunk["page"])
            print("Text:", chunk["text"][:300])

    if expected_source in top1:
        top1_correct += 1

    if expected_source in top3:
        top3_correct += 1

    if expected_source in top5:
        top5_correct += 1

    print(f"\nQuestion {number}")
    print(question)

    print(f"Expected: {expected_source}")
    print(f"Top 5:   {retrieved_sources}")

    print(
        f"Top-1: {'PASS' if expected_source in top1 else 'FAIL'} | "
        f"Top-3: {'PASS' if expected_source in top3 else 'FAIL'} | "
        f"Top-5: {'PASS' if expected_source in top5 else 'FAIL'}"
    )


# -----------------------------
# 7. Final results
# -----------------------------

total = len(questions)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Top-1 Accuracy: {top1_correct / total:.2%}")
print(f"Top-3 Accuracy: {top3_correct / total:.2%}")
print(f"Top-5 Accuracy: {top5_correct / total:.2%}")