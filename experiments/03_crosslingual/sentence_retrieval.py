#!/usr/bin/env python3
"""
Sentence-level cross-lingual retrieval using fastText embeddings.
Evaluates whether Portuguese sentence embeddings can retrieve the correct
Nheengatu translation from a held-out pool.
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent  # experiments/
FASTTEXT_DIR = BASE_DIR / "02_fasttext"
RESULTS_DIR = FASTTEXT_DIR / "results"
CORPUS_PATH = FASTTEXT_DIR / "corpus_merged.json"

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load fastText models
import fasttext
pt_model = fasttext.load_model(str(RESULTS_DIR / "model_merged_300d.pt.bin"))
nhe_model = fasttext.load_model(str(RESULTS_DIR / "model_merged.nhe.bin"))

print("Loading merged corpus...")
with open(CORPUS_PATH, encoding="utf-8") as f:
    data = json.load(f)

print(f"Total sentence pairs: {len(data)}")

# Shuffle and split: 80% train, 20% test
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Train pairs: {len(train_data)}")
print(f"Test pairs:  {len(test_data)}")

# ------------------------------------------------------------
# Sentence embedding: average of word vectors
# ------------------------------------------------------------
def sent_embed(text, model):
    words = text.lower().split()
    vectors = [model.get_word_vector(w) for w in words if w in model.words]
    if not vectors:
        return np.zeros(model.get_dimension())
    return np.mean(vectors, axis=0)

print("Computing sentence embeddings for test set...")
test_pt_embs = np.array([sent_embed(p['pt'], pt_model) for p in test_data])
test_nhe_embs = np.array([sent_embed(p['nhe'], nhe_model) for p in test_data])

# Normalize to unit length (cosine similarity)
test_pt_embs = test_pt_embs / np.linalg.norm(test_pt_embs, axis=1, keepdims=True)
test_nhe_embs = test_nhe_embs / np.linalg.norm(test_nhe_embs, axis=1, keepdims=True)

# ------------------------------------------------------------
# Retrieval evaluation
# ------------------------------------------------------------
print("\nEvaluating cross-lingual retrieval...")
precision_1 = 0
precision_5 = 0
precision_10 = 0
mrr_sum = 0
n_queries = len(test_pt_embs)

for i, pt_emb in enumerate(test_pt_embs):
    # Compute cosine similarities to all NHE sentences
    sims = np.dot(test_nhe_embs, pt_emb)  # both normalized, so dot = cosine
    ranked_idx = np.argsort(-sims)
    
    # Check rank of correct NHE sentence (same index i)
    true_rank = np.where(ranked_idx == i)[0][0] + 1  # 1-indexed
    mrr_sum += 1.0 / true_rank
    
    if true_rank == 1:
        precision_1 += 1
    if true_rank <= 5:
        precision_5 += 1
    if true_rank <= 10:
        precision_10 += 1

print(f"\n=== Retrieval Results (Test set of {n_queries} pairs) ===")
print(f"Precision@1:  {precision_1 / n_queries:.4f}")
print(f"Precision@5:  {precision_5 / n_queries:.4f}")
print(f"Precision@10: {precision_10 / n_queries:.4f}")
print(f"MRR:          {mrr_sum / n_queries:.4f}")

# ------------------------------------------------------------
# Qualitative examples
# ------------------------------------------------------------
print("\n=== Example Retrievals ===")
for i in range(min(5, n_queries)):
    pt_emb = test_pt_embs[i]
    sims = np.dot(test_nhe_embs, pt_emb)
    top_idx = np.argmax(sims)
    print(f"\nQuery {i+1}:")
    print(f"  PT:  {test_data[i]['pt'][:100]}...")
    print(f"  True NHE: {test_data[i]['nhe'][:100]}...")
    print(f"  Pred NHE: {test_data[top_idx]['nhe'][:100]}...")
    print(f"  Correct? {top_idx == i}")