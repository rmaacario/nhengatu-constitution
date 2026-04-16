#!/usr/bin/env python3
"""
Sentence-level cross-lingual retrieval using XLM-RoBERTa embeddings.
Evaluates whether XLM-R sentence embeddings can retrieve the correct
Nheengatu translation from a held-out pool.
"""

import json
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent.parent  # experiments/
FASTTEXT_DIR = BASE_DIR / "02_fasttext"
CORPUS_PATH = FASTTEXT_DIR / "corpus_merged.json"

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load corpus
print("Loading merged corpus...")
with open(CORPUS_PATH, encoding="utf-8") as f:
    data = json.load(f)

print(f"Total sentence pairs: {len(data)}")

# Shuffle and split: 80% train, 20% test (same as before)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Train pairs: {len(train_data)}")
print(f"Test pairs:  {len(test_data)}")

# ------------------------------------------------------------
# XLM-R Embedding
# ------------------------------------------------------------
model_name = "xlm-roberta-base"
print(f"\nLoading XLM-R model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def embed_texts(texts, batch_size=32):
    """Mean-pool XLM-R last hidden state over non-padding tokens."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded)
        # Mean pooling over valid tokens
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
        if i % 200 == 0:
            print(f"  Embedded {i}/{len(texts)}")
    return np.vstack(all_embs)

print("\nComputing XLM-R embeddings for test set...")
print("  Portuguese sentences...")
test_pt_embs = embed_texts([p['pt'] for p in test_data])
print("  Nheengatu sentences...")
test_nhe_embs = embed_texts([p['nhe'] for p in test_data])

# ------------------------------------------------------------
# Retrieval Evaluation
# ------------------------------------------------------------
print("\nEvaluating cross-lingual retrieval...")
precision_1 = 0
precision_5 = 0
precision_10 = 0
mrr_sum = 0
n_queries = len(test_pt_embs)

for i, pt_emb in enumerate(test_pt_embs):
    sims = np.dot(test_nhe_embs, pt_emb)  # cosine similarity (vectors already normalized)
    ranked_idx = np.argsort(-sims)
    true_rank = np.where(ranked_idx == i)[0][0] + 1
    mrr_sum += 1.0 / true_rank
    if true_rank == 1:
        precision_1 += 1
    if true_rank <= 5:
        precision_5 += 1
    if true_rank <= 10:
        precision_10 += 1

print(f"\n=== XLM-R Retrieval Results (Test set of {n_queries} pairs) ===")
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