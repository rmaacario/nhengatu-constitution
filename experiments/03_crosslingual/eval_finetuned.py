#!/usr/bin/env python3
#!/usr/bin/env python3
"""Evaluate fine‑tuned XLM‑R on sentence retrieval (same test split)."""

import json
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).parent.parent
FASTTEXT_DIR = BASE_DIR / "02_fasttext"
CORPUS_PATH = FASTTEXT_DIR / "corpus_merged.json"
MODEL_PATH = Path(__file__).parent / "results" / "xlmr_finetuned"

rng = random.Random(42)
np.random.seed(42)
torch.manual_seed(42)

# Load and split
with open(CORPUS_PATH, encoding="utf-8") as f:
    data = json.load(f)
rng.shuffle(data)
split_idx = int(0.8 * len(data))
test_data = data[split_idx:]
print(f"Test pairs: {len(test_data)}")

# Load fine‑tuned model
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModel.from_pretrained(str(MODEL_PATH))
model.eval()

def embed_texts(texts, batch_size=32):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded)
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)

print("Embedding test sentences...")
pt_embs = embed_texts([p['pt'] for p in test_data])
nhe_embs = embed_texts([p['nhe'] for p in test_data])

# Retrieval
p1 = p5 = p10 = mrr = 0
n = len(pt_embs)
for i, pt in enumerate(pt_embs):
    sims = np.dot(nhe_embs, pt)
    rank = np.argsort(-sims)
    pos = np.where(rank == i)[0][0] + 1
    mrr += 1.0 / pos
    if pos == 1: p1 += 1
    if pos <= 5: p5 += 1
    if pos <= 10: p10 += 1

print(f"\nFine‑tuned XLM‑R Retrieval (n={n})")
print(f"P@1:  {p1/n:.4f}")
print(f"P@5:  {p5/n:.4f}")
print(f"P@10: {p10/n:.4f}")
print(f"MRR:  {mrr/n:.4f}")