#!/usr/bin/env python3
"""
Fine-tune XLM-RoBERTa on parallel Portuguese–Nheengatu sentences
using MultipleNegativesRankingLoss (contrastive learning).
"""

import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
FASTTEXT_DIR = BASE_DIR / "02_fasttext"
CORPUS_PATH = FASTTEXT_DIR / "corpus_merged.json"
OUTPUT_DIR = Path(__file__).parent / "results" / "xlmr_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16          # Reduce if you run out of memory
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = random.Random(42)
np.random.seed(42)
torch.manual_seed(42)

print(f"Using device: {DEVICE}")

# ------------------------------------------------------------
# Load and split data
# ------------------------------------------------------------
print("Loading merged corpus...")
with open(CORPUS_PATH, encoding="utf-8") as f:
    data = json.load(f)

rng.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Train pairs: {len(train_data)}")
print(f"Test pairs:  {len(test_data)}")

# ------------------------------------------------------------
# Dataset and DataLoader
# ------------------------------------------------------------
class ParallelDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]['pt'], self.pairs[idx]['nhe']

def collate_fn(batch, tokenizer):
    pt_texts = [item[0] for item in batch]
    nhe_texts = [item[1] for item in batch]

    pt_enc = tokenizer(pt_texts, padding=True, truncation=True,
                       max_length=MAX_LENGTH, return_tensors='pt')
    nhe_enc = tokenizer(nhe_texts, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors='pt')
    return pt_enc, nhe_enc

# ------------------------------------------------------------
# Model and Tokenizer
# ------------------------------------------------------------
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE)

dataset = ParallelDataset(train_data)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, tokenizer))

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# ------------------------------------------------------------
# Mean Pooling Helper
# ------------------------------------------------------------
def mean_pool(model_output, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (model_output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
print("\nStarting fine-tuning...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for step, (pt_enc, nhe_enc) in enumerate(loader):
        # Move to device
        pt_enc = {k: v.to(DEVICE) for k, v in pt_enc.items()}
        nhe_enc = {k: v.to(DEVICE) for k, v in nhe_enc.items()}

        # Forward pass
        pt_out = model(**pt_enc)
        nhe_out = model(**nhe_enc)

        pt_embs = mean_pool(pt_out, pt_enc['attention_mask'])
        nhe_embs = mean_pool(nhe_out, nhe_enc['attention_mask'])

        # Contrastive loss (MultipleNegativesRankingLoss)
        pt_embs = F.normalize(pt_embs, p=2, dim=1)
        nhe_embs = F.normalize(nhe_embs, p=2, dim=1)
        scores = torch.matmul(pt_embs, nhe_embs.T) * 20  # temperature scaling
        labels = torch.arange(len(pt_embs), device=DEVICE)
        loss = F.cross_entropy(scores, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"  Epoch {epoch+1}, step {step}: loss = {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

# ------------------------------------------------------------
# Save Fine‑Tuned Model
# ------------------------------------------------------------
print(f"\nSaving fine‑tuned model to {OUTPUT_DIR}")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("Done.")