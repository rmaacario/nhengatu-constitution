#!/usr/bin/env python3
"""
Align Portuguese and Nheengatu fastText embeddings using orthogonal Procrustes.
Seed dictionary built from linguistically-motivated loanword adaptation.
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent  # experiments/
FASTTEXT_DIR = BASE_DIR / "02_fasttext"
RESULTS_DIR = FASTTEXT_DIR / "results"
RESULTS_DIR = FASTTEXT_DIR / "results"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# Load fastText models
import fasttext
PRETRAINED_PT = Path(__file__).parent / "cc.pt.300.bin"
if PRETRAINED_PT.exists():
    pt_model = fasttext.load_model(str(PRETRAINED_PT))
    print("Loaded pretrained Portuguese fastText (Common Crawl, 300d)")
else:
    print("Pretrained PT model not found. Run: curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz && gunzip cc.pt.300.bin.gz")
    exit(1)
nhe_model = fasttext.load_model(str(RESULTS_DIR / "model_clean.nhe.bin"))

pt_words = pt_model.words
nhe_words = nhe_model.words
print(f"PT vocab: {len(pt_words)}  NHE vocab: {len(nhe_words)}")

# ------------------------------------------------------------
# MONOLINGUAL SANITY CHECK
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("MONOLINGUAL EMBEDDING SANITY CHECK")
print("=" * 60)

pt_test_words = {
    'direito': ['lei', 'justiça', 'dever', 'obrigação'],
    'municipal': ['federal', 'estadual', 'local', 'nacional'],
    'trabalho': ['emprego', 'serviço', 'função', 'atividade'],
}

for word, related in pt_test_words.items():
    if word in pt_words:
        neighbors = pt_model.get_nearest_neighbors(word, k=10)
        top_words = [w for _, w in neighbors]
        found = [w for w in top_words if w in related]
        print(f"\n{word}:")
        print(f"  top 5 neighbors: {top_words[:5]}")
        print(f"  expected related found: {found}")
    else:
        print(f"\n{word}: not in vocabulary")

# ------------------------------------------------------------
# BUILD CLEAN SEED DICTIONARY (Loanword Adaptation)
# ------------------------------------------------------------
def adapt_pt_to_nhe(word: str) -> str:
    """Apply common Nheengatu phonological adaptations to a Portuguese word."""
    w = word.lower()
    w = re.sub(r'ção$', 'sãu', w)
    w = re.sub(r'ções$', 'sũes', w)
    w = re.sub(r'ç', 's', w)                # 'ç' often becomes 's'
    w = re.sub(r'o$', 'u', w)
    w = re.sub(r'e$', 'i', w)
    w = re.sub(r'^es', 'is', w)
    w = re.sub(r'lh', 'y', w)
    w = re.sub(r'nh', 'n', w)
    return w

print("\n" + "=" * 60)
print("BUILDING SEED DICTIONARY (Loanword Adaptation)")
print("=" * 60)

cognate_pairs = []
for pt_w in pt_words:
    if len(pt_w) < 4:
        continue
    adapted = adapt_pt_to_nhe(pt_w)
    if adapted in nhe_words and adapted != pt_w:
        cognate_pairs.append((pt_w, adapted))

print(f"Cognate/loanword pairs found: {len(cognate_pairs)}")

# Fallback: use clean identical overlap if not enough loanword pairs
if len(cognate_pairs) < 30:
    print("Few loanword pairs; supplementing with clean identical overlap.")
    overlap = set(pt_words) & set(nhe_words)
    clean_overlap = [w for w in overlap
                     if len(w) >= 4 and w.isalpha() and not w.isdigit()]
    for w in clean_overlap:
        if w not in dict(cognate_pairs):
            cognate_pairs.append((w, w))
    print(f"Total seed pairs after adding overlap: {len(cognate_pairs)}")

# Build seed dictionary (keep first occurrence in case of duplicates)
seed_dict = {}
for pt, nhe in cognate_pairs:
    if pt not in seed_dict:
        seed_dict[pt] = nhe

print(f"Final seed dictionary size: {len(seed_dict)}")
if len(seed_dict) < 10:
    print("ERROR: Too few seed pairs. Consider manual curation.")
    exit(1)

# Save seed dict for inspection
with open(OUT_DIR / "seed_dict_clean.json", "w", encoding="utf-8") as f:
    json.dump(seed_dict, f, ensure_ascii=False, indent=2)
print(f"Seed dictionary saved to {OUT_DIR / 'seed_dict_clean.json'}")

# Show some examples
print("\nSample seed pairs:")
for i, (pt, nhe) in enumerate(seed_dict.items()):
    if i >= 20:
        break
    print(f"  {pt:20s} → {nhe}")

# ------------------------------------------------------------
# ALIGNMENT (Procrustes)
# ------------------------------------------------------------
src_words = list(seed_dict.keys())
trg_words = [seed_dict[pt] for pt in src_words]

src_vecs = np.array([pt_model.get_word_vector(w) for w in src_words])
trg_vecs = np.array([nhe_model.get_word_vector(w) for w in trg_words])

M = src_vecs.T @ trg_vecs
U, _, Vt = np.linalg.svd(M)
W = U @ Vt

np.save(OUT_DIR / "procrustes_W.npy", W)
print(f"\nMapping matrix saved to {OUT_DIR / 'procrustes_W.npy'}")

# ------------------------------------------------------------
# EVALUATION (on training pairs)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("EVALUATION (on seed dictionary pairs)")
print("=" * 60)

correct_1 = 0
correct_5 = 0
total = len(src_words)

nhe_vecs_all = np.array([nhe_model.get_word_vector(w) for w in nhe_words])

for pt, true_nhe in zip(src_words, trg_words):
    pt_vec = pt_model.get_word_vector(pt)
    mapped_vec = W @ pt_vec
    sims = np.dot(nhe_vecs_all, mapped_vec) / (
        np.linalg.norm(nhe_vecs_all, axis=1) * np.linalg.norm(mapped_vec)
    )
    nearest_idx = np.argsort(-sims)[:5]
    pred_nhe = [nhe_words[idx] for idx in nearest_idx]
    if true_nhe == pred_nhe[0]:
        correct_1 += 1
    if true_nhe in pred_nhe:
        correct_5 += 1

print(f"Precision@1: {correct_1/total:.4f}")
print(f"Precision@5: {correct_5/total:.4f}")

# ------------------------------------------------------------
# EXAMPLE TRANSLATIONS
# ------------------------------------------------------------
test_words = ['direito', 'lei', 'povo', 'língua', 'brasil', 'constituição', 'direitos']
print("\nExample translations (mapped PT → nearest NHE):")
for w in test_words:
    if w in pt_words:
        pt_vec = pt_model.get_word_vector(w)
        mapped = W @ pt_vec
        sims = np.dot(nhe_vecs_all, mapped) / (
            np.linalg.norm(nhe_vecs_all, axis=1) * np.linalg.norm(mapped)
        )
        top_idx = np.argmax(sims)
        print(f"  {w:15s} → {nhe_words[top_idx]}")
    else:
        print(f"  {w:15s} not in PT vocab")