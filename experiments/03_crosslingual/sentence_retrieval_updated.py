#!/usr/bin/env python3
"""
Sentence-level cross-lingual retrieval using fastText embeddings on CLEAN corpus.
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
FASTTEXT_DIR = BASE_DIR / "02_fasttext"
RESULTS_DIR = FASTTEXT_DIR / "results"

# Usar modelos TREINADOS NO CORPUS LIMPO
CORPUS_PATH = FASTTEXT_DIR / "corpus_clean.json"
PT_MODEL_PATH = RESULTS_DIR / "model_clean.pt.bin"
NHE_MODEL_PATH = RESULTS_DIR / "model_clean.nhe.bin"

print("="*60)
print("SENTENCE RETRIEVAL - CORPUS LIMPO (4,997 pares)")
print("="*60)

# Verificar se os arquivos existem
if not PT_MODEL_PATH.exists():
    print(f"ERRO: Modelo PT não encontrado: {PT_MODEL_PATH}")
    print("Execute primeiro: cd ../02_fasttext && python3 train.py")
    exit(1)

if not NHE_MODEL_PATH.exists():
    print(f"ERRO: Modelo NHE não encontrado: {NHE_MODEL_PATH}")
    exit(1)

if not CORPUS_PATH.exists():
    print(f"ERRO: Corpus não encontrado: {CORPUS_PATH}")
    exit(1)

# Set seed
random.seed(42)
np.random.seed(42)

# Load models
import fasttext
print("\n1. Carregando modelos...")
pt_model = fasttext.load_model(str(PT_MODEL_PATH))
nhe_model = fasttext.load_model(str(NHE_MODEL_PATH))
print(f"   PT: {len(pt_model.words)} palavras, dim={pt_model.get_dimension()}")
print(f"   NHE: {len(nhe_model.words)} palavras, dim={nhe_model.get_dimension()}")

# Load corpus
print("\n2. Carregando corpus limpo...")
with open(CORPUS_PATH, encoding="utf-8") as f:
    data = json.load(f)
print(f"   Total de pares: {len(data)} (31 pares ruins removidos)")

# Estratificar por fonte para evitar vazamento de dados
print("\n3. Separando por fonte (estratificado)...")
by_source = defaultdict(list)
for item in data:
    source = item.get('source', 'unknown')
    by_source[source].append(item)

train_data = []
test_data = []
for source, items in by_source.items():
    split_idx = int(0.8 * len(items))
    train_data.extend(items[:split_idx])
    test_data.extend(items[split_idx:])
    print(f"   {source}: {len(items)} total, {len(items[:split_idx])} treino, {len(items[split_idx:])} teste")

random.shuffle(test_data)
print(f"\n   Total treino: {len(train_data)}")
print(f"   Total teste: {len(test_data)}")

# Sentence embedding
def sent_embed(text, model):
    words = text.lower().split()
    vectors = [model.get_word_vector(w) for w in words if w in model.words]
    if not vectors:
        return np.zeros(model.get_dimension())
    return np.mean(vectors, axis=0)

print("\n4. Computando embeddings...")
test_pt_embs = np.array([sent_embed(p['pt'], pt_model) for p in test_data])
test_nhe_embs = np.array([sent_embed(p['nhe'], nhe_model) for p in test_data])

# Normalizar
test_pt_embs = test_pt_embs / np.linalg.norm(test_pt_embs, axis=1, keepdims=True)
test_nhe_embs = test_nhe_embs / np.linalg.norm(test_nhe_embs, axis=1, keepdims=True)

# Avaliação
print("\n5. Avaliando retrieval...")
precision_1 = precision_5 = precision_10 = mrr_sum = 0
n_queries = len(test_pt_embs)

for i, pt_emb in enumerate(test_pt_embs):
    sims = np.dot(test_nhe_embs, pt_emb)
    ranked_idx = np.argsort(-sims)
    true_rank = np.where(ranked_idx == i)[0][0] + 1
    mrr_sum += 1.0 / true_rank
    
    if true_rank == 1:
        precision_1 += 1
    if true_rank <= 5:
        precision_5 += 1
    if true_rank <= 10:
        precision_10 += 1

print(f"\n" + "="*60)
print(f"RESULTADOS - CORPUS LIMPO (sem os 31 pares problemáticos)")
print(f"="*60)
print(f"Test set size: {n_queries} pares")
print(f"Precision@1:  {precision_1 / n_queries:.4f} ({precision_1}/{n_queries})")
print(f"Precision@5:  {precision_5 / n_queries:.4f} ({precision_5}/{n_queries})")
print(f"Precision@10: {precision_10 / n_queries:.4f} ({precision_10}/{n_queries})")
print(f"MRR:          {mrr_sum / n_queries:.4f}")

# Exemplos
print("\n" + "="*60)
print("EXEMPLOS DE RETRIEVAL")
print("="*60)
for i in range(min(3, n_queries)):
    pt_emb = test_pt_embs[i]
    sims = np.dot(test_nhe_embs, pt_emb)
    top_idx = np.argmax(sims)
    print(f"\nQuery {i+1}:")
    print(f"  PT:    {test_data[i]['pt'][:80]}...")
    print(f"  True:  {test_data[i]['nhe'][:80]}...")
    print(f"  Pred:  {test_data[top_idx]['nhe'][:80]}...")
    print(f"  Match: {'✅' if top_idx == i else '❌'} (rank={np.where(np.argsort(-sims) == i)[0][0]+1})")

print("\n✅ Avaliação concluída!")
