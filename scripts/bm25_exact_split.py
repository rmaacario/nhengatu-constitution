import json
import random
import numpy as np
from rank_bm25 import BM25Okapi

def char_ngram_tokenizer(text, n=3):
    text = text.lower()
    return [text[i:i+n] for i in range(len(text)-n+1)]

# Carregar o mesmo corpus
with open('experiments/02_fasttext/corpus_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Reproduzir exatamente a mesma divisão (seed 42, 80/20, test = último 20%)
random.seed(42)
random.shuffle(data)
split_idx = int(0.8 * len(data))
test_data = data[split_idx:]   # 1000 pares

print(f"Test set size: {len(test_data)}")
print("Primeira frase do teste (confira se é a mesma da avaliação XLM-R):")
print(test_data[0]['pt'][:100])

queries = [p['pt'] for p in test_data]
candidates = [p['nhe'] for p in test_data]

candidate_tokens = [char_ngram_tokenizer(sent) for sent in candidates]
bm25 = BM25Okapi(candidate_tokens)

p1 = p5 = p10 = 0
mrr = 0.0
n = len(queries)

for i, query in enumerate(queries):
    query_tokens = char_ngram_tokenizer(query)
    scores = bm25.get_scores(query_tokens)
    ranked = np.argsort(scores)[::-1]
    rank = np.where(ranked == i)[0][0] + 1
    mrr += 1.0 / rank
    if rank == 1:
        p1 += 1
    if rank <= 5:
        p5 += 1
    if rank <= 10:
        p10 += 1

print(f"\nBM25 (char 3-gram) no mesmo test set (n={n})")
print(f"P@1: {p1/n:.4f}  ({p1}/{n})")
print(f"P@5: {p5/n:.4f}  ({p5}/{n})")
print(f"P@10: {p10/n:.4f} ({p10}/{n})")
print(f"MRR: {mrr/n:.4f}")