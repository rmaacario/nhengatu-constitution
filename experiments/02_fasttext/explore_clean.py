# experiments/02_fasttext/explore_clean.py
import fasttext, numpy as np, json
from collections import Counter
from pathlib import Path

pt  = fasttext.load_model('experiments/02_fasttext/results/model_clean.pt.bin')
nhe = fasttext.load_model('experiments/02_fasttext/results/model_clean.nhe.bin')

print('=' * 60)
print('PORTUGUESE NEIGHBORS (clean)')
print('=' * 60)
for word in ['direito', 'lei', 'liberdade', 'povo', 'estado', 'poder', 'cidadão']:
    neighbors = pt.get_nearest_neighbors(word, k=5)
    print(f'\n{word}:')
    for score, n in neighbors:
        print(f'  {n} ({score:.3f})')

print()
print('=' * 60)
print('NHEENGATU NEIGHBORS (clean)')
print('=' * 60)
data = json.load(open('sentence_output_cli/sentence_pairs.json', encoding='utf-8'))
nhe_words = []
for pair in data:
    nhe_words.extend(pair['nhe'].lower().split())
skip = {'ta','kuá','u','a','i','e','o','de','da','do','se','na','no','asui','upé','waá'}
top = [w for w,c in Counter(nhe_words).most_common(200)
       if len(w) > 3 and w not in skip][:12]
for word in top:
    neighbors = nhe.get_nearest_neighbors(word, k=5)
    print(f'\n{word}:')
    for score, n in neighbors:
        print(f'  {n} ({score:.3f})')

print()
print('=' * 60)
print('MORPHOLOGY: -ita plural suffix (clean)')
print('=' * 60)
ita = [w for w in nhe.words if w.endswith('ita') and len(w) > 4]
print(f'Words ending in -ita: {len(ita)}')

pairs = []
for w in ita:
    root = w[:-3]
    if len(root) > 2 and root in nhe.words:
        v1 = nhe.get_word_vector(w)
        v2 = nhe.get_word_vector(root)
        sim = float(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        pairs.append((root, w, sim))
pairs.sort(key=lambda x: -x[2])

print(f'Root-plural pairs found: {len(pairs)}')
print()
print('Top root -> plural pairs:')
for root, plural, sim in pairs[:15]:
    print(f'  {root:18} -> {plural:22} ({sim:.3f})')

# Separate native vs loanword pairs
print()
pt_roots = set(pt.words)
loanwords = [(r,p,s) for r,p,s in pairs if r.lower() in pt_roots]
native    = [(r,p,s) for r,p,s in pairs if r.lower() not in pt_roots]
print(f'Loanword plurals (PT root in NHE): {len(loanwords)}')
print(f'Likely native NHE plurals:         {len(native)}')
print()
print('Sample loanword pairs:')
for root, plural, sim in loanwords[:8]:
    print(f'  {root:18} -> {plural:22} ({sim:.3f})')
print()
print('Sample native pairs:')
for root, plural, sim in native[:8]:
    print(f'  {root:18} -> {plural:22} ({sim:.3f})')
