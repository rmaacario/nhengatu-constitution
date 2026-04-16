# experiments/02_fasttext/explore.py
import fasttext
import numpy as np
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent
data_dir = BASE_DIR
out_dir  = BASE_DIR / "results"

pt  = fasttext.load_model(str(out_dir / 'model_merged.pt.bin'))
nhe = fasttext.load_model(str(out_dir / 'model_merged.nhe.bin'))

print('=' * 60)
print('PORTUGUESE NEIGHBORS (merged corpus)')
print('=' * 60)
for word in ['direito', 'lei', 'liberdade', 'povo', 'estado', 'poder', 'cidadão', 'língua']:
    neighbors = pt.get_nearest_neighbors(word, k=5)
    print(f'\n{word}:')
    for score, n in neighbors:
        print(f'  {n} ({score:.3f})')

print()
print('=' * 60)
print('NHEENGATU NEIGHBORS (merged corpus)')
print('=' * 60)

# Load corpus to get frequent words
with open(data_dir / 'corpus_merged.json', encoding='utf-8') as f:
    data = json.load(f)

nhe_words = []
for pair in data:
    nhe_words.extend(pair['nhe'].lower().split())

skip = {'ta','kuá','u','a','i','e','o','de','da','do','se','na','no','asui','upé','waá','ko','pe','ne'}
top = [w for w, c in Counter(nhe_words).most_common(300)
       if len(w) > 3 and w not in skip][:15]
for word in top:
    neighbors = nhe.get_nearest_neighbors(word, k=5)
    print(f'\n{word}:')
    for score, n in neighbors:
        print(f'  {n} ({score:.3f})')

print()
print('=' * 60)
print('MORPHOLOGY: -ita plural suffix (merged)')
print('=' * 60)
ita = [w for w in nhe.words if w.endswith('ita') and len(w) > 4]
print(f'Words ending in -ita: {len(ita)}')

pairs = []
for w in ita:
    root = w[:-3]
    if len(root) > 2 and root in nhe.words:
        v1 = nhe.get_word_vector(w)
        v2 = nhe.get_word_vector(root)
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        pairs.append((root, w, sim))
pairs.sort(key=lambda x: -x[2])

print(f'Root-plural pairs: {len(pairs)}')
print('\nTop pairs:')
for root, plural, sim in pairs[:15]:
    print(f'  {root:20} -> {plural:24} ({sim:.3f})')

pt_roots  = set(pt.words)
loanwords = [(r, p, s) for r, p, s in pairs if r.lower() in pt_roots]
native    = [(r, p, s) for r, p, s in pairs if r.lower() not in pt_roots]
print(f'\nLoanword plurals: {len(loanwords)}')
print(f'Native NHE plurals: {len(native)}')

print()
print('=' * 60)
print('CROSS-DOMAIN CHECK')
print('=' * 60)
textbook_words = ['uikú', 'usú', 'upitá', 'unheẽ', 'aá']
for word in textbook_words:
    if word in nhe.words:
        neighbors = nhe.get_nearest_neighbors(word, k=5)
        print(f'\n{word} (everyday verb):')
        for score, n in neighbors:
            print(f'  {n} ({score:.3f})')