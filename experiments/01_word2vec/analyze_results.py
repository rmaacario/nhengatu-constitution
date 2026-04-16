# experiments/01_word2vec/analyze_results.py
import json
import numpy as np
from gensim.models import Word2Vec
from pathlib import Path
from collections import Counter

print("="*70)
print("WORD2VEC LINGUISTIC ANALYSIS")
print("Portuguese vs Nheengatu Semantic Spaces")
print("="*70)

# Load models
pt_model = Word2Vec.load("experiments/01_word2vec/results/pt_w2v_large.model")
nhe_model = Word2Vec.load("experiments/01_word2vec/results/nhe_w2v_large.model")

print(f"\n📊 Model info:")
print(f"   PT vocabulary: {len(pt_model.wv)} words")
print(f"   NHE vocabulary: {len(nhe_model.wv)} words")

# 1. Find the most distinctive words (highest frequency in corpus)
print(f"\n🔍 MOST FREQUENT WORDS IN VOCABULARY:")

# Load corpus to get frequencies
with open('data/processed/sentence_pairs.json', 'r') as f:
    data = json.load(f)

pt_freq = Counter()
nhe_freq = Counter()
for item in data:
    pt_freq.update(item['pt'].lower().split())
    nhe_freq.update(item['nhe'].lower().split())

print(f"\n   Portuguese top 10:")
for word, count in pt_freq.most_common(10):
    if word in pt_model.wv:
        print(f"      {word}: {count} occurrences")

print(f"\n   Nheengatu top 10:")
for word, count in nhe_freq.most_common(10):
    if word in nhe_model.wv:
        print(f"      {word}: {count} occurrences")

# 2. Find words that are semantically similar across languages (potential translation pairs)
print(f"\n🔗 POTENTIAL TRANSLATION PAIRS (via word2vec neighbors):")

translation_candidates = {
    'pt_word': ['direito', 'liberdade', 'justiça', 'povo', 'estado', 'lei', 'poder'],
    'nhe_word': ['direitu', 'timaãresésá', 'supῖtu', 'miíra', 'tetãma', 'suãtisá', 'kirῖbawasá']
}

for pt_w, nhe_w in zip(translation_candidates['pt_word'], translation_candidates['nhe_word']):
    if pt_w in pt_model.wv and nhe_w in nhe_model.wv:
        # Get nearest neighbors in Nheengatu space for the Portuguese word's vector?
        # This requires cross-lingual mapping, but for now just show both spaces
        pt_neighbors = pt_model.wv.most_similar(pt_w, topn=3)
        nhe_neighbors = nhe_model.wv.most_similar(nhe_w, topn=3)
        print(f"\n   {pt_w} ↔ {nhe_w}")
        print(f"      PT neighbors: {pt_neighbors}")
        print(f"      NHE neighbors: {nhe_neighbors}")

# 3. Morphological analysis: Find singular-plural pairs in Nheengatu
print(f"\n🔬 MORPHOLOGICAL ANALYSIS: Nheengatu Plural Formation (-ita suffix)")

# Find all words ending with 'ita'
ita_words = [w for w in nhe_model.wv.key_to_index.keys() if w.endswith('ita')]
print(f"   Words with '-ita' suffix: {len(ita_words)}")
print(f"   Examples: {ita_words[:15]}")

# Find potential singular forms (remove 'ita' and check if exists)
singular_plural_pairs = []
for plural in ita_words:
    singular = plural[:-3]  # Remove 'ita'
    if singular in nhe_model.wv:
        similarity = nhe_model.wv.similarity(singular, plural)
        singular_plural_pairs.append((singular, plural, similarity))

print(f"\n   Singular-plural pairs found: {len(singular_plural_pairs)}")
if singular_plural_pairs:
    print(f"   Top 10 most similar singular-plural pairs:")
    for s, p, sim in sorted(singular_plural_pairs, key=lambda x: -x[2])[:10]:
        print(f"      {s} → {p} (cosine: {sim:.3f})")

# 4. Find the most distinctive Nheengatu words (high frequency, specific meaning)
print(f"\n🌟 DISTINCTIVE NHEENGATU WORDS (high frequency, unique to Nheengatu):")

# Words that appear in Nheengatu but not in Portuguese vocabulary
nhe_specific = [w for w in nhe_model.wv.key_to_index.keys() 
                if w not in pt_model.wv.key_to_index.keys() 
                and nhe_freq[w] > 10]
print(f"   Words not in PT vocab (freq >10): {len(nhe_specific)}")
print(f"   Examples: {nhe_specific[:20]}")

# 5. Cross-lingual analogy test
print(f"\n🧠 ANALOGY TESTS:")

# Portuguese analogies
if all(w in pt_model.wv for w in ['brasil', 'brasileiro', 'estado', 'estadual']):
    try:
        result = pt_model.wv.most_similar(positive=['brasileiro', 'estado'], negative=['brasil'], topn=3)
        print(f"\n   PT: brasil : brasileiro = estado : {result}")
    except:
        print("   PT analogy failed")

# Nheengatu analogies (if we have enough data)
if all(w in nhe_model.wv for w in ['braziu', 'braziupúra', 'tetãma', 'istadu']):
    try:
        result = nhe_model.wv.most_similar(positive=['braziupúra', 'tetãma'], negative=['braziu'], topn=3)
        print(f"\n   NHE: braziu : braziupúra = tetãma : {result}")
    except:
        print("   NHE analogy failed")

print("\n✅ Analysis complete!")
