"""
experiments/01_word2vec/train_word2vec.py
Word2Vec baseline for Portuguese-Nheengatu parallel corpus.
"""

import json
import sys
from pathlib import Path
from gensim.models import Word2Vec
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load corpus
corpus_path = Path("data/processed/sentence_pairs.json")
with open(corpus_path, 'r') as f:
    data = json.load(f)

print("="*60)
print("WORD2VEC BASELINE - Portuguese/Nheengatu")
print("="*60)
print(f"Corpus: {len(data)} sentence pairs")
print(f"Source: {corpus_path}")

# Prepare sentences
pt_sentences = [item['pt'].lower().split() for item in data]
nhe_sentences = [item['nhe'].lower().split() for item in data]

# Train models
models = {}

for lang, sents in [('pt', pt_sentences), ('nhe', nhe_sentences)]:
    print(f"\n🔧 Training {lang} model...")
    
    # Small window (syntactic)
    m_small = Word2Vec(sents, vector_size=100, window=3, min_count=3, epochs=10)
    m_small.save(f"experiments/01_word2vec/results/{lang}_w2v_small.model")
    
    # Large window (semantic)
    m_large = Word2Vec(sents, vector_size=100, window=10, min_count=3, epochs=10)
    m_large.save(f"experiments/01_word2vec/results/{lang}_w2v_large.model")
    
    models[lang] = {'small': m_small, 'large': m_large}
    
    print(f"   Vocabulary: {len(m_small.wv)} / {len(m_large.wv)}")

# Test similarities
print("\n🔍 TESTING WORD SIMILARITIES")
print("-"*40)

# Portuguese test
print("\nPortuguese (window=10):")
for word in ['direito', 'liberdade', 'justiça', 'povo']:
    if word in models['pt']['large'].wv:
        sim = models['pt']['large'].wv.most_similar(word, topn=3)
        print(f"  {word}: {sim}")

# Nheengatu test
print("\nNheengatu (window=10):")
for word in ['direitu', 'timaãresésá', 'miíra', 'tetãma']:
    if word in models['nhe']['large'].wv:
        sim = models['nhe']['large'].wv.most_similar(word, topn=3)
        print(f"  {word}: {sim}")

print("\n✅ Models saved to experiments/01_word2vec/results/")
