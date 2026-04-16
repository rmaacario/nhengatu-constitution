#!/usr/bin/env python3
"""Word2Vec baseline for Portuguese-Nheengatu parallel corpus."""

import json
import sys
from pathlib import Path
from gensim.models import Word2Vec
from collections import Counter

# Get project root (4 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
corpus_path = PROJECT_ROOT / "data/processed/merged_5028_pairs.json"

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
for lang, sents in [('pt', pt_sentences), ('nhe', nhe_sentences)]:
    print(f"\n🔧 Training {lang} model...")
    
    # Small window (syntactic)
    m_small = Word2Vec(sents, vector_size=100, window=3, min_count=3, epochs=10)
    m_small.save(str(PROJECT_ROOT / f"experiments/01_word2vec/results/{lang}_w2v_small.model"))
    
    # Large window (semantic)
    m_large = Word2Vec(sents, vector_size=100, window=10, min_count=3, epochs=10)
    m_large.save(str(PROJECT_ROOT / f"experiments/01_word2vec/results/{lang}_w2v_large.model"))
    
    print(f"   Vocabulary: {len(m_small.wv)} / {len(m_large.wv)}")

print("\n✅ Models saved to experiments/01_word2vec/results/")
