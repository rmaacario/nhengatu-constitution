#!/usr/bin/env python3
"""Train FastText models on Nheengatu and Portuguese corpora"""

import fasttext
from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CORPUS_PT = PROJECT_ROOT / "data/processed/corpus_clean.pt"
CORPUS_NHE = PROJECT_ROOT / "data/processed/corpus_clean.nhe"
OUTPUT_DIR = PROJECT_ROOT / "experiments/02_fasttext/results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("FASTTEXT TRAINING")
print("="*60)

# Train Portuguese model
print(f"\n📚 Training Portuguese model from {CORPUS_PT}")
if CORPUS_PT.exists():
    model_pt = fasttext.train_unsupervised(
        str(CORPUS_PT),
        model='skipgram',
        dim=300,
        epoch=10,
        minCount=3
    )
    model_pt.save_model(str(OUTPUT_DIR / "model_clean.pt.bin"))
    print(f"✅ Portuguese model saved ({OUTPUT_DIR / 'model_clean.pt.bin'})")
else:
    print(f"❌ Corpus not found: {CORPUS_PT}")

# Train Nheengatu model
print(f"\n📚 Training Nheengatu model from {CORPUS_NHE}")
if CORPUS_NHE.exists():
    model_nhe = fasttext.train_unsupervised(
        str(CORPUS_NHE),
        model='skipgram',
        dim=300,
        epoch=10,
        minCount=3
    )
    model_nhe.save_model(str(OUTPUT_DIR / "model_clean.nhe.bin"))
    print(f"✅ Nheengatu model saved ({OUTPUT_DIR / 'model_clean.nhe.bin'})")
else:
    print(f"❌ Corpus not found: {CORPUS_NHE}")

print("\n✅ FastText training complete!")
