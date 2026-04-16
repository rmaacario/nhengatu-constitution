#!/usr/bin/env python3
"""Quick test to verify the pipeline works with sample data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corpus_pipeline.sentence_split import SentenceSplitter, SentenceAligner


def main():
    print("Testing sentence splitter...")
    
    splitter = SentenceSplitter()
    aligner = SentenceAligner()
    
    # Test Portuguese
    pt_text = "I – soberania; II – cidadania; III – dignidade da pessoa humana."
    pt_sents = splitter.split_portuguese(pt_text)
    print(f"PT split: {len(pt_sents)} segments")
    for i, s in enumerate(pt_sents[:3]):
        print(f"  {i+1}: {s[:50]}...")
    
    # Test Nheengatu
    nhe_text = "I- Suberania II- Mírasá-itá III- Míra sikuesá"
    nhe_sents = splitter.split_nheengatu(nhe_text)
    print(f"\nNHE split: {len(nhe_sents)} segments")
    for i, s in enumerate(nhe_sents[:3]):
        print(f"  {i+1}: {s[:50]}...")
    
    # Test alignment
    aligned = aligner.align(pt_sents, nhe_sents, 1, "caput")
    print(f"\nAligned pairs: {len(aligned)}")
    for pair in aligned[:3]:
        print(f"  [{pair.confidence}] PT: {pair.pt[:40]}... → NHE: {pair.nhe[:40]}...")
    
    print("\n✓ Quick test passed!")


if __name__ == "__main__":
    main()