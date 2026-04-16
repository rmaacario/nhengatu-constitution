#!/usr/bin/env python3
"""Quality filter for sentence pairs before XLM-R training."""

import json
import re
import argparse
from pathlib import Path

def filter_pairs(data, max_words=100, max_ratio=4.0):
    """Apply quality filters to sentence pairs."""
    clean = []
    stats = {'total': len(data), 'ratio': 0, 'long': 0, 'revoked': 0}
    
    for p in data:
        pt, nhe = p['pt'], p['nhe']
        pt_words = len(pt.split())
        nhe_words = len(nhe.split())
        
        # Length ratio check
        ratio = len(pt) / max(len(nhe), 1)
        if ratio > max_ratio or ratio < 1/max_ratio:
            stats['ratio'] += 1
            continue
            
        # Length check
        if pt_words > max_words or nhe_words > max_words:
            stats['long'] += 1
            continue
            
        # Revoked articles check
        if re.search(r'\(?\s*[Rr]evogad', pt):
            stats['revoked'] += 1
            continue
            
        clean.append(p)
    
    print(f"Original: {stats['total']} | Clean: {len(clean)} | Removed: {stats['total']-len(clean)}")
    print(f"  - Ratio outliers: {stats['ratio']}")
    print(f"  - Too long: {stats['long']}")
    print(f"  - Revoked: {stats['revoked']}")
    
    return clean

def main():
    parser = argparse.ArgumentParser(description='Filter parallel corpus for XLM-R training')
    parser.add_argument('input', help='Input JSON file with sentence pairs')
    parser.add_argument('-o', '--output', help='Output JSON file (default: input_clean.json)')
    parser.add_argument('--max-words', type=int, default=100, help='Maximum words per sentence')
    parser.add_argument('--max-ratio', type=float, default=4.0, help='Maximum length ratio')
    
    args = parser.parse_args()
    
    # Load data
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter
    clean = filter_pairs(data, args.max_words, args.max_ratio)
    
    # Save output
    output_path = args.output or args.input.replace('.json', '_clean.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(clean)} pairs to {output_path}")

if __name__ == '__main__':
    main()