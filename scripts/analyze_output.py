import json
import re
from collections import Counter, defaultdict

print("="*70)
print("CORPUS ANALYSIS - 1,301 SENTENCE PAIRS")
print("="*70)

with open('sentence_output_hybrid/sentence_pairs.json', 'r') as f:
    data = json.load(f)

# 1. Basic stats
print("\n📊 BASIC STATISTICS:")
print(f"   Total sentence pairs: {len(data)}")
print(f"   Unique articles: {len(set(p['article'] for p in data))}")
print(f"   Average PT length: {sum(len(p['pt']) for p in data)/len(data):.0f} chars")
print(f"   Average NHE length: {sum(len(p['nhe']) for p in data)/len(data):.0f} chars")

# 2. Article distribution
print("\n📊 TOP ARTICLES BY SENTENCE COUNT:")
art_counts = Counter(p['article'] for p in data)
for art, count in art_counts.most_common(10):
    print(f"   Article {art}: {count} sentences")

# 3. Inciso distribution
print("\n📊 INCISO TYPES:")
inciso_counts = Counter(p['unit_type'] for p in data if p['unit_type'].startswith('inciso'))
print(f"   Total inciso types: {len(inciso_counts)}")
print(f"   Max inciso number: {max(int(re.search(r'inciso_(\d+)', t).group(1)) for t in inciso_counts if re.search(r'inciso_(\d+)', t))}")
print(f"   Most common incisos: {inciso_counts.most_common(5)}")

# 4. Length distribution
print("\n📊 SENTENCE LENGTH DISTRIBUTION (PT words):")
lengths = [len(p['pt'].split()) for p in data]
print(f"   Min: {min(lengths)} words")
print(f"   Max: {max(lengths)} words")
print(f"   Avg: {sum(lengths)/len(lengths):.1f} words")
print(f"   Median: {sorted(lengths)[len(lengths)//2]} words")

# 5. Check for issues
print("\n⚠️  QUALITY CHECKS:")
# Check for empty NHE
empty_nhe = [p for p in data if not p['nhe'].strip()]
print(f"   Empty NHE sentences: {len(empty_nhe)}")
# Check for very short
short = [p for p in data if len(p['pt'].split()) < 3]
print(f"   Very short PT (<3 words): {len(short)}")
# Check for NHE characters in PT
nhe_chars = set('ŨẼÃÕÂÊÎÔÛÁÉÍÓÚ')
nhe_in_pt = [p for p in data if any(c in p['pt'] for c in nhe_chars)]
print(f"   NHE chars in PT (potential error): {len(nhe_in_pt)}")

# 6. Sample quality
print("\n📝 SAMPLE SENTENCES BY TYPE:")
print("\n--- CAPUT example ---")
caput = [p for p in data if p['unit_type'] == 'caput'][:1]
if caput:
    print(f"PT: {caput[0]['pt'][:200]}...")
    print(f"NHE: {caput[0]['nhe'][:200]}...")

print("\n--- INCISO example ---")
inciso = [p for p in data if p['unit_type'].startswith('inciso')][:1]
if inciso:
    print(f"PT: {inciso[0]['pt'][:200]}...")
    print(f"NHE: {inciso[0]['nhe'][:200]}...")

print("\n--- PREAMBLE example ---")
preamble = [p for p in data if p['unit_type'] == 'preamble']
if preamble:
    print(f"PT: {preamble[0]['pt'][:200]}...")
    print(f"NHE: {preamble[0]['nhe'][:200]}...")

# 7. Article 5 deep dive
print("\n📝 ARTICLE 5 DEEP DIVE:")
art5 = [p for p in data if p['article'] == 5]
print(f"   Article 5 has {len(art5)} sentence pairs")
inciso_5 = [p for p in art5 if p['unit_type'] == 'inciso_1']
if inciso_5:
    print(f"   First inciso: {inciso_5[0]['pt'][:150]}...")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE - CORPUS READY FOR MT TRAINING")
print("="*70)
