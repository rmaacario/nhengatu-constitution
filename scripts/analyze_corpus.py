import json
from collections import Counter

print("="*60)
print("CORPUS ANALYSIS")
print("="*60)

# Load your corpus
with open('sentence_output_cli/sentence_pairs.json', 'r') as f:
    data = json.load(f)

print(f"\n✅ Total sentence pairs: {len(data)}")

# Vocabulary stats
pt_vocab = set()
nhe_vocab = set()

for item in data:
    pt_vocab.update(item['pt'].lower().split())
    nhe_vocab.update(item['nhe'].lower().split())

print(f"\n📚 Vocabulary size:")
print(f"   Portuguese: {len(pt_vocab)} unique words")
print(f"   Nheengatu:  {len(nhe_vocab)} unique words")

# Word frequencies
pt_freq = Counter()
nhe_freq = Counter()

for item in data:
    for word in item['pt'].lower().split():
        pt_freq[word] += 1
    for word in item['nhe'].lower().split():
        nhe_freq[word] += 1

print(f"\n📊 Most frequent words - Portuguese:")
for word, count in pt_freq.most_common(10):
    print(f"   {word}: {count}")

print(f"\n📊 Most frequent words - Nheengatu:")
for word, count in nhe_freq.most_common(10):
    print(f"   {word}: {count}")

# Look for Nheengatu agglutination patterns
print(f"\n🔬 Agglutination patterns (Nheengatu suffixes):")
suffixes = ['-ita', '-sá', '-wa', '-tá', '-rã', '-pé', '-kití']
pattern_words = {}
for word in nhe_vocab:
    for suffix in suffixes:
        if word.endswith(suffix):
            root = word[:-len(suffix)]
            if root and len(root) > 1:
                pattern_words.setdefault(suffix, []).append(word)
                break

for suffix, words in pattern_words.items():
    print(f"   Suffix '{suffix}': {len(words)} words")
    if len(words) > 3:
        print(f"      Examples: {words[:5]}")

# Length statistics
pt_lengths = [len(item['pt'].split()) for item in data]
nhe_lengths = [len(item['nhe'].split()) for item in data]

print(f"\n📏 Sentence length (words):")
print(f"   PT - min: {min(pt_lengths)}, max: {max(pt_lengths)}, avg: {sum(pt_lengths)/len(pt_lengths):.1f}")
print(f"   NHE - min: {min(nhe_lengths)}, max: {max(nhe_lengths)}, avg: {sum(nhe_lengths)/len(nhe_lengths):.1f}")

# Find interesting cases
print(f"\n🔍 Interesting finds:")
# Shortest sentences
shortest = min(data, key=lambda x: len(x['pt'].split()))
print(f"   Shortest PT: '{shortest['pt']}'")
print(f"   Corresponding NHE: '{shortest['nhe']}'")

# Longest sentences
longest = max(data, key=lambda x: len(x['pt'].split()))
print(f"\n   Longest PT: {len(longest['pt'].split())} words")
print(f"   Preview: {longest['pt'][:150]}...")

# Article distribution
articles = Counter(item['article'] for item in data)
print(f"\n📑 Articles with most sentence pairs:")
for art, count in articles.most_common(10):
    print(f"   Article {art}: {count} pairs")

print("\n✅ Analysis complete!")
