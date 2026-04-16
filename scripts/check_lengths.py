import json

with open('experiments/02_fasttext/corpus_clean.json', 'r') as f:
    data = json.load(f)

# First, print the keys of the first item to see structure
print("Keys in first item:", data[0].keys())
print("First item sample:", data[0])
print("-" * 50)

tycho_lengths = []
casasnovas_lengths = []

for item in data:
    # Try different possible key names
    source = item.get('source', '') or item.get('corpus', '') or item.get('src', '')
    nhe_text = item.get('nhe', '') or item.get('target', '') or item.get('nheengatu', '')
    if not nhe_text:
        continue
    word_count = len(nhe_text.split())
    if 'tycho' in source.lower():
        tycho_lengths.append(word_count)
    elif 'casasnovas' in source.lower():
        casasnovas_lengths.append(word_count)

print(f"Found {len(tycho_lengths)} Tycho sentences, {len(casasnovas_lengths)} Casasnovas sentences")
if tycho_lengths:
    print(f"Tycho mean length: {sum(tycho_lengths)/len(tycho_lengths):.1f} words")
if casasnovas_lengths:
    print(f"Casasnovas mean length: {sum(casasnovas_lengths)/len(casasnovas_lengths):.1f} words")