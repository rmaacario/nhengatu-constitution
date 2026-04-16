

# experiments/02_fasttext/merge_corpora.py
import json, re
from pathlib import Path
from numbers_parser import Document

# --- Load constitution corpus ---
constitution = json.load(open('../../sentence_output_cli/sentence_pairs.json', encoding='utf-8'))

print(f'Constitution corpus: {len(constitution)} pairs')

# --- Load textbook corpus from .numbers file ---
# Try both possible filenames
numbers_candidates = [
    'nheengatu_merged.numbers',
    '../../data/nheengatu_merged.numbers',
    'data/nheengatu_merged.numbers',
]
numbers_path = None
for c in numbers_candidates:
    if Path(c).exists():
        numbers_path = c
        break

if not numbers_path:
    print("ERROR: Could not find .numbers file. Put it in the project root.")
    exit(1)

doc   = Document(numbers_path)
table = doc.sheets[0].tables[0]
textbook = []
for r in range(1, table.num_rows):
    nhe = str(table.cell(r, 2).value).strip()
    pt  = str(table.cell(r, 3).value).strip()
    src = str(table.cell(r, 0).value).strip()
    if nhe and pt and nhe != 'None' and pt != 'None':
        textbook.append({'pt': pt, 'nhe': nhe, 'source': src[:60]})

print(f'Textbook corpus:     {len(textbook)} pairs')

# --- Clean and merge ---
def clean(text):
    text = text.strip().replace('\n', ' ')
    text = re.sub(r'([.,;:!?()\[\]])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

merged = []
seen   = set()

for pair in constitution + textbook:
    pt  = clean(pair['pt'])
    nhe = clean(pair['nhe'])
    key = (pt.lower(), nhe.lower())
    if key in seen or len(pt.split()) < 2 or len(nhe.split()) < 2:
        continue
    seen.add(key)
    merged.append({
        'pt':     pt,
        'nhe':    nhe,
        'source': pair.get('source', 'constitution')
    })

print(f'Merged (deduplicated): {len(merged)} pairs')

# --- Save ---
out = Path('.')
with open(out / 'corpus_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

Path(out / 'corpus_merged.pt').write_text(
    '\n'.join(p['pt']  for p in merged), encoding='utf-8')
Path(out / 'corpus_merged.nhe').write_text(
    '\n'.join(p['nhe'] for p in merged), encoding='utf-8')

print()
print('Files written:')
print('  experiments/02_fasttext/corpus_merged.json')
print('  experiments/02_fasttext/corpus_merged.pt')
print('  experiments/02_fasttext/corpus_merged.nhe')
