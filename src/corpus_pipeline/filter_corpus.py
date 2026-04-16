import json, re

with open('/mnt/user-data/uploads/merged_5028_pairs.json') as f:
    data = json.load(f)

def is_clean(p):
    pt, nhe = p['pt'], p['nhe']
    pt_words = len(pt.split())
    nhe_words = len(nhe.split())
    # Remove ratio outliers
    ratio = len(pt) / max(len(nhe), 1)
    if ratio > 4 or ratio < 0.25:
        return False
    # Remove very long pairs (XLM-R truncation)
    if pt_words > 100 or nhe_words > 100:
        return False
    # Remove revoked articles
    if re.search(r'\(?\s*[Rr]evogad', pt):
        return False
    return True

clean = [p for p in data if is_clean(p)]
print(f'Original: {len(data)} | Clean: {len(clean)} | Removed: {len(data)-len(clean)}')

with open('corpus_clean_5028.json', 'w', encoding='utf-8') as f:
    json.dump(clean, f, ensure_ascii=False, indent=2)
