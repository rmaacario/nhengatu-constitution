# experiments/02_fasttext/train.py
import fasttext
from pathlib import Path

BASE_DIR = Path(__file__).parent
data_dir = BASE_DIR    
out_dir  = BASE_DIR / "results"
out_dir.mkdir(exist_ok=True)

# Optional: retrain Portuguese at 300d if you want a matching from-scratch baseline
# but we won't use it for the alignment with pretrained PT.
print('Training Portuguese model on merged corpus (300d, optional)...')
pt = fasttext.train_unsupervised(
    str(data_dir / 'corpus_merged.pt'),
    model='skipgram', dim=300, epoch=20,
    minCount=1, minn=3, maxn=6, thread=4
)
pt.save_model(str(out_dir / 'model_merged_300d.pt.bin'))
print(f'  Vocabulary: {len(pt.words)} words')

print('Training Nheengatu model on merged corpus (300d)...')
nhe = fasttext.train_unsupervised(
    str(data_dir / 'corpus_merged.nhe'),
    model='skipgram', dim=300, epoch=20,
    minCount=1, minn=3, maxn=6, thread=4
)
nhe.save_model(str(out_dir / 'model_merged.nhe.bin'))   # overwrites the 100d version
print(f'  Vocabulary: {len(nhe.words)} words')
print('Done.')