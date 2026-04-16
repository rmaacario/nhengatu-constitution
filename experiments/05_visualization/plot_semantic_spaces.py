import json
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("="*70)
print("VISUALIZING SEMANTIC SPACES")
print("Portuguese vs Nheengatu")
print("="*70)

# Load models
pt_model = Word2Vec.load("experiments/01_word2vec/results/pt_w2v_large.model")
nhe_model = Word2Vec.load("experiments/01_word2vec/results/nhe_w2v_large.model")

# 1. Plot: Most frequent words in both languages
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Portuguese words
pt_words = ['de', 'e', 'a', 'o', 'do', 'da', 'os', 'ou', 'dos', 'em']
pt_vectors = np.array([pt_model.wv[w] for w in pt_words if w in pt_model.wv])

pca_pt = PCA(n_components=2)
pt_2d = pca_pt.fit_transform(pt_vectors)

axes[0].scatter(pt_2d[:, 0], pt_2d[:, 1], alpha=0.7, color='blue')
for i, word in enumerate([w for w in pt_words if w in pt_model.wv]):
    axes[0].annotate(word, (pt_2d[i, 0], pt_2d[i, 1]), fontsize=10)
axes[0].set_title('Portuguese: Most Frequent Words')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# Nheengatu words
nhe_words = ['ta', 'kuá', 'asui', 'upé', 'mayê', 'ũbeu', 'u', 'rupí', 'waá', 'aárama']
nhe_vectors = np.array([nhe_model.wv[w] for w in nhe_words if w in nhe_model.wv])

pca_nhe = PCA(n_components=2)
nhe_2d = pca_nhe.fit_transform(nhe_vectors)

axes[1].scatter(nhe_2d[:, 0], nhe_2d[:, 1], alpha=0.7, color='green')
for i, word in enumerate([w for w in nhe_words if w in nhe_model.wv]):
    axes[1].annotate(word, (nhe_2d[i, 0], nhe_2d[i, 1]), fontsize=10)
axes[1].set_title('Nheengatu: Most Frequent Words')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('experiments/05_visualization/plots/freq_words_pca.png', dpi=150)
print("✅ Saved: freq_words_pca.png")

# 2. Plot: Semantic fields comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

semantic_fields = {
    'direito/direitu': (['direito', 'lei', 'justiça', 'poder', 'estado'], 
                        ['direitu', 'suãtisá', 'supῖtu', 'kirῖbawasá', 'tetãma']),
    'povo/miíra': (['povo', 'cidadão', 'nação', 'sociedade', 'comunidade'],
                   ['miíra', 'sidadãu', 'tetãma', 'payẽ', 'yumuatiri']),
}

for idx, (title, (pt_words, nhe_words)) in enumerate(semantic_fields.items()):
    # Portuguese
    pt_vecs = np.array([pt_model.wv[w] for w in pt_words if w in pt_model.wv])
    if len(pt_vecs) > 0:
        pca = PCA(n_components=2)
        pt_2d = pca.fit_transform(pt_vecs)
        axes[idx, 0].scatter(pt_2d[:, 0], pt_2d[:, 1], color='blue', alpha=0.7)
        for i, w in enumerate([w for w in pt_words if w in pt_model.wv]):
            axes[idx, 0].annotate(w, (pt_2d[i, 0], pt_2d[i, 1]), fontsize=9)
        axes[idx, 0].set_title(f'Portuguese: {title}')
    
    # Nheengatu
    nhe_vecs = np.array([nhe_model.wv[w] for w in nhe_words if w in nhe_model.wv])
    if len(nhe_vecs) > 0:
        pca = PCA(n_components=2)
        nhe_2d = pca.fit_transform(nhe_vecs)
        axes[idx, 1].scatter(nhe_2d[:, 0], nhe_2d[:, 1], color='green', alpha=0.7)
        for i, w in enumerate([w for w in nhe_words if w in nhe_model.wv]):
            axes[idx, 1].annotate(w, (nhe_2d[i, 0], nhe_2d[i, 1]), fontsize=9)
        axes[idx, 1].set_title(f'Nheengatu: {title}')
    
    # Combined
    axes[idx, 2].scatter(pt_2d[:, 0], pt_2d[:, 1], color='blue', alpha=0.7, label='PT')
    axes[idx, 2].scatter(nhe_2d[:, 0], nhe_2d[:, 1], color='green', alpha=0.7, label='NHE')
    axes[idx, 2].set_title(f'Combined: {title}')
    axes[idx, 2].legend()

plt.tight_layout()
plt.savefig('experiments/05_visualization/plots/semantic_fields.png', dpi=150)
print("✅ Saved: semantic_fields.png")

# 3. Plot: Singular-plural pairs similarity
singular_plural_pairs = [
    ('aára', 'aáraita', 0.995), ('kutaárawá', 'kutaárawáita', 0.994),
    ('suraára', 'suraáraita', 0.989), ('uka', 'ukaita', 0.988),
    ('amũ', 'amũita', 0.988), ('mã', 'mãita', 0.986),
]

fig, ax = plt.subplots(figsize=(10, 6))
pairs = [f"{s}→{p}" for s, p, _ in singular_plural_pairs]
similarities = [sim for _, _, sim in singular_plural_pairs]

bars = ax.barh(pairs, similarities, color='coral')
ax.set_xlabel('Cosine Similarity')
ax.set_title('Nheengatu Singular-Plural Pairs Similarity')
ax.set_xlim(0.98, 1.0)

for i, (bar, sim) in enumerate(zip(bars, similarities)):
    ax.text(sim + 0.001, bar.get_y() + bar.get_height()/2, f'{sim:.3f}', va='center')

plt.tight_layout()
plt.savefig('experiments/05_visualization/plots/singular_plural_similarity.png', dpi=150)
print("✅ Saved: singular_plural_similarity.png")

print("\n✅ All visualizations saved to experiments/05_visualization/plots/")
