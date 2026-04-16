"""
Experiments module for linguistic analysis of the parallel corpus.
Run after corpus generation to produce word2vec models and visualizations.
"""

import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("Warning: gensim not installed. Run: pip install gensim")

try:
    from plotnine import *
    HAS_PLOTNINE = True
except ImportError:
    HAS_PLOTNINE = False
    print("Warning: plotnine not installed. Run: pip install plotnine")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")


def run_word2vec_experiment(corpus_path: Path, output_dir: Path):
    """Run word2vec analysis on the parallel corpus."""
    
    # Convert to string for gensim compatibility
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("WORD2VEC LINGUISTIC ANALYSIS")
    print("Portuguese vs Nheengatu Semantic Spaces")
    print("="*70)
    
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 Corpus: {len(data)} sentence pairs")
    
    # Prepare sentences
    pt_sentences = [item['pt'].lower().split() for item in data]
    nhe_sentences = [item['nhe'].lower().split() for item in data]
    
    # Train models
    print(f"\n🔧 Training word2vec models...")
    
    models = {}
    for lang, sents in [('pt', pt_sentences), ('nhe', nhe_sentences)]:
        # Small window (syntactic)
        m_small = Word2Vec(sents, vector_size=100, window=3, min_count=3, epochs=10)
        m_small.save(str(output_dir / f"{lang}_w2v_small.model"))
        
        # Large window (semantic)
        m_large = Word2Vec(sents, vector_size=100, window=10, min_count=3, epochs=10)
        m_large.save(str(output_dir / f"{lang}_w2v_large.model"))
        
        models[lang] = {'small': m_small, 'large': m_large}
        print(f"   {lang}: {len(m_small.wv)} / {len(m_large.wv)} words")
    
    # Analyze
    analyze_word2vec(models, data, output_dir)
    
    if HAS_PLOTNINE and HAS_SKLEARN:
        generate_visualizations(models, data, output_dir)
    else:
        print("\n⚠️ Skipping visualizations (install plotnine and scikit-learn)")
    
    return models


def analyze_word2vec(models, data, output_dir):
    """Linguistic analysis of word2vec models."""
    
    pt_model = models['pt']['large']
    nhe_model = models['nhe']['large']
    
    print(f"\n📊 Model info:")
    print(f"   PT vocabulary: {len(pt_model.wv)} words")
    print(f"   NHE vocabulary: {len(nhe_model.wv)} words")
    
    # Word frequencies
    pt_freq = Counter()
    nhe_freq = Counter()
    for item in data:
        pt_freq.update(item['pt'].lower().split())
        nhe_freq.update(item['nhe'].lower().split())
    
    print(f"\n🔍 MOST FREQUENT WORDS:")
    print(f"\n   Portuguese top 10:")
    for word, count in pt_freq.most_common(10):
        if word in pt_model.wv:
            print(f"      {word}: {count}")
    
    print(f"\n   Nheengatu top 10:")
    for word, count in nhe_freq.most_common(10):
        if word in nhe_model.wv:
            print(f"      {word}: {count}")
    
    # Translation candidates
    print(f"\n🔗 POTENTIAL TRANSLATION PAIRS:")
    pairs = {
        'direito': 'direitu', 'liberdade': 'timaãresésá', 'justiça': 'supῖtu',
        'povo': 'miíra', 'estado': 'tetãma', 'lei': 'suãtisá', 'poder': 'kirῖbawasá'
    }
    
    for pt_w, nhe_w in pairs.items():
        if pt_w in pt_model.wv and nhe_w in nhe_model.wv:
            pt_neighbors = pt_model.wv.most_similar(pt_w, topn=3)
            nhe_neighbors = nhe_model.wv.most_similar(nhe_w, topn=3)
            print(f"\n   {pt_w} ↔ {nhe_w}")
            print(f"      PT neighbors: {pt_neighbors}")
            print(f"      NHE neighbors: {nhe_neighbors}")
    
    # Morphological analysis
    print(f"\n🔬 MORPHOLOGICAL ANALYSIS: Nheengatu Plural Formation")
    ita_words = [w for w in nhe_model.wv.key_to_index.keys() if w.endswith('ita')]
    print(f"   Words with '-ita' suffix: {len(ita_words)}")
    print(f"   Examples: {ita_words[:15]}")
    
    singular_plural = []
    for plural in ita_words:
        singular = plural[:-3]
        if singular in nhe_model.wv:
            sim = float(nhe_model.wv.similarity(singular, plural))
            singular_plural.append((singular, plural, sim))
    
    print(f"\n   Singular-plural pairs: {len(singular_plural)}")
    if singular_plural:
        print(f"   Top 10:")
        for s, p, sim in sorted(singular_plural, key=lambda x: -x[2])[:10]:
            print(f"      {s} → {p} (cosine: {sim:.3f})")
    
    # Save analysis results
    results = {
        'pt_vocab_size': len(pt_model.wv),
        'nhe_vocab_size': len(nhe_model.wv),
        'pt_top_words': [(w, c) for w, c in pt_freq.most_common(20) if w in pt_model.wv],
        'nhe_top_words': [(w, c) for w, c in nhe_freq.most_common(20) if w in nhe_model.wv],
        'singular_plural_pairs': [(s, p, float(sim)) for s, p, sim in singular_plural],
        'ita_suffix_count': len(ita_words)
    }
    
    with open(output_dir / 'word2vec_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Analysis saved to {output_dir / 'word2vec_analysis.json'}")


def generate_visualizations(models, data, output_dir):
    """Generate publication-quality plots using plotnine."""
    
    print(f"\n📊 Generating visualizations...")
    
    pt_model = models['pt']['large']
    nhe_model = models['nhe']['large']
    
    # Plot 1: PCA of frequent words
    pt_words = ['de', 'e', 'a', 'o', 'do', 'da', 'os', 'as', 'um', 'para', 'com', 'em']
    pt_words = [w for w in pt_words if w in pt_model.wv]
    pt_vecs = np.array([pt_model.wv[w] for w in pt_words])
    pca_pt = PCA(n_components=2)
    pt_2d = pca_pt.fit_transform(pt_vecs)
    
    nhe_words = ['ta', 'kuá', 'asui', 'upé', 'mayê', 'ũbeu', 'u', 'rupí', 'waá']
    nhe_words = [w for w in nhe_words if w in nhe_model.wv]
    nhe_vecs = np.array([nhe_model.wv[w] for w in nhe_words])
    pca_nhe = PCA(n_components=2)
    nhe_2d = pca_nhe.fit_transform(nhe_vecs)
    
    # Create dataframes
    pt_df = pd.DataFrame({'word': pt_words, 'PC1': pt_2d[:, 0], 'PC2': pt_2d[:, 1], 'lang': 'PT'})
    nhe_df = pd.DataFrame({'word': nhe_words, 'PC1': nhe_2d[:, 0], 'PC2': nhe_2d[:, 1], 'lang': 'NHE'})
    combined = pd.concat([pt_df, nhe_df])
    
    plot1 = (ggplot(combined, aes(x='PC1', y='PC2', color='lang')) +
             geom_point(size=3) +
             geom_text(aes(label='word'), size=8, ha='center', va='bottom') +
             facet_wrap('~lang', scales='free') +
             labs(title='PCA of Frequent Words', x='PC1', y='PC2') +
             theme_minimal() +
             theme(figure_size=(12, 6)))
    
    plot1.save(str(output_dir / 'pca_freq_words.png'), dpi=300, verbose=False)
    print("   ✅ pca_freq_words.png")
    
    # Plot 2: Singular-plural similarity
    singular_plural = []
    for word in nhe_model.wv.key_to_index.keys():
        if word.endswith('ita') and len(word) > 3:
            singular = word[:-3]
            if singular in nhe_model.wv:
                singular_plural.append({
                    'singular': singular,
                    'plural': word,
                    'similarity': float(nhe_model.wv.similarity(singular, word))
                })
    
    if singular_plural:
        sp_df = pd.DataFrame(singular_plural).sort_values('similarity', ascending=False).head(15)
        sp_df['pair'] = sp_df['singular'] + ' → ' + sp_df['plural']
        
        plot2 = (ggplot(sp_df, aes(x='reorder(pair, similarity)', y='similarity')) +
                 geom_col(fill='#A23B72', alpha=0.8) +
                 coord_flip() +
                 labs(title='Nheengatu Plural Formation (-ita suffix)', 
                      x='Singular → Plural', y='Cosine Similarity') +
                 theme_minimal() +
                 theme(figure_size=(10, 8)))
        
        plot2.save(str(output_dir / 'agglutination_similarity.png'), dpi=300, verbose=False)
        print("   ✅ agglutination_similarity.png")
    
    print(f"\n💾 Visualizations saved to {output_dir}")


def add_experiment_command(cli):
    """Add experiment command to CLI."""
    import click
    
    @cli.command('experiments')
    @click.option('--corpus', default='data/processed/sentence_pairs.json', 
                  type=click.Path(exists=True))
    @click.option('--out', default='experiments/results', type=click.Path())
    def experiments_cmd(corpus, out):
        """Run word2vec experiments on the parallel corpus."""
        if not HAS_GENSIM:
            click.echo("Error: gensim not installed. Run: pip install gensim")
            return
        
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        run_word2vec_experiment(Path(corpus), out_dir)
        click.echo(f"\n✅ Experiments complete! Results in {out_dir}")
    
    return experiments_cmd
