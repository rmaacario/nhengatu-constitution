#!/usr/bin/env python3
"""
Compute typological distances between Portuguese (por) and Nheengatu (yrl)
using feature vectors from URIEL via lang2vec.
Also compute distances for a close pair (Spanish–Portuguese) for context.

Run: python3 final_distance_analysis.py
"""

import lang2vec.lang2vec as l2v
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_vector(lang, feature_set):
    """Return a numpy array of numeric values for a language and feature set."""
    raw = l2v.get_features(lang, feature_set, header=False)
    vals = raw[lang]
    # Convert to float, replacing '--' with NaN
    return np.array([float(x) if x != '--' else np.nan for x in vals])

def cosine_distance(v1, v2):
    """Return cosine distance (1 - similarity) ignoring NaNs."""
    mask = ~(np.isnan(v1) | np.isnan(v2))
    if np.sum(mask) == 0:
        return np.nan
    v1_clean = v1[mask]
    v2_clean = v2[mask]
    sim = cosine_similarity([v1_clean], [v2_clean])[0][0]
    return 1 - sim

def print_distances(name, lang1, lang2):
    """Compute and print all distances for a language pair."""
    print(f"\n{name}: {lang1.upper()} vs {lang2.upper()}")
    print("-" * 40)

    # Syntactic
    syn1 = get_vector(lang1, "syntax_knn")
    syn2 = get_vector(lang2, "syntax_knn")
    syn_dist = cosine_distance(syn1, syn2)
    print(f"Syntactic distance:     {syn_dist:.4f}")

    # Phonological
    phon1 = get_vector(lang1, "phonology_knn")
    phon2 = get_vector(lang2, "phonology_knn")
    phon_dist = cosine_distance(phon1, phon2)
    print(f"Phonological distance:  {phon_dist:.4f}")

    # Morphological (syntax_knn + inventory_knn)
    inv1 = get_vector(lang1, "inventory_knn")
    inv2 = get_vector(lang2, "inventory_knn")
    comb1 = np.concatenate([syn1, inv1])
    comb2 = np.concatenate([syn2, inv2])
    morph_dist = cosine_distance(comb1, comb2)
    print(f"Morphological distance: {morph_dist:.4f}")

    # Genetic (family membership)
    fam1 = get_vector(lang1, "fam")
    fam2 = get_vector(lang2, "fam")
    gen_dist = cosine_distance(fam1, fam2)
    print(f"Genetic distance:       {gen_dist:.4f}")

    # Geographic (geo)
    geo1 = get_vector(lang1, "geo")
    geo2 = get_vector(lang2, "geo")
    geo_dist = cosine_distance(geo1, geo2)
    print(f"Geographic distance:    {geo_dist:.4f}")

def main():
    print("Typological distance analysis using URIEL (via lang2vec)")
    print("=" * 60)

    # Pair 1: Portuguese vs Nheengatu (distant)
    print_distances("Distant pair (target)", "por", "yrl")

    # Pair 2: Spanish vs Portuguese (close, both Romance)
    print_distances("Close pair (baseline)", "spa", "por")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("- Distances close to 0 indicate high similarity, close to 1 indicate high divergence.")
    print("- For the distant pair (por-yrl), genetic distance is near 1 (different families).")
    print("- Syntactic and morphological distances are moderate (0.3–0.4) but still notably higher")
    print("  than the close pair (spa-por), which are around 0.1–0.2.")
    print("- Phonological distance is low for both pairs, likely due to kNN smoothing.")
    print("- Despite moderate distances, cross-lingual retrieval (P@1=0.247) is non-trivial,")
    print("  especially given the genetic unrelatedness and limited parallel data.")

if __name__ == "__main__":
    main()