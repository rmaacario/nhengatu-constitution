# Nheengatu Constitution Corpus

> **Dataset**: 5,028 Portuguese–Nheengatu sentence pairs  
> **Focus**: low-resource NLP, cross-lingual alignment, indigenous language preservation

A bilingual parallel corpus pipeline that extracts and aligns Portuguese–Nheengatu text from the Brazilian Constitution (1988).

---

## 📚 Corpus Sources

- **Portuguese**: Senado Federal edition (EC 116/2022)  
- **Nheengatu**: *Mundu Sa Turusu Waá* (CNJ/STF, 2023)

---

## 📊 Current Data

- **5,028 sentence pairs** → `data/processed/merged_5028_pairs.json`
- **4,997 sentence pairs** → `data/processed/corpus_clean.json` (cleaned for fine-tuning)
- **High-confidence subset (~671 pairs)** → `sentence_output/`
- **Word2Vec models** → `experiments/01_word2vec/results/`
- **XLM-R model** → `experiments/03_crosslingual/results/`

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler
```

### Installation & Run

```bash
# Clone repository
git clone <your-repo-url>
cd constitution-corpus

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
make install

# Download source PDFs
# Portuguese Constitution
wget https://www2.senado.leg.br/bdsf/bitstream/handle/id/602786/CF_EC116_2022.pdf

# Nheengatu Constitution
wget https://www.cnj.jus.br/wp-content/uploads/2023/11/constituicao-nheengatu-web.pdf

# Move and rename
mv CF_EC116_2022.pdf data/raw/constituicao-pt.pdf
mv constituicao-nheengatu-web.pdf data/raw/constituicao-nhe.pdf

# Run pipeline
make run
```

---

## 📁 Project Structure

```bash
nhengatu-constitution/
├── src/corpus_pipeline/     # Pipeline code (extract, clean, align, export)
├── data/
│   ├── processed/           
│   │   ├── corpus_clean.json      (4,997 pairs) ← XLM-R fine-tuning
│   │   ├── corpus_merged.json      (5,028 pairs) ← raw merged corpus
│   │   ├── merged_5028_pairs.json  (5,028 pairs) ← final aligned pairs
│   │   └── xlmr_train_clean.json   (4,997 pairs) ← training data
│   └── raw/                       # Original PDFs (git-ignored)
│       ├── constituicao-pt.pdf
│       └── constituicao-nhe.pdf
├── sentence_output/               # Current sentence alignments (671 pairs)
│   ├── sentence_pairs.json
│   ├── sentence_pairs.tsv
│   └── train.nhe / train.pt
├── experiments/                   # Research experiments
│   ├── 01_word2vec/              # Word embeddings
│   ├── 02_fasttext/              # FastText models (with symlinks to data/)
│   ├── 03_crosslingual/          # VecMap & XLM-R fine-tuning
│   ├── 05_visualization/         # Semantic visualizations
│   └── 06_typological_analysis/  # Language comparison
├── notebooks/colab/               # Google Colab notebooks
│   └── finetune_xlmr.ipynb       # XLM-R fine-tuning demo
├── tests/                         # Unit tests
├── config/                        # Configuration files
│   └── config.yaml
├── output/                        # Pipeline outputs (article/unit alignment)
├── scripts/                       # Utility scripts
├── Makefile                       # Build automation
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
```

---

## 🛠️ Usage

```bash
make run      # Run full pipeline
make test     # Run tests
make clean    # Clean cache
make install  # Install dependencies
```

### CLI

```bash
corpus-pipeline run

corpus-pipeline run \
  --pt data/raw/constituicao-pt.pdf \
  --nhe data/raw/constituicao-nhe.pdf

corpus-pipeline check \
  --pt data/raw/constituicao-pt.pdf \
  --nhe data/raw/constituicao-nhe.pdf
```

---

## ⚙️ Configuration

Edit `config/config.yaml`:

- PDF extraction settings
- Alignment parameters (`max_unit_count_delta`, `split_markers`)
- Output formats (`json`, `tsv`, `csv`)

---

## 🔧 Known Fixups

- `Art 102.` → `Art. 102.`
- `Art.. 158.` → `Art. 158.`
- `Art. 141.Upawá` → `Art. 141. Upawá`

---

## 📈 Alignment Quality

| Quality | Articles | % |
|--------|---------|---|
| Perfect | ~202 | 81% |
| Partial | ~43 | 17% |
| Fallback | ~5 | 2% |

---

## 🧪 Testing

```bash
make test-unit
make test-integration
make test
make test-cov
```

---

## 📝 Known Limitations

- Incisos not separately aligned
- Some soft hyphens may remain
- Translation is not word-for-word
- ADCT not included

---

## 📄 License

MIT License

Copyright (c) 2024 Rafael Macário

---

## 📧 Contact

Rafael Macário  
Email: [rafael.macario@usp.br](mailto:rafael.macario@usp.br)  
Repository: https://github.com/rmaacario/nhengatu-constitution