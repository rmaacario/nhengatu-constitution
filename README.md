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

# Place PDFs in data/raw/
cp /path/to/constituicao-pt.pdf data/raw/
cp /path/to/constituicao-nhe.pdf data/raw/

# Run pipeline
make run
```

---

## 📁 Project Structure

```bash
constitution-corpus/
├── src/              # Pipeline code
├── data/
│   ├── processed/    # Final dataset (5,028 pairs)
│   └── raw/          # Original PDFs
├── sentence_output/  # Current alignments
├── experiments/      # Research experiments
├── tests/            # Unit tests
└── config/           # Configuration
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