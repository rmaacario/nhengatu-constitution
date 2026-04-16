# Constitution Corpus Pipeline

A bilingual parallel corpus pipeline that extracts Portuguese–Nheengatu aligned text from the Brazilian Constitution (1988).

---

## 📚 Corpus Sources

- **Portuguese**: Senado Federal edition, compiled up to EC 116/2022  
- **Nheengatu**: *Mundu Sa Turusu Waá* translation (CNJ/STF, 2023)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler
```

### Installation

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

## 📊 Output Files

Outputs are written to `output/`:

| File | Description | Count |
|------|------------|------|
| corpus_articles.json | Article-level pairs | 250 |
| corpus_units.json | Paragraph-level pairs | ~860 |
| corpus_units.tsv | TSV format | ~860 |
| corpus.pt / corpus.nhe | Moses format | ~860 |
| alignment_report.json | Quality metrics | — |

---

## 📁 Project Structure

```bash
constitution-corpus/
├── src/corpus_pipeline/   # Pipeline code
├── sentence_output/       # Sentence alignments (~5k pairs)
├── data/
│   ├── processed/         # Final corpus
│   └── raw/               # Original PDFs (git-ignored)
├── experiments/           # Research experiments
├── tests/                 # Unit tests
├── config/                # Configuration
└── scripts/               # Utilities
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
# Run with defaults
corpus-pipeline run

# Custom paths
corpus-pipeline run \
  --pt data/raw/constituicao-pt.pdf \
  --nhe data/raw/constituicao-nhe.pdf

# Validate only
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

The pipeline automatically corrects known issues in the Nheengatu PDF:

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
make test-unit        # Unit tests only
make test-integration # Requires PDFs
make test             # All tests
make test-cov         # Coverage report
```

---

## 📝 Known Limitations

- Incisos not separately aligned
- Some soft hyphens may remain
- Translation is not word-for-word
- ADCT not included

---

## 🤝 Contributing

- Fork repository
- Create feature branch
- Add tests
- Run `make test`
- Submit pull request

---

## 📄 License

[Your license]

---

## 📧 Contact

[Your contact information]