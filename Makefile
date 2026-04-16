# ============================================================
# Constitution Corpus Pipeline — Makefile
# ============================================================
.DEFAULT_GOAL := help
.PHONY: help install install-dev run check test test-unit test-integration
.PHONY: lint clean distclean exp-word2vec exp-fasttext exp-crosslingual
.PHONY: exp-xlmr exp-viz exp-typology exp-all download-pdfs

# Paths
PYTHON     := python3
PIP        := $(PYTHON) -m pip
SRC        := src/corpus_pipeline
CFG        := config/config.yaml
PT_PDF     := data/raw/constituicao-pt.pdf
NHE_PDF    := data/raw/constituicao-nhe.pdf
OUT_DIR    := output

# Help
help:
	@echo ""
	@echo "  Constitution Corpus Pipeline"
	@echo ""
	@echo "  MAIN PIPELINE:"
	@echo "  make install          Install dependencies"
	@echo "  make run              Run full pipeline"
	@echo "  make check            Validate PDFs"
	@echo ""
	@echo "  TESTING:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests"
	@echo ""
	@echo "  EXPERIMENTS:"
	@echo "  make exp-word2vec     Train Word2Vec models"
	@echo "  make exp-fasttext     Train FastText models"
	@echo "  make exp-xlmr         XLM-R fine-tuning (Colab)"
	@echo "  make exp-viz          Generate visualizations"
	@echo "  make exp-typology     Typological analysis"
	@echo "  make exp-all          Run all experiments"
	@echo ""
	@echo "  MAINTENANCE:"
	@echo "  make clean            Remove output files"
	@echo ""

# Installation
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# Pipeline
run: $(PT_PDF) $(NHE_PDF)
	$(PYTHON) scripts/run_pipeline.py run \
		--pt $(PT_PDF) --nhe $(NHE_PDF) \
		--out $(OUT_DIR) --config $(CFG)

check: $(PT_PDF) $(NHE_PDF)
	$(PYTHON) scripts/run_pipeline.py check \
		--pt $(PT_PDF) --nhe $(NHE_PDF) --config $(CFG)

download-pdfs:
	@mkdir -p data/raw
	curl -L -A "Mozilla/5.0" -o $(PT_PDF) \
	  "https://www.gov.br/defesa/pt-br/acesso-a-informacao/governanca/governanca-do-setor-de-defesa/legislacao-basica-1/arquivos/2022/constituicao-da-republica-federativa-do-brasil.pdf/@@download/file"
	curl -L -o $(NHE_PDF) \
	  "https://bibliotecadigital.mdh.gov.br/jspui/bitstream/192/12153/1/constituicao-nheengatu-web.pdf"

$(PT_PDF):
	@echo "ERROR: Portuguese PDF not found. Run 'make download-pdfs'"
	@exit 1

$(NHE_PDF):
	@echo "ERROR: Nheengatu PDF not found. Run 'make download-pdfs'"
	@exit 1

# Testing
test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v --tb=short

test-unit:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_clean.py tests/test_frontmatter.py -v

# Cleanup
clean:
	rm -rf $(OUT_DIR)/*.json $(OUT_DIR)/*.tsv $(OUT_DIR)/*.csv $(OUT_DIR)/corpus.*

distclean: clean
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__ .pytest_cache

# ============================================================
# EXPERIMENTS
# ============================================================

exp-word2vec:
	@echo "=== Training Word2Vec ==="
	cd experiments/01_word2vec && python train_word2vec.py

exp-crosslingual:
	@echo "=== Cross-lingual Alignment ==="
	@echo "Run: cd experiments/03_crosslingual && python align_vecmap.py"

exp-xlmr:
	@echo "=== XLM-R Fine-tuning ==="
	@echo "Run on Colab: https://colab.research.google.com/github/rmaacario/nhengatu-constitution/blob/main/notebooks/colab/finetune_xlmr.ipynb"

exp-viz:
	@echo "=== Visualizations ==="
	cd experiments/05_visualization && python plot_semantic_spaces.py

exp-typology:
	@echo "=== Typological Analysis ==="
	cd experiments/06_typological_analysis && python compare_por_yrl.py

exp-all: exp-word2vec exp-viz exp-typology
	@echo "✅ All compatible experiments done!"

# ============================================================
# DATA PREPARATION for FastText
# ============================================================

prepare-fasttext:
	@echo "=== Preparing FastText corpora ==="
	@if [ ! -f data/processed/corpus_clean.pt ]; then \
		echo "Creating corpus_clean.pt from corpus_clean.json..."; \
		python -c "import json; data=json.load(open('data/processed/corpus_clean.json')); open('data/processed/corpus_clean.pt','w').write('\n'.join([d['pt'] for d in data]))"; \
		echo "✅ corpus_clean.pt created"; \
	fi
	@if [ ! -f data/processed/corpus_clean.nhe ]; then \
		echo "Creating corpus_clean.nhe from corpus_clean.json..."; \
		python -c "import json; data=json.load(open('data/processed/corpus_clean.json')); open('data/processed/corpus_clean.nhe','w').write('\n'.join([d['nhe'] for d in data]))"; \
		echo "✅ corpus_clean.nhe created"; \
	fi
	@echo "✅ FastText data ready"

# Update exp-fasttext to prepare data first
exp-fasttext:
	@echo "=== FastText Training ==="
	@cd experiments/02_fasttext && \
		[ -f corpus_clean.pt ] || python -c "import json; d=json.load(open('../../data/processed/corpus_clean.json')); open('corpus_clean.pt','w').write('\n'.join([x['pt'] for x in d]))"
	cd experiments/02_fasttext && python train.py
	@echo ""
	@echo "=== Exploring FastText Models ==="
	cd experiments/02_fasttext && python explore_clean.py
