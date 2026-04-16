# ============================================================
# Constitution Corpus Pipeline — Makefile
# ============================================================
.DEFAULT_GOAL := help
.PHONY: help install install-dev run check test test-unit test-integration \
        lint clean distclean exp-word2vec exp-fasttext exp-crosslingual \
        exp-xlmr exp-viz exp-typology exp-all

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
PYTHON     := python3
PIP        := $(PYTHON) -m pip
SRC        := src/corpus_pipeline
CFG        := config/config.yaml
PT_PDF     := data/raw/constituicao-pt.pdf
NHE_PDF    := data/raw/constituicao-nhe.pdf
OUT_DIR    := output

# ---------------------------------------------------------------
# Help
# ---------------------------------------------------------------
help:
	@echo ""
	@echo "  Constitution Corpus Pipeline"
	@echo ""
	@echo "  MAIN PIPELINE:"
	@echo "  make install          Install runtime dependencies"
	@echo "  make install-dev      Install runtime + dev dependencies"
	@echo "  make run              Run the full pipeline"
	@echo "  make check            Validate PDFs without writing output"
	@echo ""
	@echo "  TESTING:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only (no PDFs needed)"
	@echo "  make test-integration Run integration tests (requires PDFs)"
	@echo ""
	@echo "  EXPERIMENTS:"
	@echo "  make exp-word2vec     Train Word2Vec models"
	@echo "  make exp-fasttext     Train FastText models"
	@echo "  make exp-xlmr         Fine-tune XLM-R (Colab recommended)"
	@echo "  make exp-viz          Generate visualizations"
	@echo "  make exp-typology     Run typological analysis"
	@echo "  make exp-all          Run all compatible experiments"
	@echo ""
	@echo "  MAINTENANCE:"
	@echo "  make lint             Run ruff linter"
	@echo "  make clean            Remove output files"
	@echo "  make distclean        Remove output files and caches"
	@echo ""

# ---------------------------------------------------------------
# Installation
# ---------------------------------------------------------------
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------
run: $(PT_PDF) $(NHE_PDF)
	$(PYTHON) scripts/run_pipeline.py run \
		--pt  $(PT_PDF) \
		--nhe $(NHE_PDF) \
		--out $(OUT_DIR) \
		--config $(CFG)

check: $(PT_PDF) $(NHE_PDF)
	$(PYTHON) scripts/run_pipeline.py check \
		--pt  $(PT_PDF) \
		--nhe $(NHE_PDF) \
		--config $(CFG)

$(PT_PDF):
	@echo "ERROR: Portuguese PDF not found at $(PT_PDF)"
	@echo "  Run: make download-pdfs or place PDF manually"
	@exit 1

$(NHE_PDF):
	@echo "ERROR: Nheengatu PDF not found at $(NHE_PDF)"
	@echo "  Run: make download-pdfs or place PDF manually"
	@exit 1

download-pdfs:
	@echo "=== Downloading PDFs ==="
	curl -L -A "Mozilla/5.0" -o data/raw/constituicao-pt.pdf \
	  "https://www.gov.br/defesa/pt-br/acesso-a-informacao/governanca/governanca-do-setor-de-defesa/legislacao-basica-1/arquivos/2022/constituicao-da-republica-federativa-do-brasil.pdf/@@download/file"
	curl -L -o data/raw/constituicao-nhe.pdf \
	  "https://bibliotecadigital.mdh.gov.br/jspui/bitstream/192/12153/1/constituicao-nheengatu-web.pdf"
	@echo "✅ PDFs downloaded to data/raw/"

# ---------------------------------------------------------------
# Testing
# ---------------------------------------------------------------
test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v --tb=short

test-unit:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_clean.py tests/test_frontmatter.py -v --tb=short

test-integration:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_segment.py tests/test_sentence_split.py -v --tb=short

test-cov:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

# ---------------------------------------------------------------
# Lint
# ---------------------------------------------------------------
lint:
	$(PYTHON) -m ruff check $(SRC)/ tests/ 2>/dev/null || echo "ruff not installed, run: pip install ruff"

# ---------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------
clean:
	rm -rf $(OUT_DIR)/*.json $(OUT_DIR)/*.tsv $(OUT_DIR)/*.csv \
	       $(OUT_DIR)/corpus.pt $(OUT_DIR)/corpus.nhe

distclean: clean
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__ \
	       .pytest_cache htmlcov .coverage \
	       *.egg-info src/*.egg-info

# ============================================================
# EXPERIMENTS
# ============================================================

exp-word2vec:
	@echo "=== Training Word2Vec models ==="
	cd experiments/01_word2vec && python train_word2vec.py

exp-fasttext:
	@echo "=== FastText Training ==="
	@echo "⚠️ FastText requires Linux. On macOS, use Colab."
	@echo "Pre-trained models exist in experiments/02_fasttext/results/"

exp-xlmr:
	@echo "=== XLM-R Fine-tuning ==="
	@echo "👉 Recommended: Run on Google Colab"
	@echo "   https://colab.research.google.com/github/rmaacario/nhengatu-constitution/blob/main/notebooks/colab/finetune_xlmr.ipynb"

exp-viz:
	@echo "=== Generating Visualizations ==="
	cd experiments/05_visualization && python plot_semantic_spaces.py

exp-typology:
	@echo "=== Typological Analysis ==="
	cd experiments/06_typological_analysis && python compare_por_yrl.py

exp-all: exp-word2vec exp-viz exp-typology
	@echo ""
	@echo "=== All compatible experiments completed! ==="