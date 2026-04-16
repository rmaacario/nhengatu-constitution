# ============================================================
# Constitution Corpus Pipeline — Makefile
# ============================================================
.DEFAULT_GOAL := help
.PHONY: help install install-dev run check test test-unit test-integration \
        lint clean distclean

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
	@echo "  make install          Install runtime dependencies"
	@echo "  make install-dev      Install runtime + dev dependencies"
	@echo "  make run              Run the full pipeline"
	@echo "  make check            Validate PDFs without writing output"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only (no PDFs needed)"
	@echo "  make test-integration Run integration tests (requires PDFs)"
	@echo "  make lint             Run ruff linter"
	@echo "  make clean            Remove output files"
	@echo "  make distclean        Remove output files and caches"
	@echo ""

# ---------------------------------------------------------------
# Installation
# ---------------------------------------------------------------
install:
	$(PIP) install -r requirements.txt --break-system-packages

install-dev:
	$(PIP) install -r requirements.txt --break-system-packages
	$(PIP) install -e . --break-system-packages

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
	@echo "  Copy the PDF to data/raw/ or set PT_PDF= on the command line."
	@exit 1

$(NHE_PDF):
	@echo "ERROR: Nheengatu PDF not found at $(NHE_PDF)"
	@echo "  Copy the PDF to data/raw/ or set NHE_PDF= on the command line."
	@exit 1

# ---------------------------------------------------------------
# Testing
# ---------------------------------------------------------------
test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v --tb=short

test-unit:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v --tb=short \
		--ignore=tests/test_pipeline.py

test-integration:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_pipeline.py -v --tb=short

test-cov:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ \
		--ignore=tests/test_pipeline.py \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

# ---------------------------------------------------------------
# Lint
# ---------------------------------------------------------------
lint:
	$(PYTHON) -m ruff check $(SRC) tests/

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

.PHONY: exp-word2vec exp-fasttext exp-crosslingual exp-viz exp-all

# Word2Vec experiments
exp-word2vec:
	@echo "=== Training Word2Vec models ==="
	cd experiments/01_word2vec && python train_word2vec.py --corpus ../../data/processed/corpus_merged.nhe --output results/
	cd experiments/01_word2vec && python train_word2vec.py --corpus ../../data/processed/corpus_merged.pt --output results/
	@echo "✅ Word2Vec models saved to experiments/01_word2vec/results/"

# FastText experiments
exp-fasttext:
	@echo "=== Training FastText models ==="
	cd experiments/02_fasttext && python train.py --corpus ../../data/processed/corpus_merged.nhe --output results/
	cd experiments/02_fasttext && python train.py --corpus ../../data/processed/corpus_merged.pt --output results/
	@echo "✅ FastText models saved to experiments/02_fasttext/results/"

# Cross-lingual alignment
exp-crosslingual:
	@echo "=== Cross-lingual alignment ==="
	cd experiments/03_crosslingual && python align_vecmap.py \
		--src_emb ../01_word2vec/results/pt_w2v.model \
		--tgt_emb ../01_word2vec/results/nhe_w2v.model
	@echo "✅ Alignment done. Results in experiments/03_crosslingual/results/"

# XLM-R fine-tuning (requires GPU)
exp-xlmr:
	@echo "=== XLM-R fine-tuning (GPU recommended) ==="
	@echo "Running on Colab is recommended:"
	@echo "  https://colab.research.google.com/github/rmaacario/nhengatu-constitution/blob/main/notebooks/colab/finetune_xlmr.ipynb"
	@echo ""
	@echo "For local training (if GPU available):"
	cd experiments/03_crosslingual && python finetune_xlmr.py \
		--train ../../data/processed/xlmr_train_clean.json \
		--output results/xlmr_finetuned

# Visualization
exp-viz:
	@echo "=== Generating visualizations ==="
	cd experiments/05_visualization && python plot_semantic_spaces.py
	@echo "✅ Plots saved to experiments/05_visualization/plots/"

# Typological analysis
exp-typology:
	@echo "=== Typological analysis ==="
	cd experiments/06_typological_analysis && python compare_por_yrl.py
	@echo "✅ Results saved to por_yrl_comparison.tsv"

# Run all experiments (except XLM-R which needs GPU)
exp-all: exp-word2vec exp-fasttext exp-viz exp-typology
	@echo ""
	@echo "=== All experiments completed! ==="
	@echo "Word2Vec: experiments/01_word2vec/results/"
	@echo "FastText: experiments/02_fasttext/results/"
	@echo "Visualizations: experiments/05_visualization/plots/"
	@echo "Typology: por_yrl_comparison.tsv"
	@echo ""
	@echo "For XLM-R fine-tuning, use: make exp-xlmr (GPU required) or run on Colab"

# Quick test with small data
exp-test:
	@echo "=== Testing experiments with small sample ==="
	@echo "Not implemented yet - use exp-all for full run"
