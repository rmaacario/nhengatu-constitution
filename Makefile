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
