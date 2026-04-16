#!/usr/bin/env python3
"""
scripts/run_pipeline.py
-----------------------
Run the pipeline without installing the package.

    python scripts/run_pipeline.py \
        --pt  data/raw/constituicao-portuguesa.pdf \
        --nhe data/raw/constituicao-nheengatu-web.pdf

All arguments are optional; defaults come from config/config.yaml.
"""

import sys
from pathlib import Path

# Put src/ on the path so we can import corpus_pipeline without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corpus_pipeline.cli import main

if __name__ == "__main__":
    main()
