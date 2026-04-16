#!/usr/bin/env python3
"""Run sentence-level pipeline without installing."""

import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corpus_pipeline.cli import main

if __name__ == "__main__":
    # Default to sentences command if no args
    if len(sys.argv) == 1:
        sys.argv.extend(["sentences", "--help"])
    
    main()