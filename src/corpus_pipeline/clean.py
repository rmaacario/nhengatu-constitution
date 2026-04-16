"""
clean.py — raw PDF text → clean text ready for segmentation.
"""
from __future__ import annotations
import re
from corpus_pipeline.config import Config


def clean_portuguese(text: str, cfg: Config) -> str:
    text = text.replace("\f", "\n")
    if cfg.normalisation.strip_page_numbers:
        text = re.sub(cfg.normalisation.page_number_pattern, "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_nheengatu(text: str, cfg: Config) -> str:
    # Running page header (also removes embedded page number)
    text = re.sub(cfg.nheengatu.running_header_pattern, "\n", text)
    # BEL chars in preface
    if cfg.nheengatu.strip_bell:
        text = text.replace("\x07", "")
    # Remaining form-feeds
    text = text.replace("\f", "\n")
    # Article header typo fixups (from audit)
    for find, replace in cfg.articles.nhe_header_fixups:
        text = text.replace(find, replace)
    if cfg.normalisation.strip_page_numbers:
        text = re.sub(cfg.normalisation.page_number_pattern, "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalise_segment(text: str, cfg: Config) -> str:
    """Flatten a multi-line segment to one clean line for TSV / Moses output."""
    if cfg.normalisation.fix_soft_hyphens:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    if cfg.normalisation.collapse_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()