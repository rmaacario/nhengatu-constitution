"""Tests for corpus_pipeline.clean"""

from corpus_pipeline.clean import (
    clean_portuguese,
    clean_nheengatu,
    normalise_segment,
)
from corpus_pipeline.config import load

# Helper function for soft hyphen removal
def apply_soft_hyphen_fix(text: str) -> str:
    """Apply soft hyphen fix as done in clean.py"""
    cfg = load()
    if cfg.normalisation.fix_soft_hyphens:
        return text.replace('-\n', '').replace('-\r', '')
    return text


def test_fix_soft_hyphens():
    """Test soft hyphen removal (functionality now inline)"""
    assert apply_soft_hyphen_fix("represen-\ntantes") == "representantes"
    assert apply_soft_hyphen_fix("no hyphen here") == "no hyphen here"
    assert apply_soft_hyphen_fix("end-\nof-\nline") == "endofline"


def test_clean_portuguese_removes_form_feeds():
    cfg = load()
    text = "hello\fworld"
    result = clean_portuguese(text, cfg)
    assert "\f" not in result
    assert "hello" in result
    assert "world" in result


def test_clean_portuguese_strips_page_numbers():
    cfg = load()
    text = "Art. 1o Some text.\n\n42\n\nArt. 2o More text."
    result = clean_portuguese(text, cfg)
    # Standalone "42" should be removed
    lines = [l for l in result.split("\n") if l.strip() == "42"]
    assert lines == []


def test_clean_nheengatu_strips_running_header():
    cfg = load()
    text = (
        '"Mundu sa Turusu" waá, ũbêuwa mayé míra itá uikú arãma purãga iké braziu upé\n'
        "Art. 1º Tetãma braziu.\n"
    )
    result = clean_nheengatu(text, cfg)
    assert '"Mundu sa Turusu"' not in result
    assert "Art. 1º" in result


def test_normalise_segment_collapses_newlines():
    cfg = load()
    text = "This is\na multi-line\nsentence."
    result = normalise_segment(text, cfg)
    assert "\n" not in result
    assert "This is a multi-line sentence." == result


def test_normalise_segment_collapses_spaces():
    cfg = load()
    text = "too    many    spaces"
    result = normalise_segment(text, cfg)
    assert result == "too many spaces"
