"""Tests for corpus_pipeline.clean"""

from corpus_pipeline.clean import (
    clean_portuguese,
    clean_nheengatu,
    normalise_segment,
    _fix_soft_hyphens,
)


def test_fix_soft_hyphens():
    assert _fix_soft_hyphens("represen-\ntantes") == "representantes"
    assert _fix_soft_hyphens("no hyphen here") == "no hyphen here"
    assert _fix_soft_hyphens("end-\nof-\nline") == "endofline"


def test_clean_portuguese_removes_form_feeds(cfg):
    text = "hello\fworld"
    result = clean_portuguese(text, cfg)
    assert "\f" not in result
    assert "hello" in result
    assert "world" in result


def test_clean_portuguese_strips_page_numbers(cfg):
    text = "Art. 1o Some text.\n\n42\n\nArt. 2o More text."
    result = clean_portuguese(text, cfg)
    # Standalone "42" should be removed
    lines = [l for l in result.split("\n") if l.strip() == "42"]
    assert lines == []


def test_clean_nheengatu_strips_running_header(cfg):
    text = (
        '"Mundu sa Turusu" waá, ũbêuwa mayé míra itá uikú arãma purãga iké braziu upé\n'
        "Art. 1º Tetãma braziu.\n"
    )
    result = clean_nheengatu(text, cfg)
    assert '"Mundu sa Turusu"' not in result
    assert "Art. 1º" in result


def test_normalise_segment_collapses_newlines(cfg):
    text = "This is\na multi-line\nsentence."
    result = normalise_segment(text, cfg)
    assert "\n" not in result
    assert "This is a multi-line sentence." == result


def test_normalise_segment_collapses_spaces(cfg):
    text = "too    many    spaces"
    result = normalise_segment(text, cfg)
    assert result == "too many spaces"
