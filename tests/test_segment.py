"""Tests for corpus_pipeline.segment"""

import pytest
from corpus_pipeline.clean import clean_portuguese, clean_nheengatu
from corpus_pipeline.segment import (
    locate_body_portuguese,
    locate_body_nheengatu,
    extract_articles,
    extract_preamble,
    split_units,
    Article,
    Body,
)


# ---------------------------------------------------------------------------
# locate_body_portuguese
# ---------------------------------------------------------------------------

def test_locate_body_portuguese_finds_start(cfg, pt_sample):
    text = clean_portuguese(pt_sample, cfg)
    body = locate_body_portuguese(text, cfg)
    assert body.start > 0
    assert "Nós, representantes" in body.content


def test_locate_body_portuguese_finds_end(cfg, pt_sample):
    text = clean_portuguese(pt_sample, cfg)
    body = locate_body_portuguese(text, cfg)
    # ADCT content should NOT be in the body
    assert "eleições municipais" not in body.content


def test_locate_body_portuguese_raises_on_bad_text(cfg):
    with pytest.raises(ValueError, match="PREÂMBULO"):
        locate_body_portuguese("no preamble here", cfg)


# ---------------------------------------------------------------------------
# locate_body_nheengatu
# ---------------------------------------------------------------------------

def test_locate_body_nheengatu_finds_start(cfg, nhe_sample):
    text = clean_nheengatu(nhe_sample, cfg)
    body = locate_body_nheengatu(text, cfg)
    assert "Tetãma braziu" in body.content


def test_locate_body_nheengatu_raises_when_heading_missing(cfg):
    with pytest.raises(ValueError, match="YUPIRUNGÁ RẼDEWÁ"):
        locate_body_nheengatu("no heading here", cfg)


# ---------------------------------------------------------------------------
# extract_articles
# ---------------------------------------------------------------------------

def test_extract_articles_portuguese(cfg, pt_sample):
    text = clean_portuguese(pt_sample, cfg)
    body = locate_body_portuguese(text, cfg)
    arts = extract_articles(body, lang="pt", cfg=cfg)
    assert 1 in arts
    assert 2 in arts
    assert 3 in arts
    assert arts[1].lang == "pt"


def test_extract_articles_nheengatu(cfg, nhe_sample):
    text = clean_nheengatu(nhe_sample, cfg)
    body = locate_body_nheengatu(text, cfg)
    arts = extract_articles(body, lang="nhe", cfg=cfg)
    assert 1 in arts
    assert 2 in arts
    assert arts[1].lang == "nhe"


def test_extract_articles_content(cfg, pt_sample):
    text = clean_portuguese(pt_sample, cfg)
    body = locate_body_portuguese(text, cfg)
    arts = extract_articles(body, lang="pt", cfg=cfg)
    assert "soberania" in arts[1].raw.lower()
    assert "Legislativo" in arts[2].raw


# ---------------------------------------------------------------------------
# extract_preamble
# ---------------------------------------------------------------------------

def test_extract_preamble_portuguese(cfg, pt_sample):
    text = clean_portuguese(pt_sample, cfg)
    body = locate_body_portuguese(text, cfg)
    preamble = extract_preamble(body, lang="pt", cfg=cfg)
    assert preamble is not None
    assert "Assembleia Nacional Constituinte" in preamble


def test_extract_preamble_nheengatu(cfg, nhe_sample):
    text = clean_nheengatu(nhe_sample, cfg)
    body = locate_body_nheengatu(text, cfg)
    preamble = extract_preamble(body, lang="nhe", cfg=cfg)
    assert preamble is not None
    assert "yaãmuyakũtasá" in preamble


# ---------------------------------------------------------------------------
# split_units
# ---------------------------------------------------------------------------

def _make_article(number: int, raw: str, lang: str) -> Article:
    return Article(number=number, raw=raw, lang=lang)


def test_split_units_caput_only(cfg):
    art = _make_article(2, "Art. 2o São Poderes da União, o Legislativo.", "pt")
    units = split_units(art, cfg)
    assert len(units) == 1
    assert units[0].unit_type == "caput"


def test_split_units_with_paragrafo_unico(cfg):
    raw = (
        "Art. 1o A República.\n"
        "Parágrafo único. Todo o poder emana do povo."
    )
    art = _make_article(1, raw, "pt")
    units = split_units(art, cfg)
    types = [u.unit_type for u in units]
    assert "caput" in types
    assert "paragrafo_unico" in types


def test_split_units_with_numbered_paragraphs(cfg):
    raw = (
        "Art. 3o Constituem objetivos:\n"
        "§ 1o Primeiro parágrafo.\n"
        "§ 2o Segundo parágrafo."
    )
    art = _make_article(3, raw, "pt")
    units = split_units(art, cfg)
    assert len(units) == 3
    assert units[0].unit_type == "caput"
    assert units[1].unit_type == "paragrafo"
    assert units[2].unit_type == "paragrafo"


def test_split_units_nhe_paragrafo_unico(cfg):
    raw = (
        "Art. 1º Tetãma braziu.\n"
        "Ũbeusá yepéyũtu. Kirῖbawasá uri míra-itá sui."
    )
    art = _make_article(1, raw, "nhe")
    units = split_units(art, cfg)
    types = [u.unit_type for u in units]
    assert "paragrafo_unico" in types
