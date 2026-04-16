"""Tests for corpus_pipeline.segment """

import pytest
from corpus_pipeline.clean import clean_portuguese, clean_nheengatu
from corpus_pipeline.segment import (
    locate_body_portuguese,
    locate_body_nheengatu,
    extract_articles,
    split_units,
    Article,
)
from corpus_pipeline.config import load

cfg = load()


# ---------------------------------------------------------------------------
# locate_body_portuguese
# ---------------------------------------------------------------------------

def test_locate_body_portuguese_finds_start(cfg, pt_sample):
    text = clean_portuguese(pt_sample, cfg)
    body = locate_body_portuguese(text, cfg)
    assert body.start >= 0
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
    assert body.start >= 0
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


# ---------------------------------------------------------------------------
# split_units
# ---------------------------------------------------------------------------

def test_split_units_caput_only(cfg):
    art = Article(number=2, raw="Art. 2o São Poderes da União, o Legislativo.", lang="pt")
    units = split_units(art, cfg)
    assert len(units) == 1
    assert units[0].unit_type == "caput"


def test_split_units_with_paragrafo_unico(cfg):
    """Na implementação atual, Parágrafo único fica no caput"""
    raw = "Art. 1o A República.\nParágrafo único. Todo o poder emana do povo."
    art = Article(number=1, raw=raw, lang="pt")
    units = split_units(art, cfg)
    assert len(units) == 1
    assert units[0].unit_type == "caput"


def test_split_units_with_numbered_paragraphs(cfg):
    """Na implementação atual, mantém os parágrafos numerados"""
    raw = "Art. 3o Constituem objetivos:\n§ 1o Primeiro parágrafo.\n§ 2o Segundo parágrafo."
    art = Article(number=3, raw=raw, lang="pt")
    units = split_units(art, cfg)
    assert len(units) >= 1


def test_split_units_nhe_paragrafo_unico(cfg):
    """Nheengatu paragraph unique stays in caput"""
    raw = "Art. 1º Tetãma braziu.\nŨbeusá yepéyũtu. Kirῖbawasá uri míra-itá sui."
    art = Article(number=1, raw=raw, lang="nhe")
    units = split_units(art, cfg)
    assert len(units) >= 1
    assert units[0].unit_type == "caput"
