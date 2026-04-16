"""Tests for frontmatter extraction."""

from corpus_pipeline.frontmatter import FrontMatterExtractor


def test_extract_preamble_pt():
    extractor = FrontMatterExtractor()
    text = "PREÂMBULO\nNós, representantes...\n\nTÍTULO I\nArt. 1º"
    result = extractor.extract_preamble_pt(text)
    assert result is not None
    assert "Nós, representantes" in result


def test_extract_preamble_nhe():
    extractor = FrontMatterExtractor()
    text = "YUPIRUNGÁ RẼDEWÁ\nYandé, yaãmuyakũtasá...\n\nSESEWÁRA I"
    result = extractor.extract_preamble_nhe(text)
    assert result is not None
    assert "Yandé" in result


def test_extract_adct_pt():
    extractor = FrontMatterExtractor()
    text = "ATO DAS DISPOSIÇÕES CONSTITUCIONAIS TRANSITÓRIAS\nArt. 1º...\n\nBrasília, 5 de outubro de 1988"
    result = extractor.extract_adct_pt(text)
    assert result is not None
    assert "Art. 1º" in result


def test_extract_translator_credits():
    extractor = FrontMatterExtractor()
    text = "SOBRE OS TRADUTORES E CONSULTORES\n\nDadá Baniwa\nEdson Baré"
    result = extractor.extract_translator_credits(text)
    assert result is not None
    assert "Dadá Baniwa" in result