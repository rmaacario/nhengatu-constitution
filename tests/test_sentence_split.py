"""Tests for sentence_split module."""

from corpus_pipeline.sentence_split import SentenceSplitter, SentenceAligner


def test_split_portuguese_roman_items():
    splitter = SentenceSplitter()
    text = "I – soberania; II – cidadania; III – dignidade"
    result = splitter.split_portuguese(text)
    assert len(result) == 3
    assert "I – soberania" in result[0]


def test_split_portuguese_sentences():
    splitter = SentenceSplitter()
    text = "First sentence. Second sentence! Third sentence?"
    result = splitter.split_portuguese(text)
    assert len(result) == 3


def test_split_nheengatu_roman_items():
    splitter = SentenceSplitter()
    text = "I- Suberania II- Mírasá-itá III- Míra sikuesá"
    result = splitter.split_nheengatu(text)
    assert len(result) == 3


def test_aligner_perfect_match():
    aligner = SentenceAligner()
    pt = ["A", "B", "C"]
    nhe = ["X", "Y", "Z"]
    result = aligner.align(pt, nhe, 1, "caput")
    assert len(result) == 3
    assert result[0].confidence == 1.0


def test_aligner_partial_match():
    aligner = SentenceAligner()
    pt = ["A", "B", "C"]
    nhe = ["X", "Y"]
    result = aligner.align(pt, nhe, 1, "caput")
    assert len(result) == 2
    assert result[0].confidence == 0.8