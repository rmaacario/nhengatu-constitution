"""
align.py — align PT and NHE articles/units into parallel pairs.

Key design: NO data is discarded.
  - Every unit gets a pair entry.
  - If one side has more units, the extra units get empty-string partner.
  - Empty-partner pairs are flagged alignment="unmatched" and excluded from
    Moses output but kept in JSON/TSV for downstream review.
  - Versioned duplicates ("39:v2") in NHE are matched against their base PT key.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Literal

from corpus_pipeline.config import Config
from corpus_pipeline.segment import Article, split_units, ADCT_OFFSET

AlignmentQuality = Literal["perfect", "partial", "unmatched"]


@dataclass
class ArticlePair:
    article:   int | str
    pt:        str
    nhe:       str
    pt_chars:  int
    nhe_chars: int
    is_adct:   bool = False


@dataclass
class UnitPair:
    article:    int | str
    unit_type:  str
    unit_index: int
    alignment:  AlignmentQuality
    pt:         str
    nhe:        str
    is_adct:    bool = False


@dataclass
class PreamblePair:
    pt: str
    nhe: str


@dataclass
class ForewordPair:
    pt: str
    nhe: str


@dataclass
class AlignmentReport:
    total_pt_articles:  int
    total_nhe_articles: int
    matched_articles:   int
    versioned_duplicates: list[str]   # NHE keys like "39:v2" that got matched
    only_pt:            list
    only_nhe:           list
    unit_perfect:       int
    unit_partial:       int
    unit_unmatched:     int
    total_unit_pairs:   int
    has_preamble:       bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_adct(key: int | str) -> bool:
    if isinstance(key, int):
        return key >= ADCT_OFFSET
    # string keys like "103-A" are main body
    return False


def _base_key(key: int | str) -> int | str:
    """Map "39:v2" → 39, "103-A" → "103-A", 5 → 5."""
    if isinstance(key, str) and ":v" in key:
        return int(key.split(":v")[0])
    return key


# ---------------------------------------------------------------------------
# Article-level alignment
# ---------------------------------------------------------------------------

def align_articles(
    pt_articles:  dict,
    nhe_articles: dict,
    cfg: Config,
) -> tuple[list[ArticlePair], AlignmentReport]:
    pt_nums  = set(pt_articles.keys())
    nhe_nums = set(nhe_articles.keys())

    # NHE versioned keys ("39:v2") → map to their PT base key
    versioned: list[str] = [k for k in nhe_nums if isinstance(k, str) and ":v" in k]
    versioned_to_pt: dict = {}   # "39:v2" → 39
    for vkey in versioned:
        bkey = _base_key(vkey)
        if bkey in pt_nums:
            versioned_to_pt[vkey] = bkey

    # Regular matches (exact key)
    common   = sorted(pt_nums & nhe_nums, key=lambda k: (isinstance(k, str), k))
    only_pt  = sorted(pt_nums  - nhe_nums - set(versioned_to_pt.values()), key=str)
    only_nhe = sorted(nhe_nums - pt_nums  - set(versioned),                key=str)

    if only_pt:
        print(f"  [align] PT-only articles: {only_pt[:20]}", file=sys.stderr)
    if only_nhe:
        print(f"  [align] NHE-only articles: {only_nhe[:20]}", file=sys.stderr)

    pairs: list[ArticlePair] = []

    # Exact matches
    for n in common:
        adct = _is_adct(n)
        pairs.append(ArticlePair(
            article   = n,
            pt        = pt_articles[n].raw,
            nhe       = nhe_articles[n].raw,
            pt_chars  = len(pt_articles[n].raw),
            nhe_chars = len(nhe_articles[n].raw),
            is_adct   = adct,
        ))

    # Versioned duplicate matches ("39:v2" ↔ PT Art.39)
    for vkey, pt_key in versioned_to_pt.items():
        pairs.append(ArticlePair(
            article   = vkey,
            pt        = pt_articles[pt_key].raw,
            nhe       = nhe_articles[vkey].raw,
            pt_chars  = len(pt_articles[pt_key].raw),
            nhe_chars = len(nhe_articles[vkey].raw),
            is_adct   = False,
        ))

    report = AlignmentReport(
        total_pt_articles    = len(pt_nums),
        total_nhe_articles   = len(nhe_nums),
        matched_articles     = len(pairs),
        versioned_duplicates = list(versioned_to_pt.keys()),
        only_pt              = [str(k) for k in only_pt],
        only_nhe             = [str(k) for k in only_nhe],
        unit_perfect=0, unit_partial=0, unit_unmatched=0,
        total_unit_pairs=0, has_preamble=False,
    )
    return pairs, report


# ---------------------------------------------------------------------------
# Unit-level alignment
# ---------------------------------------------------------------------------

def align_units(
    pt_articles:  dict,
    nhe_articles: dict,
    cfg:          Config,
    preamble:     PreamblePair | None = None,
) -> tuple[list[UnitPair], AlignmentReport]:

    pt_nums  = set(pt_articles.keys())
    nhe_nums = set(nhe_articles.keys())

    versioned: list[str] = [k for k in nhe_nums if isinstance(k, str) and ":v" in k]
    versioned_to_pt: dict = {
        vkey: _base_key(vkey)
        for vkey in versioned
        if _base_key(vkey) in pt_nums
    }

    common   = sorted(pt_nums & nhe_nums, key=lambda k: (isinstance(k, str), str(k)))
    only_pt  = sorted(pt_nums  - nhe_nums  - set(versioned_to_pt.values()), key=str)
    only_nhe = sorted(nhe_nums - pt_nums   - set(versioned),                key=str)

    pairs: list[UnitPair] = []
    perfect = partial = unmatched_count = 0

    # Preamble
    if preamble:
        pairs.append(UnitPair(
            article=0, unit_type="preamble", unit_index=0,
            alignment="perfect", pt=preamble.pt, nhe=preamble.nhe,
        ))

    def _add_pairs(pt_key, nhe_key, article_label):
        nonlocal perfect, partial, unmatched_count
        adct = _is_adct(pt_key) if isinstance(pt_key, int) else False

        pt_units  = split_units(pt_articles[pt_key],   cfg)
        nhe_units = split_units(nhe_articles[nhe_key], cfg)

        pt_count  = len(pt_units)
        nhe_count = len(nhe_units)
        min_count = min(pt_count, nhe_count)
        quality   = "perfect" if pt_count == nhe_count else "partial"

        if quality == "perfect":
            perfect += 1
        else:
            partial += 1

        # Aligned pairs (zip to shorter side)
        for i in range(min_count):
            pairs.append(UnitPair(
                article    = article_label,
                unit_type  = pt_units[i].unit_type,
                unit_index = pt_units[i].unit_index,
                alignment  = quality,
                pt         = pt_units[i].text,
                nhe        = nhe_units[i].text,
                is_adct    = adct,
            ))

        # Unmatched PT tail
        for i in range(min_count, pt_count):
            unmatched_count += 1
            pairs.append(UnitPair(
                article    = article_label,
                unit_type  = pt_units[i].unit_type,
                unit_index = pt_units[i].unit_index,
                alignment  = "unmatched",
                pt         = pt_units[i].text,
                nhe        = "",
                is_adct    = adct,
            ))

        # Unmatched NHE tail
        for i in range(min_count, nhe_count):
            unmatched_count += 1
            pairs.append(UnitPair(
                article    = article_label,
                unit_type  = nhe_units[i].unit_type,
                unit_index = nhe_units[i].unit_index,
                alignment  = "unmatched",
                pt         = "",
                nhe        = nhe_units[i].text,
                is_adct    = adct,
            ))

    # Exact-match articles
    for n in common:
        _add_pairs(n, n, n)

    # Versioned duplicates
    for vkey, pt_key in versioned_to_pt.items():
        _add_pairs(pt_key, vkey, vkey)

    report = AlignmentReport(
        total_pt_articles    = len(pt_nums),
        total_nhe_articles   = len(nhe_nums),
        matched_articles     = len(common) + len(versioned_to_pt),
        versioned_duplicates = list(versioned_to_pt.keys()),
        only_pt              = [str(k) for k in only_pt],
        only_nhe             = [str(k) for k in only_nhe],
        unit_perfect         = perfect,
        unit_partial         = partial,
        unit_unmatched       = unmatched_count,
        total_unit_pairs     = len(pairs),
        has_preamble         = preamble is not None,
    )
    return pairs, report