"""
pipeline.py — top-level orchestrator.

    run(pt_pdf, nhe_pdf, cfg, out_dir) -> PipelineResult

Steps: extract → clean → segment → align → export
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline import extract, clean, segment, align, export
from corpus_pipeline.align import AlignmentReport, PreamblePair
from corpus_pipeline.config import Config


@dataclass
class PipelineResult:
    article_pairs: list
    unit_pairs:    list
    report:        AlignmentReport
    written_files: dict[str, Path]
    elapsed_sec:   float


def run(pt_pdf, nhe_pdf, cfg: Config, out_dir: Path | None = None, log=print) -> PipelineResult:
    t0 = time.perf_counter()

    log("▶ 1/5 Extract …")
    pt_raw  = extract.extract(pt_pdf)
    nhe_raw = extract.extract(nhe_pdf)
    log(f"  PT {len(pt_raw):,} chars / NHE {len(nhe_raw):,} chars")

    log("▶ 2/5 Clean …")
    pt_clean  = clean.clean_portuguese(pt_raw, cfg)
    nhe_clean = clean.clean_nheengatu(nhe_raw, cfg)

    log("▶ 3/5 Segment …")
    pt_body  = segment.locate_body_portuguese(pt_clean, cfg)
    nhe_body = segment.locate_body_nheengatu(nhe_clean, cfg)

    pt_arts  = segment.extract_articles(pt_body,  lang="pt",  cfg=cfg)
    nhe_arts = segment.extract_articles(nhe_body, lang="nhe", cfg=cfg)
    log(f"  PT {len(pt_arts)} articles / NHE {len(nhe_arts)} articles")

    # ADCT
    pt_adct_body  = segment.locate_adct_portuguese(pt_clean, cfg)
    nhe_adct_body = segment.locate_adct_nheengatu(nhe_clean, cfg)
    adct_count = 0
    if pt_adct_body and nhe_adct_body:
        pt_adct  = segment.extract_adct_articles(pt_adct_body,  lang="pt",  cfg=cfg)
        nhe_adct = segment.extract_adct_articles(nhe_adct_body, lang="nhe", cfg=cfg)
        pt_arts.update(pt_adct)
        nhe_arts.update(nhe_adct)
        adct_count = len(set(pt_adct) & set(nhe_adct))
        log(f"  ADCT: {adct_count} matched article(s)")

    pt_pre  = segment.extract_preamble(pt_body,  lang="pt",  cfg=cfg)
    nhe_pre = segment.extract_preamble(nhe_body, lang="nhe", cfg=cfg)
    preamble = PreamblePair(pt=pt_pre, nhe=nhe_pre) if (pt_pre and nhe_pre) else None
    log(f"  Preamble: {'✓' if preamble else 'NOT FOUND'}")

    log("▶ 4/5 Align …")
    article_pairs, _ = align.align_articles(pt_arts, nhe_arts, cfg)
    unit_pairs, report = align.align_units(pt_arts, nhe_arts, cfg, preamble=preamble)
    log(f"  {len(article_pairs)} article pairs / {len(unit_pairs)} unit pairs")
    log(f"  perfect={report.unit_perfect} partial={report.unit_partial} unmatched={report.unit_unmatched}")
    if report.versioned_duplicates:
        log(f"  versioned duplicates matched: {report.versioned_duplicates}")
    if report.only_pt:
        log(f"  only in PT: {report.only_pt[:10]}")
    if report.only_nhe:
        log(f"  only in NHE: {report.only_nhe[:10]}")

    log("▶ 5/5 Export …")
    written = export.export_all(article_pairs, unit_pairs, report, cfg, out_dir)
    for k, p in written.items():
        log(f"  [{k}] → {p}")

    elapsed = time.perf_counter() - t0
    log(f"✓ {elapsed:.1f}s")
    return PipelineResult(article_pairs, unit_pairs, report, written, elapsed)