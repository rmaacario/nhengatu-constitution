"""
export.py
---------
Aligned pairs → output files.

Supported formats (from config.output.formats):
    json   → corpus_articles.json + corpus_units.json
    tsv    → corpus_units.tsv
    csv    → corpus_units.csv
    moses  → corpus.pt + corpus.nhe  (synced, one segment per line)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from corpus_pipeline.align import ArticlePair, UnitPair, AlignmentReport
from corpus_pipeline.clean import normalise_segment
from corpus_pipeline.config import Config


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def export_all(
    article_pairs: list[ArticlePair],
    unit_pairs:    list[UnitPair],
    report:        AlignmentReport,
    cfg:           Config,
    out_dir:       Path | None = None,
) -> dict[str, Path]:
    """
    Write every format listed in cfg.output.formats.
    Returns a dict mapping a short format key to the Path written.
    """
    out_dir = out_dir or Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = list(cfg.output.formats)
    written: dict[str, Path] = {}

    if "json" in formats:
        written.update(_write_json(article_pairs, unit_pairs, cfg, out_dir))

    if "tsv" in formats:
        written["tsv"] = _write_delimited(unit_pairs, cfg, out_dir, delimiter="\t",
                                          filename="corpus_units.tsv")

    if "csv" in formats:
        written["csv"] = _write_delimited(unit_pairs, cfg, out_dir, delimiter=",",
                                          filename="corpus_units.csv")

    if "moses" in formats:
        written.update(_write_moses(unit_pairs, cfg, out_dir))

    written["report"] = _write_report(report, out_dir)
    return written


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_json(
    article_pairs: list[ArticlePair],
    unit_pairs:    list[UnitPair],
    cfg:           Config,
    out_dir:       Path,
) -> dict[str, Path]:
    indent = 2 if cfg.output.pretty_json else None

    art_path = out_dir / "corpus_articles.json"
    art_path.write_text(
        json.dumps(
            [
                {
                    "article":   p.article,
                    "pt_chars":  p.pt_chars,
                    "nhe_chars": p.nhe_chars,
                    "pt":        p.pt,
                    "nhe":       p.nhe,
                }
                for p in article_pairs
            ],
            ensure_ascii=False,
            indent=indent,
        ),
        encoding="utf-8",
    )

    units_path = out_dir / "corpus_units.json"
    units_path.write_text(
        json.dumps(
            [
                {
                    "article":    p.article,
                    "unit_type":  p.unit_type,
                    "unit_index": p.unit_index,
                    "alignment":  p.alignment,
                    "pt":         p.pt,
                    "nhe":        p.nhe,
                }
                for p in unit_pairs
            ],
            ensure_ascii=False,
            indent=indent,
        ),
        encoding="utf-8",
    )

    return {"json_articles": art_path, "json_units": units_path}


def _write_delimited(
    unit_pairs: list[UnitPair],
    cfg:        Config,
    out_dir:    Path,
    delimiter:  str,
    filename:   str,
) -> Path:
    path = out_dir / filename
    fieldnames = ["article", "unit_type", "unit_index", "alignment", "pt", "nhe"]
    quoting = csv.QUOTE_ALL if delimiter == "," else csv.QUOTE_MINIMAL

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, delimiter=delimiter, quoting=quoting
        )
        writer.writeheader()
        for p in unit_pairs:
            writer.writerow({
                "article":    p.article,
                "unit_type":  p.unit_type,
                "unit_index": p.unit_index,
                "alignment":  p.alignment,
                "pt":         _flat(p.pt,  cfg),
                "nhe":        _flat(p.nhe, cfg),
            })
    return path


def _write_moses(
    unit_pairs: list[UnitPair],
    cfg:        Config,
    out_dir:    Path,
) -> dict[str, Path]:
    """
    Two synchronised plain-text files.
    Line N of corpus.pt corresponds exactly to line N of corpus.nhe.
    """
    pt_path  = out_dir / "corpus.pt"
    nhe_path = out_dir / "corpus.nhe"

    with (
        pt_path.open("w",  encoding="utf-8") as fh_pt,
        nhe_path.open("w", encoding="utf-8") as fh_nhe,
    ):
        for p in unit_pairs:
            fh_pt.write(_flat(p.pt,  cfg) + "\n")
            fh_nhe.write(_flat(p.nhe, cfg) + "\n")

    return {"moses_pt": pt_path, "moses_nhe": nhe_path}


def _write_report(report: AlignmentReport, out_dir: Path) -> Path:
    path = out_dir / "alignment_report.json"
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _flat(text: str, cfg: Config) -> str:
    """Flatten a (possibly multi-line) segment to a single clean line."""
    return normalise_segment(text, cfg)
