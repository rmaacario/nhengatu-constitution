"""
cli.py — Click CLI for the corpus pipeline.

Usage:
    # With package installed:
    corpus-pipeline run --pt <pdf> --nhe <pdf>
    corpus-pipeline check --pt <pdf> --nhe <pdf>

    # Without installing:
    PYTHONPATH=src python scripts/run_pipeline.py run ...
"""
from __future__ import annotations
from pathlib import Path
import sys

import click

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False

from corpus_pipeline import config as cfg_module
from corpus_pipeline.pipeline import run as pipeline_run


def _print(msg: str):
    print(msg)


def _make_console():
    if _RICH:
        from rich.console import Console
        return Console()
    return None


@click.group()
@click.version_option("0.1.0", prog_name="corpus-pipeline")
def main():
    """Portuguese–Nheengatu parallel corpus pipeline."""


@main.command("run")
@click.option("--pt",     "pt_pdf",     default=None, type=click.Path(exists=True))
@click.option("--nhe",    "nhe_pdf",    default=None, type=click.Path(exists=True))
@click.option("--out",                  default=None, type=click.Path(file_okay=False))
@click.option("--config", "config_path",default=None, type=click.Path(exists=True))
@click.option("--formats",              default=None,
              help="Comma-separated: json,tsv,csv,moses")
def run_cmd(pt_pdf, nhe_pdf, out, config_path, formats):
    """Run the full extraction → alignment → export pipeline."""
    cfg = cfg_module.load(config_path)

    pt_path  = Path(pt_pdf)  if pt_pdf  else Path(cfg.inputs.portuguese)
    nhe_path = Path(nhe_pdf) if nhe_pdf else Path(cfg.inputs.nheengatu)
    out_dir  = Path(out)     if out     else Path(cfg.output.dir)

    if formats:
        cfg._data["output"]["formats"] = [f.strip() for f in formats.split(",")]

    for label, path in [("Portuguese PDF", pt_path), ("Nheengatu PDF", nhe_path)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            raise SystemExit(1)

    print(f"  Portuguese PDF : {pt_path}")
    print(f"  Nheengatu PDF  : {nhe_path}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Formats        : {list(cfg.output.formats)}")
    print()

    result = pipeline_run(
        pt_pdf=pt_path, nhe_pdf=nhe_path, cfg=cfg, out_dir=out_dir, log=print
    )

    print()
    print("=== Summary ===")
    print(f"  Article pairs : {len(result.article_pairs)}")
    print(f"  Unit pairs    : {len(result.unit_pairs)}")
    print(f"    perfect     : {result.report.unit_perfect}")
    print(f"    partial     : {result.report.unit_partial}")
    print(f"    unmatched   : {result.report.unit_unmatched}")  # FIX: was unit_caput_only (field doesn't exist)
    print(f"  Elapsed (s)   : {result.elapsed_sec:.1f}")
    print()
    print("Files written:")
    for fmt, path in result.written_files.items():
        print(f"  {fmt:20s} → {path}")


@main.command("check")
@click.option("--pt",     "pt_pdf",     required=True, type=click.Path(exists=True))
@click.option("--nhe",    "nhe_pdf",    required=True, type=click.Path(exists=True))
@click.option("--config", "config_path",default=None,  type=click.Path(exists=True))
def check_cmd(pt_pdf, nhe_pdf, config_path):
    """Validate both PDFs can be parsed without writing any output."""
    from corpus_pipeline import extract, clean, segment

    cfg = cfg_module.load(config_path)

    for label, path, clean_fn, locate_fn, lang in [
        ("Portuguese", pt_pdf,  clean.clean_portuguese, segment.locate_body_portuguese, "pt"),
        ("Nheengatu",  nhe_pdf, clean.clean_nheengatu,  segment.locate_body_nheengatu,  "nhe"),
    ]:
        print(f"\nChecking {label}: {path}")
        try:
            raw     = extract.extract(path)
            print(f"  Extracted {len(raw):,} chars  ✓")
            cleaned = clean_fn(raw, cfg)
            body    = locate_fn(cleaned, cfg)
            arts    = segment.extract_articles(body, lang=lang, cfg=cfg)
            print(f"  Body: chars {body.start:,}–{body.end:,}  ✓")
            print(f"  Articles found: {len(arts)}  ✓")
            print(f"  Article range : {min(arts)} – {max(arts)}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1)


@main.command("sentences")
@click.option("--pt", "pt_pdf", required=True, type=click.Path(exists=True))
@click.option("--nhe", "nhe_pdf", required=True, type=click.Path(exists=True))
@click.option("--out", default="./sentence_output", type=click.Path(file_okay=False))
@click.option("--min-confidence", default=0.5, type=float, help="Minimum confidence score (0-1)")
@click.option("--no-frontmatter", is_flag=True, help="Skip front/back matter")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
def sentences_cmd(pt_pdf, nhe_pdf, out, min_confidence, no_frontmatter, config_path):
    """Run sentence-level alignment (splits incisos into individual sentence pairs)."""
    from corpus_pipeline.sentence_pipeline import SentencePipeline

    cfg = cfg_module.load(config_path)

    pt_path  = Path(pt_pdf)
    nhe_path = Path(nhe_pdf)
    out_dir  = Path(out)

    print(f"  Portuguese PDF : {pt_path}")
    print(f"  Nheengatu PDF  : {nhe_path}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Min confidence : {min_confidence}")
    print(f"  Frontmatter    : {'disabled' if no_frontmatter else 'enabled'}")
    print()

    pipeline = SentencePipeline(cfg)
    result = pipeline.run(
        pt_pdf=pt_path,
        nhe_pdf=nhe_path,
        out_dir=out_dir,
        include_frontmatter=not no_frontmatter,
        min_confidence=min_confidence,
    )

    print()
    print("=== Summary ===")
    print(f"  Total sentence pairs : {result['total_pairs']}")
    print(f"    from articles      : {result['total_pairs'] - result['frontmatter_sections']}")
    print(f"    from front/back    : {result['frontmatter_sections']}")
    print(f"  Article units aligned: {result['article_pairs']}")
    print()
    print("Files written:")
    for name, path in result['outputs'].items():
        print(f"  {name:15} → {path}")
# Add experiment command
from corpus_pipeline.experiments import add_experiment_command
add_experiment_command(main)
