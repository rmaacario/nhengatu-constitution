"""
sentence_pipeline.py
--------------------
HYBRID APPROACH:
1. Extract Articles (+ ADCT)
2. Per article: split_units() -> separates §-paragraphs from the caput block
3. caput unit   -> _split_list() for incisos, else prose split
4. §-units      -> prose split directly (they never contain Roman-numeral lists)
5. Frontmatter  -> uses segment.extract_preamble() (already battle-tested)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from corpus_pipeline.config import Config
from corpus_pipeline.extract import extract
from corpus_pipeline.clean import clean_portuguese, clean_nheengatu
from corpus_pipeline.segment import (
    locate_body_portuguese, locate_body_nheengatu,
    locate_adct_portuguese, locate_adct_nheengatu,
    extract_articles, extract_adct_articles, extract_section_map,
    extract_preamble, split_units,
)
from corpus_pipeline.sentence_split import SentenceSplitter, SentenceAligner, SentencePair
from corpus_pipeline.frontmatter import FrontMatterExtractor


class SentencePipeline:
    """Hybrid pipeline: articles -> structural units (§) -> incisos -> sentences"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.splitter = SentenceSplitter()
        self.aligner = SentenceAligner()
        self.fme = FrontMatterExtractor()

    def run(
        self,
        pt_pdf: Path,
        nhe_pdf: Path,
        out_dir: Path,
        include_frontmatter: bool = True,
        min_confidence: float = 0.5,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Extracting text from PDFs...")
        pt_raw = extract(pt_pdf)
        nhe_raw = extract(nhe_pdf)

        print("Cleaning text...")
        pt_clean = clean_portuguese(pt_raw, self.cfg)
        nhe_clean = clean_nheengatu(nhe_raw, self.cfg)

        print("Extracting articles and hierarchy...")
        pt_body = locate_body_portuguese(pt_clean, self.cfg)
        nhe_body = locate_body_nheengatu(nhe_clean, self.cfg)

        pt_articles = extract_articles(pt_body, lang="pt", cfg=self.cfg)
        nhe_articles = extract_articles(nhe_body, lang="nhe", cfg=self.cfg)

        # ADCT
        pt_adct_body = locate_adct_portuguese(pt_clean, self.cfg)
        nhe_adct_body = locate_adct_nheengatu(nhe_clean, self.cfg)
        adct_article_count = 0
        if pt_adct_body and nhe_adct_body:
            pt_adct_arts = extract_adct_articles(pt_adct_body, lang="pt", cfg=self.cfg)
            nhe_adct_arts = extract_adct_articles(nhe_adct_body, lang="nhe", cfg=self.cfg)
            pt_articles.update(pt_adct_arts)
            nhe_articles.update(nhe_adct_arts)
            adct_article_count = len(set(pt_adct_arts) & set(nhe_adct_arts))
            print(f"  ADCT: {adct_article_count} aligned article(s)")

        print("HYBRID ALIGNMENT: Articles -> §-Units -> Incisos -> Sentences...")
        all_sentence_pairs: list[SentencePair] = []

        common_articles = set(pt_articles.keys()) & set(nhe_articles.keys())
        print(f"  Common articles: {len(common_articles)}")

        for article_num in sorted(common_articles, key=lambda k: (isinstance(k, str), str(k))):
            pt_article  = pt_articles[article_num]
            nhe_article = nhe_articles[article_num]
            is_adct     = isinstance(article_num, int) and article_num >= 10000

            # LAYER 1: split on § markers -> caput + paragrafo + paragrafo_unico
            pt_units  = split_units(pt_article,  self.cfg)
            nhe_units = split_units(nhe_article, self.cfg)

            # Align by unit_index (0=caput, 1=first §, ...)
            pt_by_idx  = {u.unit_index: u for u in pt_units}
            nhe_by_idx = {u.unit_index: u for u in nhe_units}

            for idx in sorted(set(pt_by_idx) & set(nhe_by_idx)):
                pt_unit  = pt_by_idx[idx]
                nhe_unit = nhe_by_idx[idx]

                pairs = self._align_unit(
                    pt_text        = pt_unit.text,
                    nhe_text       = nhe_unit.text,
                    article_num    = article_num,
                    unit_type      = pt_unit.unit_type,
                    is_adct        = is_adct,
                    min_confidence = min_confidence,
                )
                all_sentence_pairs.extend(pairs)

        print(f"  Generated {len(all_sentence_pairs)} sentence pairs from articles")

        # Front/back matter
        frontmatter_pairs: list[SentencePair] = []
        if include_frontmatter:
            print("Extracting front/back matter...")
            frontmatter_pairs = self._extract_frontmatter_pairs(
                pt_body        = pt_body,
                nhe_body       = nhe_body,
                pt_full        = pt_clean,
                nhe_full       = nhe_clean,
                min_confidence = min_confidence,
            )
            print(f"  Found {len(frontmatter_pairs)} front/back matter sections")

        all_pairs = all_sentence_pairs + frontmatter_pairs

        print("Exporting...")
        outputs = self._export(all_pairs, out_dir)

        return {
            "sentence_pairs":     all_pairs,
            "total_pairs":        len(all_pairs),
            "article_pairs":      len(common_articles),
            "adct_article_pairs": adct_article_count,
            "frontmatter_sections": len(frontmatter_pairs),
            "outputs":            outputs,
        }

    # -----------------------------------------------------------------------
    # LAYER 2: within one structural unit, split incisos then sentences
    # -----------------------------------------------------------------------

    def _align_unit(
        self,
        pt_text:        str,
        nhe_text:       str,
        article_num:    int | str,
        unit_type:      str,
        is_adct:        bool,
        min_confidence: float,
    ) -> list[SentencePair]:
        """
        caput units:
          - _split_list() for Roman-numeral incisos.
            Both sides must have >= 2 items to enter list mode.
          - Otherwise: prose split the whole caput text.

        paragraph units (paragrafo / paragrafo_unico):
          - Always prose split (these never contain Roman-numeral lists).
        """
        pairs: list[SentencePair] = []

        if unit_type == "caput":
            pt_items  = self.splitter._split_list(pt_text)
            nhe_items = self.splitter._split_list(nhe_text)

            if pt_items and nhe_items:
                for i in range(min(len(pt_items), len(nhe_items))):
                    pt_item  = re.sub(r'^[IVXLCDM]+\s*[–—-]\s*', '', pt_items[i])
                    nhe_item = re.sub(r'^[IVXLCDM]+\s*[–—-]\s*', '', nhe_items[i])

                    pt_sents  = self.splitter.split_portuguese(pt_item)
                    nhe_sents = self.splitter.split_nheengatu(nhe_item)

                    for j in range(min(len(pt_sents), len(nhe_sents))):
                        conf = 1.0 if len(pt_sents) == len(nhe_sents) else 0.9
                        if conf >= min_confidence and len(pt_sents[j].split()) > 2:
                            pairs.append(SentencePair(
                                pt               = pt_sents[j],
                                nhe              = nhe_sents[j],
                                confidence       = conf,
                                source_article   = article_num,
                                source_unit_type = f"caput_inciso_{i+1}",
                                source_is_adct   = is_adct,
                            ))
                return pairs
            # fall through to prose split if no incisos

        # Prose split: §-paragraphs always; caput with no incisos
        pt_sents  = self.splitter.split_portuguese(pt_text)
        nhe_sents = self.splitter.split_nheengatu(nhe_text)

        for i in range(min(len(pt_sents), len(nhe_sents))):
            conf = 1.0 if len(pt_sents) == len(nhe_sents) else 0.8
            if conf >= min_confidence and len(pt_sents[i].split()) > 2:
                pairs.append(SentencePair(
                    pt               = pt_sents[i],
                    nhe              = nhe_sents[i],
                    confidence       = conf,
                    source_article   = article_num,
                    source_unit_type = unit_type,
                    source_is_adct   = is_adct,
                ))

        return pairs

    # -----------------------------------------------------------------------
    # Front/back matter
    # -----------------------------------------------------------------------

    def _extract_frontmatter_pairs(
        self,
        pt_body,
        nhe_body,
        pt_full:        str,
        nhe_full:       str,
        min_confidence: float,
    ) -> list[SentencePair]:
        """
        Preamble: use segment.extract_preamble() which operates on the Body
        object — it correctly handles heading offsets and end-patterns and
        avoids pre-body encoding artifacts.

        Foreword: use FrontMatterExtractor on the full cleaned text.
        Both sections are attempted independently.
        """
        pairs: list[SentencePair] = []

        # --- Preamble ---
        pt_preamble  = extract_preamble(pt_body,  lang="pt",  cfg=self.cfg)
        nhe_preamble = extract_preamble(nhe_body, lang="nhe", cfg=self.cfg)

        if pt_preamble and nhe_preamble:
            pt_sents  = self.splitter.split_portuguese(pt_preamble)
            nhe_sents = self.splitter.split_nheengatu(nhe_preamble)
            for i in range(min(len(pt_sents), len(nhe_sents))):
                if len(pt_sents[i].split()) > 3:
                    pairs.append(SentencePair(
                        pt=pt_sents[i], nhe=nhe_sents[i],
                        confidence=0.9, source_article=0,
                        source_unit_type="preamble", source_is_adct=False,
                    ))
        else:
            print(f"  [frontmatter] preamble: "
                  f"PT={'found' if pt_preamble else 'MISSING'}, "
                  f"NHE={'found' if nhe_preamble else 'MISSING'}")

        # --- Foreword ---
        pt_foreword  = self.fme.extract_foreword_pt(pt_full)
        nhe_foreword = self.fme.extract_foreword_nhe(nhe_full)

        if pt_foreword and nhe_foreword:
            pt_sents  = self.splitter.split_portuguese(pt_foreword)
            nhe_sents = self.splitter.split_nheengatu(nhe_foreword)
            for i in range(min(len(pt_sents), len(nhe_sents))):
                if len(pt_sents[i].split()) > 3:
                    pairs.append(SentencePair(
                        pt=pt_sents[i], nhe=nhe_sents[i],
                        confidence=0.9, source_article=0,
                        source_unit_type="foreword", source_is_adct=False,
                    ))
        else:
            print(f"  [frontmatter] foreword: "
                  f"PT={'found' if pt_foreword else 'MISSING'}, "
                  f"NHE={'found' if nhe_foreword else 'MISSING'}")

        return pairs

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def _export(self, pairs: list[SentencePair], out_dir: Path) -> dict:
        outputs = {}

        # JSON
        json_path = out_dir / "sentence_pairs.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"pt": p.pt, "nhe": p.nhe, "confidence": p.confidence,
                  "article": p.source_article, "unit_type": p.source_unit_type,
                  "is_adct": p.source_is_adct}
                 for p in pairs],
                f, ensure_ascii=False, indent=2,
            )
        outputs["json"] = json_path

        # Moses
        pt_path  = out_dir / "corpus_sentences.pt"
        nhe_path = out_dir / "corpus_sentences.nhe"
        with open(pt_path,  "w", encoding="utf-8") as f_pt, \
             open(nhe_path, "w", encoding="utf-8") as f_nhe:
            for p in pairs:
                f_pt.write( p.pt.replace("\n", " ").strip()  + "\n")
                f_nhe.write(p.nhe.replace("\n", " ").strip() + "\n")
        outputs["moses_pt"]  = pt_path
        outputs["moses_nhe"] = nhe_path

        # TSV
        tsv_path = out_dir / "sentence_pairs.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("article\tunit_type\tis_adct\tconfidence\tpt\tnhe\n")
            for p in pairs:
                pt_flat  = p.pt.replace("\n", " ").replace("\t", " ").strip()
                nhe_flat = p.nhe.replace("\n", " ").replace("\t", " ").strip()
                f.write(f"{p.source_article}\t{p.source_unit_type}\t"
                        f"{p.source_is_adct}\t{p.confidence}\t"
                        f"{pt_flat}\t{nhe_flat}\n")
        outputs["tsv"] = tsv_path

        return outputs