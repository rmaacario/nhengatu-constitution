"""
frontmatter.py
--------------
Extract front-matter, back-matter, and the ADCT section from both PDFs.

Sections handled
----------------
1. Bilingual foreword (Rosa Weber speech)
   PT:  "APRESENTAÇÃO" — physically embedded in the NHE PDF (pages 9–10)
        but the canonical PT version comes from the PT PDF.
   NHE: "MUKAMEĒ SÁ"   — immediately precedes the APRESENTAÇÃO in the NHE PDF.
   These are the same speech and form a genuine parallel pair.

2. Preamble
   PT:  "PREÂMBULO" heading → end before "TÍTULO I"
   NHE: "YUPIRUNGÁ RẼDEWÁ" heading (2nd occurrence, skipping TOC) → end before "SESEWÁRA"

3. ADCT (Transitional Provisions)
   Only a subset of ADCT articles were translated into Nheengatu.
   Extraction is handled by segment.py (locate_adct_*) and article extraction
   under the ADCT_OFFSET namespace.  The text-level helpers here are kept for
   any sentence-pipeline usage that needs the raw ADCT text blocks.

4. Translator credits
   NHE-only section ("SOBRE OS TRADUTORES E CONSULTORES").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentSection:
    name:     str
    pt_text:  str
    nhe_text: str


class FrontMatterExtractor:
    """Extract non-article sections from both PDFs."""

    # -----------------------------------------------------------------------
    # PT markers
    # -----------------------------------------------------------------------
    PT_PREAMBLE_HEADING   = "PREÂMBULO"
    PT_PREAMBLE_END       = r"\nTÍTULO\s+I\b"
    PT_APRESENTACAO_START = "APRESENTAÇÃO"
    # The PT PDF's APRESENTAÇÃO is not in a standalone section the same way
    # as in the NHE PDF.  We locate it via the PT PDF's own foreword area.

    # -----------------------------------------------------------------------
    # NHE markers
    # -----------------------------------------------------------------------
    NHE_FOREWORD_HEADING  = "MUKAMEĒ SÁ"
    NHE_FOREWORD_END      = "APRESENTAÇÃO"          # PT section starts here
    NHE_PREAMBLE_HEADING  = "YUPIRUNGÁ RẼDEWÁ"
    NHE_PREAMBLE_END      = r"\nSESEWÁRA\s+I\b"
    NHE_TRANSLATORS       = "SOBRE OS TRADUTORES E CONSULTORES"

    # -----------------------------------------------------------------------
    # Bilingual foreword
    # -----------------------------------------------------------------------

    def extract_foreword_nhe(self, nhe_text: str) -> Optional[str]:
        """
        Extract the NHE foreword ("MUKAMEĒ SÁ").
        Runs from the heading up to (but not including) the embedded
        Portuguese "APRESENTAÇÃO" section.
        """
        start = nhe_text.find(self.NHE_FOREWORD_HEADING)
        if start == -1:
            return None
        end = nhe_text.find(self.NHE_FOREWORD_END, start)
        if end == -1:
            return None
        content = nhe_text[start + len(self.NHE_FOREWORD_HEADING): end].strip()
        return content if content else None

    def extract_foreword_pt(self, pt_text: str) -> Optional[str]:
        """
        Extract the Portuguese foreword ("APRESENTAÇÃO") from the PT PDF.

        In the PT PDF this section appears as a standalone foreword before
        the preamble, signed by Ministra Rosa Weber.  We locate it by the
        heading and end it at the next major heading (PREÂMBULO or similar).
        """
        start_m = re.search(re.escape(self.PT_APRESENTACAO_START), pt_text)
        if not start_m:
            return None
        # End at the preamble heading
        end_m = re.search(
            re.escape(self.PT_PREAMBLE_HEADING),
            pt_text[start_m.end():]
        )
        if not end_m:
            return None
        content = pt_text[
            start_m.end(): start_m.end() + end_m.start()
        ].strip()
        return content if content else None

    # -----------------------------------------------------------------------
    # Preamble
    # -----------------------------------------------------------------------

    def extract_preamble_pt(self, text: str) -> Optional[str]:
        """Extract Portuguese preamble."""
        match = re.search(
            rf'{re.escape(self.PT_PREAMBLE_HEADING)}(.*?)(?={self.PT_PREAMBLE_END})',
            text,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(1).strip() if match else None

    def extract_preamble_nhe(self, text: str) -> Optional[str]:
        """
        Extract Nheengatu preamble.

        The heading "YUPIRUNGÁ RẼDEWÁ" appears twice: once in the TOC
        (with dot-leaders) and once as the actual section heading.  We
        take the last occurrence, which is always the real content.
        """
        matches = list(re.finditer(
            re.escape(self.NHE_PREAMBLE_HEADING),
            text
        ))
        if not matches:
            return None

        # Use the last occurrence (the actual section, not the TOC entry)
        m = matches[-1]
        after = text[m.end():]
        end_m = re.search(self.NHE_PREAMBLE_END, after)
        content = after[:end_m.start()].strip() if end_m else after.strip()
        return content if content else None

    # -----------------------------------------------------------------------
    # ADCT raw text (for sentence pipeline use)
    # -----------------------------------------------------------------------

    def extract_adct_pt(self, text: str) -> Optional[str]:
        """Extract Portuguese ADCT raw text block."""
        start_m = re.search(
            r'ATO\s+DAS\s+DISPOSIÇÕES\s+CONSTITUCIONAIS\s+TRANSITÓRIAS',
            text, re.IGNORECASE
        )
        if not start_m:
            return None
        end_m = re.search(r'Brasília,\s+5\s+de\s+outubro\s+de\s+1988', text)
        end = end_m.start() if end_m else len(text)
        content = text[start_m.start(): end].strip()
        return content if content else None

    def extract_adct_nhe(self, text: str) -> Optional[str]:
        """Extract Nheengatu ADCT raw text block."""
        start_m = re.search(
            r'ŨBEU\s+SÁ\s+TA\s+MŨDUSÁWA\s+TURUSÚ\s+WAÁ\s+KUXIIMA',
            text
        )
        if not start_m:
            return None
        sobre = text.find("SOBRE OS TRADUTORES")
        end = sobre if sobre != -1 else len(text)
        content = text[start_m.start(): end].strip()
        return content if content else None

    # -----------------------------------------------------------------------
    # Translator credits (NHE-only)
    # -----------------------------------------------------------------------

    def extract_translator_credits(self, text: str) -> Optional[str]:
        """Extract translator credits from Nheengatu PDF."""
        match = re.search(
            rf'{re.escape(self.NHE_TRANSLATORS)}(.*?)(?=\Z)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(1).strip() if match else None

    # -----------------------------------------------------------------------
    # Signatures (legacy — kept for backward compat)
    # -----------------------------------------------------------------------

    def extract_signatures_pt(self, text: str) -> Optional[str]:
        """Extract Portuguese signatures section."""
        match = re.search(
            r'(Brasília,\s+5\s+de\s+outubro\s+de\s+1988.*?)$',
            text,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(1).strip() if match else None

    def extract_signatures_nhe(self, text: str) -> Optional[str]:
        """Extract Nheengatu signatures section."""
        match = re.search(
            r'(Brasília,\s+5\s+de\s+outubro\s+de\s+1988.*?)$',
            text,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(1).strip() if match else None