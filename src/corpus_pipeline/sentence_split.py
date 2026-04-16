"""
sentence_split.py — split structural units into sentence-level pairs.

Splitting strategy (from structural audit)
------------------------------------------
Articles have two forms:

  LIST form:   caput intro + Roman-numeral items (I, II, … LXXVIII)
               → split into [intro, item_I, item_II, …]
               PT separator: em-dash (–)  ALWAYS
               NHE separator: em-dash (–) OR hyphen (-), sometimes MIXED within same article
               Alíneas (a), b), c)) are NOT split — they stay part of their inciso.

  PROSE form:  one or more plain sentences, no Roman-numeral items
               → split by punctuation

The SentenceAligner uses Gale-Church DP with expected PT/NHE ratio ≈ 1.25.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class SentencePair:
    pt:               str
    nhe:              str
    confidence:       float
    source_article:   int | str
    source_unit_type: str
    source_is_adct:   bool = False


# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

# Strip "Art. 1o " / "Art. 1º " from the very start of a unit (caput includes it)
_ART_HEADER = re.compile(r'^Art\.\.?\s*\d+(?:-[A-Z])?\s*[oº°.]?\s+', re.MULTILINE)

# Inciso line: leading whitespace + Roman numeral + separator (em-dash OR hyphen) + content
# Anchored on ^ to avoid matching mid-sentence Roman numerals in prose
_INCISO = re.compile(r'^\s+([IVXLCDM]+)\s*[–\-]\s+(.+)', re.IGNORECASE | re.MULTILINE)

# Protect these abbreviations from being treated as sentence boundaries
_PT_ABBREV  = re.compile(r'\b(?:Art|arts|art|nº|no|No|Nº|Sr|Sra|Dr|Dra|Prof|vol|cap|pp?|cf|vs|etc)\.', re.IGNORECASE)
_NHE_ABBREV = re.compile(r'\b(?:Art|arts|art|nº|no)\.', re.IGNORECASE)
_PROTECT    = "\x00"


def _protect(text: str, pat: re.Pattern) -> str:
    return pat.sub(lambda m: m.group(0).replace(".", _PROTECT), text)

def _restore(text: str) -> str:
    return text.replace(_PROTECT, ".")


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class SentenceSplitter:

    def split_portuguese(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        result = self._split_list(text)
        return result if result else self._split_prose_pt(text)

    def split_nheengatu(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        result = self._split_list(text)
        return result if result else self._split_prose_nhe(text)

    # ------------------------------------------------------------------
    # List splitting (incisos) — shared by both languages
    # ------------------------------------------------------------------

    def _split_list(self, text: str) -> list[str]:
        """
        If ≥ 2 inciso lines found, return [intro] + [each inciso].
        Returns [] if fewer than 2 incisos (caller uses prose split).
        """
        incisos = _INCISO.findall(text)       # [(numeral, content), …]
        if len(incisos) < 2:
            return []

        first_m = _INCISO.search(text)
        # Strip article header from intro, trim trailing colon
        intro   = _ART_HEADER.sub("", text[:first_m.start()], count=1).strip().rstrip(":").strip()

        items: list[str] = []
        if intro and len(intro) >= 15:
            items.append(intro)

        for numeral, content in incisos:
            content = content.strip().rstrip(";").strip()
            if content:
                items.append(f"{numeral} – {content}")

        return items

    # ------------------------------------------------------------------
    # Prose splitting
    # ------------------------------------------------------------------

    def _split_prose_pt(self, text: str) -> list[str]:
        text = _ART_HEADER.sub("", text, count=1).strip()
        text = text.replace("\n", " ")
        text = _protect(text, _PT_ABBREV)
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ\d("])', text)
        return [_restore(p).strip() for p in parts if p.strip()]

    def _split_prose_nhe(self, text: str) -> list[str]:
        text = _ART_HEADER.sub("", text, count=1).strip()
        text = text.replace("\n", " ")
        text = _protect(text, _NHE_ABBREV)
        # Try semicolons first (common boundary in Nheengatu)
        parts = re.split(r';\s+(?=[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇa-záéíóú\d("])', text)
        if len(parts) > 1:
            return [_restore(p).strip() for p in parts if p.strip()]
        parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇa-z\d("])', text)
        return [_restore(p).strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Aligner — Gale-Church DP
# ---------------------------------------------------------------------------

class SentenceAligner:
    """
    Align PT and NHE sentence lists using Gale-Church length-ratio DP.
    Expected PT/NHE character ratio: 1.25 (measured from corpus).
    Beads: 1-1, 1-2, 2-1, 1-0, 0-1.
    """
    RATIO = 1.25
    BEADS = [(1,1),(1,2),(2,1),(1,0),(0,1)]

    def align(self, pt: list[str], nhe: list[str],
              article_num: int | str, unit_type: str,
              is_adct: bool = False) -> list[SentencePair]:
        if not pt or not nhe:
            return []
        if len(pt) == len(nhe):
            return [SentencePair(p, n, 1.0, article_num, unit_type, is_adct)
                    for p, n in zip(pt, nhe)]
        beads = self._dp(pt, nhe)
        pairs = []
        for pt_g, nhe_g in beads:
            conf = 1.0 if max(len(pt_g),len(nhe_g))==1 else (0.8 if max(len(pt_g),len(nhe_g))==2 else 0.6)
            pairs.append(SentencePair(" ".join(pt_g), " ".join(nhe_g), conf, article_num, unit_type, is_adct))
        return pairs

    def _cost(self, pl, nl, i0, i1, j0, j1):
        a, b = sum(pl[i0:i1]), sum(nl[j0:j1])
        if a==0 and b==0: return float("inf")
        if a==0 or b==0:  return 2.0
        return abs(a/b - self.RATIO)

    def _dp(self, pt, nhe):
        m, n, INF = len(pt), len(nhe), float("inf")
        pl, nl = [len(s) for s in pt], [len(s) for s in nhe]
        dp   = [[INF]*(n+1) for _ in range(m+1)]
        back = [[None]*(n+1) for _ in range(m+1)]
        dp[0][0] = 0.0
        for i in range(m+1):
            for j in range(n+1):
                if dp[i][j]==INF: continue
                for di,dj in self.BEADS:
                    ni,nj = i+di, j+dj
                    if ni>m or nj>n: continue
                    c = self._cost(pl,nl,i,ni,j,nj)
                    if dp[i][j]+c < dp[ni][nj]:
                        dp[ni][nj]=dp[i][j]+c
                        back[ni][nj]=(di,dj,i,j)
        path, i, j = [], m, n
        while i>0 or j>0:
            if not back[i][j]: break
            di,dj,pi,pj = back[i][j]
            path.append((pt[pi:i], nhe[pj:j]))
            i,j = pi,pj
        return list(reversed(path))