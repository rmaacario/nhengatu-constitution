"""
segment.py
----------
Clean text → structured segments.

Level 0 — extract_section_map()  : {art_num → (titulo, capitulo, secao)} metadata
Level 1 — locate_body_*()        : (start, end) offsets of the main constitution body
Level 2 — extract_articles()     : {article_key → Article}
           Keys are:  int for plain articles (1, 2, …)
                      str for sub-articles ("103-A", "39:v2")
Level 3 — split_units()          : [Unit]  caput / incisos / paragraphs per article

Key design decisions (from structural audit)
--------------------------------------------
* NHE uses BOTH hyphen (-) and em-dash (–) for incisos, sometimes mixed
  within the same article.  split_units / sentence_split must handle both.
* Art.39 appears TWICE in the NHE main body (amended article — both translations
  printed).  The first occurrence gets key 39, the second gets key "39:v2".
  The aligner maps "39:v2" → PT's Art.39 (same article, both kept).
* Art.167-C/D/E/F/G have malformed headers fixed in clean_nheengatu().
* ADCT NHE boundary: the heading appears TWICE (line 806 = TOC entry,
  line 7231 = actual section).  adct_start_occurrence=2 in config selects
  the actual section.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from corpus_pipeline.config import Config

# Articles with number ≥ ADCT_OFFSET are ADCT articles.
ADCT_OFFSET = 10_000


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Body:
    text:  str
    start: int
    end:   int

    @property
    def content(self) -> str:
        return self.text[self.start:self.end]


@dataclass
class Article:
    number:  int | str   # int for plain, str for sub-articles / versioned duplicates
    raw:     str
    lang:    str
    titulo:  str = ""    # section metadata
    capitulo:str = ""
    secao:   str = ""


@dataclass
class Unit:
    article:    int | str
    unit_type:  str   # "caput" | "paragrafo_unico" | "paragrafo"
    unit_index: int   # 0=caput, 1=first paragraph/paragrafo, …
    text:       str
    lang:       str


# ---------------------------------------------------------------------------
# Level 0 — section map
# ---------------------------------------------------------------------------

def extract_section_map(body: Body, lang: str) -> dict[int | str, tuple[str, str, str]]:
    """
    Returns {article_key: (titulo, capitulo, secao)} for every article in the body.
    Used to enrich Article metadata but not required for alignment.
    """
    if lang == "pt":
        titulo_re   = re.compile(r'^\s{20,}TÍ?TULO\s+([IVXLCDM]+)\b.*', re.MULTILINE)
        cap_re      = re.compile(r'^\s{15,}CAP[IÍ]TULO\s+([IVXLCDM]+)\b.*', re.MULTILINE)
        sec_re      = re.compile(r'^\s{15,}SE[CÇ][AÃ][OÕ]\s+([IVXLCDM]+)\b.*', re.MULTILINE)
    else:
        titulo_re   = re.compile(r'^\s{20,}SESEW[AÁ]RA\s+([IVXLCDM]+)\b.*', re.MULTILINE)
        cap_re      = re.compile(r'^\s{15,}[ŨU]BEUS[AÁ]RA?\s+([IVXLCDM]+)\b.*', re.MULTILINE)
        sec_re      = re.compile(r'^\s{15,}SE(SÃU|SÃO|SAU)\s+([IVXLCDM]+)\b.*', re.MULTILINE)

    art_re = re.compile(r'(?:^|\n)(Art\.\.?\s*(\d+(?:-[A-Z])?)\s*[oº°.]?\s)', re.MULTILINE)

    text = body.content
    cur_titulo = cur_cap = cur_sec = ""
    result: dict = {}

    # Walk all matches in order, tracking current section context
    events: list[tuple[int, str, str]] = []   # (pos, event_type, value)
    for m in titulo_re.finditer(text):
        events.append((m.start(), 'titulo',   m.group(0).strip()))
    for m in cap_re.finditer(text):
        events.append((m.start(), 'capitulo', m.group(0).strip()))
    for m in sec_re.finditer(text):
        events.append((m.start(), 'secao',    m.group(0).strip()))
    for m in art_re.finditer(text):
        events.append((m.start(), 'article',  m.group(2)))
    events.sort(key=lambda x: x[0])

    for _, etype, val in events:
        if   etype == 'titulo':   cur_titulo = val; cur_cap = ""; cur_sec = ""
        elif etype == 'capitulo': cur_cap    = val; cur_sec = ""
        elif etype == 'secao':    cur_sec    = val
        elif etype == 'article':
            key = int(val) if val.isdigit() else val
            result[key] = (cur_titulo, cur_cap, cur_sec)

    return result


# ---------------------------------------------------------------------------
# Level 1 — body location
# ---------------------------------------------------------------------------

def locate_body_portuguese(text: str, cfg: Config) -> Body:
    start_m = re.search(cfg.portuguese.body_start_pattern, text)
    if not start_m:
        raise ValueError(
            "Portuguese body start not found.\n"
            f"  Pattern: {cfg.portuguese.body_start_pattern!r}"
        )
    end_m = re.search(cfg.portuguese.body_end_pattern, text)
    if not end_m:
        print("[segment] WARNING: ADCT boundary not found; using EOF.", file=sys.stderr)
    return Body(text=text, start=start_m.start(), end=end_m.start() if end_m else len(text))


def locate_body_nheengatu(text: str, cfg: Config) -> Body:
    heading    = cfg.nheengatu.preamble_heading
    occurrence = int(cfg.nheengatu.body_start_occurrence)
    matches    = list(re.finditer(re.escape(heading), text))
    if len(matches) < occurrence:
        raise ValueError(
            f"NHE preamble heading '{heading}' found only {len(matches)}× "
            f"(need {occurrence})."
        )
    return Body(text=text, start=matches[occurrence - 1].start(), end=len(text))


# ---------------------------------------------------------------------------
# Level 1b — ADCT location
# ---------------------------------------------------------------------------

def locate_adct_portuguese(text: str, cfg: Config) -> Optional[Body]:
    start_m = re.search(cfg.portuguese.adct_start_pattern, text, re.IGNORECASE)
    if not start_m:
        return None
    end_m = re.search(cfg.portuguese.adct_end_pattern, text[start_m.end():])
    end   = (start_m.end() + end_m.start()) if end_m else len(text)
    return Body(text=text, start=start_m.start(), end=end)


def locate_adct_nheengatu(text: str, cfg: Config) -> Optional[Body]:
    """
    The NHE ADCT heading appears twice:
      occurrence 1 → TOC entry (line ~806)
      occurrence 2 → actual section (line ~7231)
    adct_start_occurrence in config selects the correct one.
    """
    occurrence = int(cfg.nheengatu.adct_start_occurrence)
    matches    = list(re.finditer(cfg.nheengatu.adct_start_pattern, text))
    if len(matches) < occurrence:
        return None
    start = matches[occurrence - 1].start()
    return Body(text=text, start=start, end=len(text))


# ---------------------------------------------------------------------------
# Level 2 — article extraction
# ---------------------------------------------------------------------------

_ART_RE = re.compile(r'(?:^|\n)(Art\.\.?\s*(\d+(?:-[A-Z])?)\s*[oº°.]?\s)', re.MULTILINE)


def extract_articles(body: Body, lang: str, cfg: Config) -> dict[int | str, Article]:
    """
    Extract articles from a body region with suffix-aware, duplicate-safe keys.

    Key rules:
      "1"    → int(1)
      "103-A"→ str("103-A")
      Second occurrence of numeric key N → str("N:v2"), third → "N:v3", …
    """
    pattern = re.compile(cfg.articles.pattern, re.MULTILINE)
    text    = body.content
    matches = list(pattern.finditer(text))
    section_map = extract_section_map(body, lang)

    articles: dict = {}
    seen_int_keys: dict[int, int] = {}   # int_key → count seen so far

    for i, match in enumerate(matches):
        raw_num  = match.group(2)                     # "1", "103-A", etc.
        base_key = int(raw_num) if raw_num.isdigit() else raw_num

        start = match.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw   = text[start:end].strip()
        raw   = re.sub(r"^\s*\d{1,3}\s*$", "", raw, flags=re.MULTILINE)
        raw   = re.sub(r"\n{3,}", "\n\n", raw).strip()

        # Resolve versioned duplicates for integer keys
        if isinstance(base_key, int):
            count = seen_int_keys.get(base_key, 0) + 1
            seen_int_keys[base_key] = count
            key: int | str = base_key if count == 1 else f"{base_key}:v{count}"
        else:
            key = base_key

        titulo, capitulo, secao = section_map.get(base_key, ("", "", ""))
        articles[key] = Article(
            number   = key,
            raw      = raw,
            lang     = lang,
            titulo   = titulo,
            capitulo = capitulo,
            secao    = secao,
        )

    return articles


def extract_adct_articles(body: Body, lang: str, cfg: Config) -> dict[int, Article]:
    """
    Extract ADCT articles with keys offset by ADCT_OFFSET (so Art.1 → 10001).
    Duplicates within ADCT are handled the same way as main body.
    """
    pattern  = re.compile(cfg.articles.pattern, re.MULTILINE)
    text     = body.content
    matches  = list(pattern.finditer(text))
    articles: dict[int, Article] = {}
    seen: dict[int, int] = {}

    for i, match in enumerate(matches):
        raw_num  = match.group(2)
        if not raw_num.isdigit():
            continue                           # skip sub-articles in ADCT
        real_num = int(raw_num)
        count    = seen.get(real_num, 0) + 1
        seen[real_num] = count
        key      = real_num + ADCT_OFFSET      # e.g. 10001, 10002, …

        start = match.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw   = text[start:end].strip()
        articles[key] = Article(number=key, raw=raw, lang=lang)

    return articles


def extract_preamble(body: Body, lang: str, cfg: Config) -> Optional[str]:
    if lang == "pt":
        heading, end_pat = cfg.portuguese.preamble_heading, cfg.portuguese.preamble_end_pattern
    else:
        heading, end_pat = cfg.nheengatu.preamble_heading,  cfg.nheengatu.preamble_end_pattern

    text    = body.content
    start_m = re.search(re.escape(heading), text)
    if not start_m:
        return None
    after   = text[start_m.end():]
    end_m   = re.search(end_pat, after)
    content = after[:end_m.start()].strip() if end_m else after.strip()
    return content or None


# ---------------------------------------------------------------------------
# Level 3 — structural unit splitting
# ---------------------------------------------------------------------------

def split_units(article: Article, cfg: Config) -> list[Unit]:
    """
    Split an article into: caput, §N paragraphs, parágrafo único.

    re.split() with N capturing groups returns:
        [pre, g1, g2, …, gN, content, g1, g2, …, gN, content, …]
    Non-matching groups are None.
    """
    markers_cfg = cfg.units.split_markers
    n           = len(markers_cfg)
    # Wrap each pattern in a single outer group; convert any inner capturing
    # groups to non-capturing (?:...) so re.split() produces exactly n+1 slots
    # per chunk — one per marker outer group, then the content between markers.
    def _nc(pat: str) -> str:
        """Convert capturing groups (...)  →  non-capturing (?:...) in pat."""
        return re.sub(r'(?<!\\)(?:\\\\)*\(', lambda m: m.group(0)[:-1] + '(?:', pat)
    combined    = "|".join(f"({_nc(m['pattern'])})" for m in markers_cfg)
    splitter    = re.compile(combined, re.MULTILINE | re.UNICODE)

    parts = splitter.split(article.raw)
    units: list[Unit] = []

    # First part is always the caput
    caput_text = (parts[0] or "").strip()
    if caput_text:
        units.append(Unit(
            article    = article.number,
            unit_type  = "caput",
            unit_index = 0,
            text       = caput_text,
            lang       = article.lang,
        ))

    chunk      = n + 1
    para_index = 1
    i          = 1

    while i < len(parts):
        groups      = [parts[j] for j in range(i, min(i + n, len(parts)))]
        content_idx = i + n
        content     = (parts[content_idx] if content_idx < len(parts) else None) or ""
        content     = content.strip()

        label = "paragrafo"
        for j, g in enumerate(groups):
            if g is not None:
                label = markers_cfg[j]["label"]
                break

        if content:
            units.append(Unit(
                article    = article.number,
                unit_type  = label,
                unit_index = para_index,
                text       = content,
                lang       = article.lang,
            ))
            para_index += 1

        i += chunk

    return units