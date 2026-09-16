"""Clause-level chunking (P4.2).

Splits corpus markdown on clause boundaries (`## Regulation R-n: ...` /
`## Policy P-n: ...` / `## Template T-n: ...`), never on a fixed token window — the
point is that every chunk returned by retrieval is a whole, citable clause.

A clause long enough to exceed the token ceiling is split further on `(a)/(b)/(i)`
sub-clause boundaries, keeping the same regulation_number and incrementing
chunk_index — a fragment must never lose its clause id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Soft ceiling before a clause gets split on sub-clause boundaries. ~4 chars/token is a
# standard rough estimate; good enough for a chunking decision, not billing.
MAX_CHUNK_TOKENS = 800
CHARS_PER_TOKEN = 4

# Matches "## Regulation R-5: Per Party Exposure Limit", "## Policy P-1: ...",
# "## Template T-14: ...", and the Annexure sub-heading variants
# "## Regulation R-17 Annexure-I: Classification Categories (...)".
_CLAUSE_HEADING = re.compile(
    r"^##\s+(?:Regulation\s+(?P<reg>R-\d+)(?:\s+(?P<annex>Annexure-[IVX]+))?"
    r"|Policy\s+(?P<pol>P-\d+)"
    r"|Template\s+(?P<tmpl>T-\d+))"
    r":\s*(?P<title>.+)$",
    re.MULTILINE,
)

# Cross-reference phrasings: "Regulation R-7", "R-7", "regulations R-5 and R-9",
# "prescribed in R-12", "Regulation R-17 Annexure-II", "Policy P-28".
_CROSS_REF = re.compile(
    r"\b(?:Regulation|Reg\.?|Policy|Template)?\s*"
    r"((?:R|P|T)-\d+(?:\s+Annexure-[IVX]+)?)",
    re.IGNORECASE,
)

# A source-type inferred from which id prefix a clause carries.
_SOURCE_TYPE_BY_PREFIX = {"R": "sbp_regulation", "P": "internal_policy", "T": "template"}

_SUBCLAUSE_SPLIT = re.compile(r"\n(?=(?:[a-z]\.|[ivx]+\.|\d+\.)\s)")


@dataclass(frozen=True)
class RegulationChunk:
    regulation_number: str  # "R-5", "P-28", "T-14", or "R-17 Annexure-II"
    clause_title: str
    section_path: str
    content: str
    cross_references: list[str]
    source_type: str
    token_count: int
    chunk_index: int


def extract_cross_references(text: str) -> list[str]:
    """Pull every clause id mentioned in `text`, deduplicated, first-seen order."""
    seen: dict[str, None] = {}
    for match in _CROSS_REF.finditer(text):
        ref = re.sub(r"\s+", " ", match.group(1).strip().upper())
        seen.setdefault(ref, None)
    return list(seen.keys())


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_oversized(content: str) -> list[str]:
    """Split an over-length clause body on (a)/(i)/1. sub-clause boundaries.

    Falls back to returning the content whole if no sub-clause markers are found —
    better an oversized chunk than an arbitrarily truncated one.
    """
    parts = [p.strip() for p in _SUBCLAUSE_SPLIT.split(content) if p.strip()]
    return parts if len(parts) > 1 else [content]


def _clause_number(match: re.Match) -> tuple[str, str]:
    """Return (regulation_number, source_type) for a heading match."""
    if match.group("reg"):
        number = match.group("reg")
        if match.group("annex"):
            number = f"{number} {match.group('annex')}"
        return number, "sbp_regulation"
    if match.group("pol"):
        return match.group("pol"), "internal_policy"
    return match.group("tmpl"), "template"


def _section_path(markdown: str, heading_start: int) -> str:
    """The nearest preceding '# Part ...' / '# Annexure ...' H1, if any."""
    preceding = markdown[:heading_start]
    h1_matches = list(re.finditer(r"^#\s+(.+)$", preceding, re.MULTILINE))
    return h1_matches[-1].group(1).strip() if h1_matches else ""


def parse_corpus(path: str | Path) -> list[RegulationChunk]:
    """Parse one corpus markdown file into clause-level chunks.

    Never splits on a token window: splitting only ever happens on a clause boundary
    (## heading) or, for an over-length clause, on a lettered/numbered sub-clause
    boundary that keeps the same regulation_number.
    """
    markdown = Path(path).read_text(encoding="utf-8")
    headings = list(_CLAUSE_HEADING.finditer(markdown))

    chunks: list[RegulationChunk] = []
    for i, match in enumerate(headings):
        number, source_type = _clause_number(match)
        title = match.group("title").strip()
        body_start = match.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        content = markdown[body_start:body_end].strip()
        section_path = _section_path(markdown, match.start())
        full_text_for_refs = f"{title}\n{content}"
        cross_refs = [
            ref for ref in extract_cross_references(full_text_for_refs) if ref != number
        ]

        pieces = (
            _split_oversized(content)
            if _estimate_tokens(content) > MAX_CHUNK_TOKENS
            else [content]
        )
        for chunk_index, piece in enumerate(pieces):
            # Prepend section path + title so the embedded vector carries the
            # clause's identity, not just its prose (WO-P4.2).
            embed_text = f"{section_path}\n{title}\n{piece}".strip()
            chunks.append(
                RegulationChunk(
                    regulation_number=number,
                    clause_title=title,
                    section_path=section_path,
                    content=embed_text,
                    cross_references=cross_refs,
                    source_type=source_type,
                    token_count=_estimate_tokens(embed_text),
                    chunk_index=chunk_index,
                )
            )
    return chunks


def parse_corpus_dir(directory: str | Path) -> list[RegulationChunk]:
    """Parse every .md file in a directory (skips MANIFEST.md and non-clause files)."""
    directory = Path(directory)
    all_chunks: list[RegulationChunk] = []
    for md_file in sorted(directory.glob("*.md")):
        if md_file.name.upper() == "MANIFEST.MD":
            continue
        all_chunks.extend(parse_corpus(md_file))
    return all_chunks


def source_type_for(regulation_number: str) -> str:
    prefix = regulation_number.split("-")[0].strip()
    return _SOURCE_TYPE_BY_PREFIX.get(prefix, "unknown")
