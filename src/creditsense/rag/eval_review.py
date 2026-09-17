"""Human verification pass for the eval set (P4.7).

The queries and proposed relevant clauses in eval_queries.jsonl were LLM-drafted
against this corpus. That makes the labels a starting point, not ground truth — an
LLM drafting both the corpus's prose and the questions about it tends to produce
questions that just echo the corpus's own phrasing, which flatters retrieval without
meaning much. eval.py refuses to report headline numbers until every row here is
marked verified: true.

Run: python -m creditsense.rag.eval_review
Walks through each row one at a time. For each: shows the query, the proposed
relevant regulation number(s), and that clause's actual text (so you can judge the
label against the real clause, not just the proposed id). Type:
  y        - confirm the label as correct
  n        - reject; you'll be prompted for the correct regulation number(s)
  s        - skip for now, revisit later
  q        - quit, saving progress so far
"""

from __future__ import annotations

import json
from pathlib import Path

from creditsense.rag.chunking import RegulationChunk, parse_corpus_dir

DEFAULT_QUERIES_PATH = Path("src/creditsense/data/raw_corpus/eval_queries.jsonl")


def _load_queries(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _save_queries(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _clause_lookup(corpus_dir: Path) -> dict[str, list[RegulationChunk]]:
    """regulation_number -> list of chunks. A plain dict keyed by regulation_number
    would silently keep only the LAST chunk for any regulation number that has more
    than one (e.g. "R-17 Annexure-II" is three separate chunks — Eligible Collateral,
    the FSV Schedule, and Valuation Process — all sharing that same number), hiding
    the others from review entirely. Kept as full RegulationChunk objects, not just
    text, so the reviewer can also see what each clause cross-references.
    """
    chunks = parse_corpus_dir(corpus_dir)
    lookup: dict[str, list[RegulationChunk]] = {}
    for c in chunks:
        lookup.setdefault(c.regulation_number, []).append(c)
    return lookup


def review(
    queries_path: Path = DEFAULT_QUERIES_PATH,
    corpus_dir: Path = Path("src/creditsense/data/raw_corpus"),
) -> None:
    clauses = _clause_lookup(corpus_dir)
    skipped_this_session: set[str] = set()

    total = len(_load_queries(queries_path))
    print(f"{sum(1 for r in _load_queries(queries_path) if not r.get('verified'))} "
          f"of {total} rows need review.\n")

    while True:
        # Reload from disk on every iteration, not once at the start — a fix applied
        # to a row (by hand, or by someone helping review) while this session is
        # still open must be picked up before that row is shown, not silently
        # overwritten when this session eventually saves. This is why an earlier
        # session clobbered a corrected label twice before this loop was rewritten.
        rows = _load_queries(queries_path)
        candidates = [
            r for r in rows
            if not r.get("verified") and r["id"] not in skipped_this_session
        ]
        if not candidates:
            remaining = sum(1 for r in rows if not r.get("verified"))
            if remaining:
                print(f"No more rows to review this session — {remaining} skipped, revisit next run.")
            else:
                print("All rows verified.")
            return

        row = candidates[0]
        print("=" * 88)
        print(f"[{row['id']}] ({row['category']}) {row['query']}")
        print(f"Proposed relevant: {row['relevant_regulation_numbers'] or '(none — out of scope)'}")
        seen_cross_refs: set[str] = set()
        for rn in row["relevant_regulation_numbers"]:
            shown_chunks = clauses.get(rn)
            # Full clause text, not truncated — verifying a label against a clause
            # you can't fully read defeats the point of the review pass (a clause
            # cut off mid-sub-item, e.g. before part (iv), can hide the part that
            # would have changed your answer). All chunks sharing this regulation
            # number are shown, not just one — see _clause_lookup's docstring.
            if shown_chunks is None:
                print(f"\n  --- {rn} ---\n  << NOT FOUND IN CORPUS >>")
                continue
            for i, chunk in enumerate(shown_chunks):
                suffix = f" (chunk {i + 1}/{len(shown_chunks)})" if len(shown_chunks) > 1 else ""
                indented = "\n  ".join(chunk.content.splitlines())
                print(f"\n  --- {rn}{suffix} ---\n  {indented}")
                seen_cross_refs.update(chunk.cross_references)

        # This clause mentions other clauses not already in the proposed answer —
        # shown so you can judge whether the question actually needs them too
        # (a real multi-hop case) or whether, like q11 -> R-4, they're just
        # mentioned for a different sub-question than the one being asked.
        unshown_refs = seen_cross_refs - set(row["relevant_regulation_numbers"])
        if unshown_refs:
            print(f"\n  (clause(s) above also mention, not shown: {sorted(unshown_refs)})")
        print()

        answer = input("Correct? [y/n/s/q]: ").strip().lower()
        if answer == "q":
            print("Quitting — every answer so far was already saved as you went.")
            return
        if answer == "s":
            skipped_this_session.add(row["id"])
            continue
        if answer == "y":
            row["verified"] = True
        elif answer == "n":
            corrected = input(
                "Correct regulation number(s), comma-separated (blank = out of scope): "
            ).strip()
            row["relevant_regulation_numbers"] = (
                [rn.strip() for rn in corrected.split(",") if rn.strip()]
                if corrected
                else []
            )
            row["verified"] = True
        else:
            print("Unrecognized input — treating as skip for now.\n")
            skipped_this_session.add(row["id"])
            continue

        # Save immediately, not just at the end of the whole session — and reload
        # fresh right before writing, folding in only this one row's change, so an
        # edit made to a DIFFERENT row while this answer was being typed survives
        # instead of being overwritten by the stale copy of it we loaded above.
        fresh_rows = _load_queries(queries_path)
        fresh_by_id = {r["id"]: r for r in fresh_rows}
        fresh_by_id[row["id"]] = row
        _save_queries(queries_path, list(fresh_by_id.values()))
        remaining = sum(1 for r in fresh_by_id.values() if not r.get("verified"))
        print(f"Saved. {remaining} row(s) still unverified.\n")


if __name__ == "__main__":
    review()
