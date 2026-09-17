import json
from pathlib import Path
from unittest.mock import patch

import pytest

from creditsense.rag.chunking import RegulationChunk
from creditsense.rag.eval_review import review

FAKE_CLAUSES = {
    "R-1": [
        RegulationChunk(
            regulation_number="R-1",
            clause_title="Test Clause",
            section_path="Part I",
            content="Some clause text mentioning R-2.",
            cross_references=["R-2"],
            source_type="sbp_regulation",
            token_count=10,
            chunk_index=0,
        )
    ],
}


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def queries_file(tmp_path):
    path = tmp_path / "eval_queries.jsonl"
    _write_rows(
        path,
        [
            {"id": "q1", "category": "single_clause", "query": "Q1?", "relevant_regulation_numbers": ["R-1"], "verified": False, "notes": ""},
            {"id": "q2", "category": "single_clause", "query": "Q2?", "relevant_regulation_numbers": ["R-1"], "verified": False, "notes": ""},
        ],
    )
    return path


def test_confirming_a_row_saves_it_as_verified(queries_file):
    with patch("creditsense.rag.eval_review._clause_lookup", return_value=FAKE_CLAUSES):
        with patch("builtins.input", side_effect=["y", "y"]):
            review(queries_path=queries_file)

    rows = {r["id"]: r for r in _read_rows(queries_file)}
    assert rows["q1"]["verified"] is True
    assert rows["q2"]["verified"] is True


def test_saves_after_every_answer_not_only_at_the_end(queries_file):
    """Regression test: the tool used to hold everything in memory and write once
    at the end, which meant a fix applied to the file mid-session (by hand, or by
    another process) got silently overwritten when the session finally saved. This
    confirms q1 hits disk as verified before q2 is even asked about."""
    calls = {"count": 0}
    seen_before_q2 = {}

    def fake_input(prompt):
        calls["count"] += 1
        if calls["count"] == 2:
            # This is the prompt for q2 — check disk state as it stood the moment
            # q2's question is asked, i.e. immediately after q1 was answered.
            seen_before_q2["rows"] = _read_rows(queries_file)
        return "y"

    with patch("creditsense.rag.eval_review._clause_lookup", return_value=FAKE_CLAUSES):
        with patch("builtins.input", side_effect=fake_input):
            review(queries_path=queries_file)

    # By the time input() was called for q2, q1's answer must already be on disk.
    q1_state = next(r for r in seen_before_q2["rows"] if r["id"] == "q1")
    assert q1_state["verified"] is True


def test_external_edit_to_a_different_row_survives_a_save(queries_file):
    """Regression test for the exact bug hit in practice: while the session is
    mid-review of q1, someone else corrects q2's label directly in the file. That
    correction must not be lost when q1's answer is saved."""

    def fake_input(prompt):
        # Simulate an external fix to q2 landing on disk while q1 is being answered.
        rows = _read_rows(queries_file)
        for row in rows:
            if row["id"] == "q2":
                row["relevant_regulation_numbers"] = ["R-1", "R-2"]
                row["category"] = "multi_hop"
        _write_rows(queries_file, rows)
        return "y"

    with patch("creditsense.rag.eval_review._clause_lookup", return_value=FAKE_CLAUSES):
        with patch("builtins.input", side_effect=fake_input):
            review(queries_path=queries_file)

    rows = {r["id"]: r for r in _read_rows(queries_file)}
    # q2's externally-applied correction must have survived q1's save.
    assert rows["q2"]["relevant_regulation_numbers"] == ["R-1", "R-2"]
    assert rows["q2"]["category"] == "multi_hop"


def test_skip_moves_on_without_marking_verified(queries_file):
    with patch("creditsense.rag.eval_review._clause_lookup", return_value=FAKE_CLAUSES):
        with patch("builtins.input", side_effect=["s", "y"]):
            review(queries_path=queries_file)

    rows = {r["id"]: r for r in _read_rows(queries_file)}
    assert rows["q1"]["verified"] is False
    assert rows["q2"]["verified"] is True


def test_rejecting_a_label_prompts_for_the_correct_ids(queries_file):
    with patch("creditsense.rag.eval_review._clause_lookup", return_value=FAKE_CLAUSES):
        with patch("builtins.input", side_effect=["n", "R-2, R-3", "y"]):
            review(queries_path=queries_file)

    rows = {r["id"]: r for r in _read_rows(queries_file)}
    assert rows["q1"]["relevant_regulation_numbers"] == ["R-2", "R-3"]
    assert rows["q1"]["verified"] is True


def test_quit_stops_without_touching_remaining_rows(queries_file):
    with patch("creditsense.rag.eval_review._clause_lookup", return_value=FAKE_CLAUSES):
        with patch("builtins.input", side_effect=["q"]):
            review(queries_path=queries_file)

    rows = {r["id"]: r for r in _read_rows(queries_file)}
    assert rows["q1"]["verified"] is False
    assert rows["q2"]["verified"] is False


def test_already_all_verified_is_a_no_op(tmp_path, capsys):
    path = tmp_path / "eval_queries.jsonl"
    _write_rows(
        path,
        [{"id": "q1", "category": "single_clause", "query": "Q1?", "relevant_regulation_numbers": ["R-1"], "verified": True, "notes": ""}],
    )
    with patch("creditsense.rag.eval_review._clause_lookup", return_value=FAKE_CLAUSES):
        review(queries_path=path)

    assert "All rows verified" in capsys.readouterr().out
