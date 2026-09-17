from pathlib import Path

import pytest

from creditsense.rag.chunking import (
    MAX_CHUNK_TOKENS,
    extract_cross_references,
    parse_corpus,
    parse_corpus_dir,
    source_type_for,
)

CORPUS_DIR = Path("src/creditsense/data/raw_corpus")


@pytest.fixture(scope="module")
def all_chunks():
    return parse_corpus_dir(CORPUS_DIR)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Regulation R-7", ["R-7"]),
        ("R-7", ["R-7"]),
        ("regulations R-5 and R-9", ["R-5", "R-9"]),
        ("prescribed in R-12", ["R-12"]),
        ("per Regulation R-4(i)", ["R-4"]),
        ("Annexure-II to Regulation R-17", ["R-17"]),
        ("Policy P-28", ["P-28"]),
        ("Template T-14", ["T-14"]),
        ("no clause ids here", []),
    ],
)
def test_extract_cross_references_across_phrasings(text, expected):
    assert extract_cross_references(text) == expected


def test_parse_corpus_dir_returns_at_least_the_real_regulations(all_chunks):
    # The real SBP text alone (R-1..R-19 plus 4 annexure sub-sections) is the floor;
    # the full corpus (real + simulated) must be well above it.
    assert len(all_chunks) >= 20


def test_every_chunk_has_a_regulation_number(all_chunks):
    assert all(c.regulation_number for c in all_chunks)


def test_every_chunk_has_a_source_type(all_chunks):
    assert {c.source_type for c in all_chunks} <= {
        "sbp_regulation",
        "internal_policy",
        "template",
    }


def test_no_chunk_exceeds_the_token_ceiling(all_chunks):
    for chunk in all_chunks:
        assert chunk.token_count <= MAX_CHUNK_TOKENS, chunk.regulation_number


def test_every_cross_reference_resolves_to_an_existing_clause(all_chunks):
    existing_bases = {c.regulation_number.split(" ")[0] for c in all_chunks}
    for chunk in all_chunks:
        for ref in chunk.cross_references:
            base = ref.split(" ")[0]
            assert base in existing_bases, f"{chunk.regulation_number} cites undefined {ref}"


def test_chunk_content_carries_clause_identity(all_chunks):
    # Section path + title is prepended so the embedded vector carries clause
    # identity, not just prose (WO-P4.2 rule).
    sample = next(c for c in all_chunks if c.regulation_number == "R-5")
    assert sample.clause_title in sample.content


def test_real_sbp_regulations_are_source_type_sbp_regulation():
    chunks = parse_corpus(CORPUS_DIR / "sbp_prudential_sme_regulations.md")
    assert all(c.source_type == "sbp_regulation" for c in chunks)
    numbers = {c.regulation_number.split(" ")[0] for c in chunks}
    assert numbers == {f"R-{i}" for i in range(1, 20)}


def test_manifest_is_not_parsed_as_a_corpus_file(all_chunks):
    manifest_titles = {c.clause_title for c in all_chunks}
    assert "RAG Corpus Manifest" not in manifest_titles


@pytest.mark.parametrize(
    "regulation_number,expected_source_type",
    [("R-5", "sbp_regulation"), ("P-1", "internal_policy"), ("T-1", "template")],
)
def test_source_type_for(regulation_number, expected_source_type):
    assert source_type_for(regulation_number) == expected_source_type
