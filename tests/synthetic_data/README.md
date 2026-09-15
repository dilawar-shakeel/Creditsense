# CreditSense Synthetic Data — Automated Test Suite

## Files
- `creditsense_generator.py` — the data generator under test (v2, rigid-spec compliant).
- `conftest.py` — session-scoped fixture that builds the 50,000-row dataset once.
- `stat_helpers.py` — exact theoretical-moment helpers (clipped/truncated distributions)
  used as independent ground truth in the spec-adherence tests.
- `test_distribution_spec_adherence.py` — 47 tests: does every feature match the
  distribution family/parameters/bounds in Section 3 of the architectural guide?
- `test_economic_realism.py` — 27 tests: does the data make real-world economic and
  regulatory sense (correlation direction, SBP caps, label signal quality, leakage
  guardrails)?

## Run
```bash
pip install pytest numpy pandas scipy
pytest -v
```

74/74 tests pass against the current generator (verified before delivery).
