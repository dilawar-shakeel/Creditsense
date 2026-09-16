# tests/fixtures/

## messy_applications.json

60 applicant records sampled from the real synthetic portfolio (fixed seed, see
`build_messy_applications.py`), with realistic messiness injected on top:

- **OCR digit transposition** — two adjacent digits swapped in a PKR figure, like a bad
  scan of a bank statement (`ocr_digit_transposition`)
- **Currency format variation** — the same PKR amount written as `"Rs. 1,234,567"`,
  `"PKR 1234567"`, or a bare number, cycled across records (`currency_format_variation`)
- **Missing collateral** — `collateral_coverage_ratio` set to `null` (`missing_collateral`)
- **Urdu-English code-switched notes** — a free-text `notes` field mixing English and
  Urdu, e.g. *"client ka cash flow seasonal hai, Eid se pehle spike hota hai"*
  (`urdu_english_notes`)

Each record's `noise_applied` list says exactly which of the above it carries, so a test
can target a specific problem.

**Why this exists instead of a bigger generator change:** the 50,000-row portfolio
generator deliberately skipped this messiness (P1.3/P1.4 in the tracker). Reopening it
would mean regenerating and re-validating the whole dataset. This fixture is the cheap
substitute — enough for `FinancialAnalystAgent` (P5.2) to have something messy to
normalize, and enough to answer "what data-quality problem did you simulate?" honestly
in an interview.

Regenerate with:

```
python tests/fixtures/build_messy_applications.py
```
