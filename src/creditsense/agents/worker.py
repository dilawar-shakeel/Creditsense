"""CLI entry point for the agent pipeline (P5.5/P5.6 wiring, replaces the old
print-and-exit stub). Keeps the `python -m creditsense.agents.worker` entry shape
`docker-compose.yml` already references.

Runs one applicant end to end: analyze -> score -> check -> decide.

By default, loads the applicant's own document from the messy fixtures
(tests/fixtures/messy_applications.json) if a record for that applicant_id exists
there -- this exercises the real parser against realistically messy input. If no
fixture record exists, falls back to using the applicant's own stored profile as the
"document" (still real data, just not deliberately messy), so this works for any of
the 50,000 seeded applicants, not only the 60 in the fixture.

Document loading itself (`load_application`) moved to `agents/application_loader.py`
in Phase 7, so `/applications/underwrite` (P7.1) can share the exact same logic
instead of duplicating or importing a private CLI function.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from creditsense.agents.application_loader import DEFAULT_FIXTURE_PATH, load_application
from creditsense.agents.pipeline import run_underwriting
from creditsense.config import get_settings
from creditsense.db.session import session_scope


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--applicant-id", required=True)
    parser.add_argument("--fixture-path", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--requested-amount-pkr", type=float, default=None)
    parser.add_argument("--clean-facility", action="store_true")
    parser.add_argument("--tenor-months", type=int, default=None)
    args = parser.parse_args()

    print(f"{settings.agent_name} running underwriting for {args.applicant_id}...")

    with session_scope() as session:
        application = load_application(
            args.applicant_id,
            session,
            fixture_path=args.fixture_path,
            requested_amount_pkr=args.requested_amount_pkr,
            is_clean_facility=args.clean_facility,
            tenor_months=args.tenor_months,
        )
        if application is None:
            print(f"No document or applicant record found for {args.applicant_id!r}.")
            sys.exit(1)
        decision = run_underwriting(application, session)

    print(json.dumps(decision.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
