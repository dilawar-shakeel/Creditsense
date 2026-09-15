"""
Shared pytest fixtures for the CreditSense synthetic-data test suite.

The dataset is generated ONCE per test session (session-scoped fixture) since it's
deterministic (fixed seed) and regenerating it per-test would be wasteful — every test
in both test files reads from the same `dataset` fixture.
"""

import numpy as np
import pandas as pd
import pytest

import creditsense_generator as gen


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """The full synthetic CreditSense portfolio, generated once for the whole test run."""
    return gen.build_dataset(n=gen.N_RECORDS, seed=gen.RANDOM_SEED)


@pytest.fixture(scope="session")
def n_rows(dataset: pd.DataFrame) -> int:
    return len(dataset)


# --- Shared tolerance constants, so every test file uses the same statistical bar -----
# With n=50,000, the standard error of a proportion estimate is at most
# sqrt(0.5*0.5/50000) ~= 0.0022, so 5 percentage points is a very generous (>20 sigma)
# tolerance band that will not produce flaky failures from ordinary sampling noise, while
# still catching a genuinely wrong probability/weight in the generator.
CATEGORICAL_PROPORTION_TOLERANCE_PCT = 5.0

# Relative tolerance used when comparing an empirical moment (mean/median) against the
# theoretical value implied by a distribution's declared parameters. Generous enough to
# absorb the mild skew that ordinary (non-truncated) clipping introduces at the tails,
# without being so loose that a wrong parameter would slip through undetected.
MOMENT_RELATIVE_TOLERANCE = 0.20
