"""
CreditSense — Synthetic SME Credit-Risk Dataset Generator
==========================================================
Lahore / Punjab SME Portfolio | SBP Prudential Regulations for SME Financing

Generated per the CreditSense Feature Architecture & Synthetic Data Generation Guide,
Sections 1-5, under the rigid REQUIREMENT 1-6 specification.

Run: python -m creditsense.data.generators.portfolio

Moved here from tests/synthetic_data/creditsense_generator.py (Phase 1 fix 1-B) so that
regenerating the dataset does not require importing production code out of the test
tree. tests/synthetic_data/conftest.py imports build_dataset from this module.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

# ======================================================================================
# TOP-OF-FILE CONFIG BLOCK (Requirement 6.2) — every tunable constant lives here, once.
# ======================================================================================

N_RECORDS = 50_000                  # portfolio size to synthesize
RANDOM_SEED = 42                    # single seed feeding the one Generator (Req 6.1)

LATENT_LOADING = 0.62               # Req 1.3: fraction of a feature's quantile position
                                     # explained by its latent factor vs idiosyncratic
                                     # noise. Must be in [0.55, 0.70] per Req 1.3.

TARGET_DEFAULT_RATE = 0.10          # midpoint of the Section-5 8-12% target band
LABEL_FLIP_RATE = 0.04              # Section-5 label-noise flip rate

# --- Req 4.1: cross-feature logical-coherence ceilings on declared_income_ratio -------
# A Non-Filer cannot plausibly declare a majority of bank turnover to FBR; a Late-Filer
# is partially compliant. These are deterministic post-processing caps, not part of the
# copula — tunable business assumptions, stated here explicitly.
DECLARED_INCOME_CEILING_BY_FILER_STATUS = {
    "Non-Filer": 0.50,
    "Late-Filer": 0.75,
    "Filer": 1.00,
}

# --- Req 3.1: SBP PR-SME regulatory constants -----------------------------------------
# Source: SBP Prudential Regulations for SME Financing, reissued by SH&SFD Department,
# dated 06-Nov-2025, effective 01-Jan-2026. These figures OVERRIDE the older turnover
# breakpoints (Micro <=30M / SE 30-400M / ME >400M) printed in the source architectural
# guide's Section 2/#17 — that guide text predates this circular and is superseded here.
SBP_TIER_TURNOVER_BREAKS = {
    "Micro": (0, 30_000_000),
    "SE": (30_000_000, 150_000_000),        # "Above Micro" Small Enterprise
    "ME": (150_000_000, 800_000_000),
    # Turnover > 800,000,000 PKR is OUT OF SME SCOPE entirely (reclassified as
    # corporate/commercial banking) — the turnover distribution below is TRUNCATED at
    # this ceiling (Req 2.2/3.1), not clipped, so no ineligible obligor is generated.
}
SBP_CLEAN_LENDING_CAP = 50_000_000          # Reg. R-9: flat, identical across all tiers
SBP_TIER_SECURED_CEILING = {                # Reg. R-5: per-party exposure ceiling
    "Micro": 100_000_000,                   # Micro & SE share one ceiling under R-5
    "SE": 100_000_000,
    "ME": 500_000_000,
}
SBP_REGULATION_REMINDER = (
    "NOTE: SBP tier/ceiling figures below reflect the SH&SFD circular dated 06-Nov-2025 "
    "(effective 01-Jan-2026). Re-verify against the current circular before production use."
)

# ======================================================================================
# SECTOR METADATA (Group A #4 taxonomy + Group B conditioning)
# ======================================================================================
# Req 2.5 ASSUMPTION LOG: the guide names explicit load-shedding/FX conditioning groups
# for only a subset of sectors ("Textile/Steel", "Construction/Food", "Retail/Trade",
# "IT/Services" for load-shedding; "Auto Parts/Pharma", "Retail/Food" for FX). All 11
# sectors in the #4 taxonomy are assigned below, with sectors not explicitly named in the
# guide mapped to the closest analogous group by the guide's own stated rationale (energy
# intensity for load-shedding; import dependence for FX):
#   - Rice/Agri-processing, Auto Parts/Engineering, Leather/Sports Goods -> "mid" load
#     group (manufacturing-adjacent, alongside Construction/Food) and "low" FX group
#     (domestic raw materials) except Auto Parts, which the guide explicitly puts in the
#     high-FX "Auto Parts/Pharma" group.
#   - Steel/Re-rolling -> "high" load group (energy-intensive, alongside Textile) and
#     "high" FX group (imported scrap/billets).
#   - Other Services -> "low" load group and "low" FX group (services, non-industrial).
#   - Pharma/Surgical -> "low" load group (non-industrial) but "high" FX group (per the
#     guide's explicit "Auto Parts/Pharma" naming).
# Ordinal risk weight (1=lowest, 5=highest) drives sector ordering below so that a rising
# latent Factor-3 quantile deterministically walks toward higher-risk sectors.
SECTORS = [
    # name,                          prob, risk_wt, load_group, fx_group,     industrial
    ("IT/Tech Services",             0.06, 1,       "lowest",   "low",        False),
    ("Pharma/Surgical",              0.05, 1,       "low",      "high",       False),
    ("Other Services",               0.05, 2,       "low",      "low",        False),
    ("Retail/Trade",                 0.22, 2,       "low",      "low",        False),
    ("Food Processing",              0.08, 3,       "mid",      "low",        True),
    ("Rice/Agri-processing",         0.05, 3,       "mid",      "low",        True),
    ("Auto Parts/Engineering",       0.10, 3,       "mid",      "high",       True),
    ("Leather/Sports Goods",         0.06, 4,       "mid",      "low",        True),
    ("Textile/Garments",             0.18, 4,       "high",     "high",       True),
    ("Construction/Bldg Materials",  0.10, 5,       "mid",      "low",        True),
    ("Steel/Re-rolling",             0.05, 5,       "high",     "high",       True),
]
SECTOR_NAMES = [s[0] for s in SECTORS]
SECTOR_PROBS = np.array([s[1] for s in SECTORS])
SECTOR_RISK_WEIGHT = {s[0]: s[2] for s in SECTORS}
SECTOR_LOAD_GROUP = {s[0]: s[3] for s in SECTORS}
SECTOR_FX_GROUP = {s[0]: s[4] for s in SECTORS}
SECTOR_INDUSTRIAL = {s[0]: s[5] for s in SECTORS}
assert np.isclose(SECTOR_PROBS.sum(), 1.0), "Sector probabilities must sum to 1."
SECTOR_CUM_PROBS = np.cumsum(SECTOR_PROBS)

LOAD_SHEDDING_PARAMS = {  # Group B #11, sector-conditioned (Req 2.4)
    "high": (7.0, 1.5), "mid": (5.0, 1.5), "low": (3.0, 1.5), "lowest": (1.0, 1.0),
}
FX_DEPENDENCY_PARAMS = {  # Group B #13, Beta(a, b) scaled to [0, 1], sector-conditioned
    "high": (3.0, 4.0), "low": (1.5, 5.0),
}
CAPTIVE_POWER_PROB = 0.35            # Group B #12, industrial sectors only
CAPTIVE_POWER_LOGNORM = (4.0, 0.8)   # capacity (kW), (mu, sigma) of underlying normal


# ======================================================================================
# GAUSSIAN-COPULA + DISTRIBUTION HELPERS
# ======================================================================================

def sample_latent_factors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Req 1.1/1.2: one multivariate-normal call, corr(F1,F2)=0.45 exactly, F3 independent."""
    mean = np.zeros(3)
    cov = np.array([
        [1.00, 0.45, 0.00],
        [0.45, 1.00, 0.00],
        [0.00, 0.00, 1.00],
    ])
    return rng.multivariate_normal(mean, cov, size=n)


def factor_to_quantile(latent: np.ndarray, rng: np.random.Generator,
                        loading: float = LATENT_LOADING, invert: bool = False) -> np.ndarray:
    """
    Req 1.3: blend a latent factor with a FRESH idiosyncratic noise draw, then map through
    the standard-normal CDF (Phi) to a Uniform(0,1) quantile.
    `invert=True` is used for every feature where the factor's natural direction is
    inverted. Inverted features in this script: bank_statement_volatility,
    late_payment_count_12m, utility_default_count_12m, working_capital_cycle_days,
    multi_banking_behavior (all: higher Liquidity Discipline -> LOWER value).
    """
    sign = -1.0 if invert else 1.0
    idio = rng.standard_normal(latent.shape[0])  # fresh draw every call, never reused
    blended = sign * loading * latent + np.sqrt(1.0 - loading ** 2) * idio
    return stats.norm.cdf(blended)


def independent_quantile(n: int, rng: np.random.Generator) -> np.ndarray:
    """Req 1.4: Uniform(0,1) draw for features with no Section-4 factor assignment."""
    return rng.uniform(0.0, 1.0, size=n)


def categorical_from_quantile(u: np.ndarray, categories: list[str], probs: list[float]) -> np.ndarray:
    """Invert a Uniform(0,1) quantile into a categorical label via cumulative thresholds.
    `categories`/`probs` ordered so low-u -> categories[0]."""
    cum = np.cumsum(probs)
    cum[-1] = 1.0
    idx = np.searchsorted(cum, u, side="right")
    idx = np.clip(idx, 0, len(categories) - 1)
    return np.array(categories)[idx]


def lognormal_ppf(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """LogNormal ppf; (mu, sigma) parameterize the underlying normal (guide convention)."""
    return stats.lognorm.ppf(u, s=sigma, scale=np.exp(mu))


def clipped_ppf(u: np.ndarray, dist, lo: float, hi: float, **params) -> np.ndarray:
    """Req 2.2 'ordinary clip' path — sample unbounded, then clip. Used ONLY for bounds
    the guide frames as a soft/rare-tail trim, never for a hard scope-defining ceiling."""
    return np.clip(dist.ppf(u, **params), lo, hi)


def truncated_ppf(u: np.ndarray, dist, lo: float, hi: float, **params) -> np.ndarray:
    """Req 2.2 'truncated' path — rescale u into the CDF mass strictly inside [lo, hi]
    BEFORE inverting, so no probability mass piles up at the boundary. Used wherever the
    bound is a real, scope-defining ceiling (years_in_business, annual_bank_turnover_pkr,
    ecib_score's fixed bureau-score scale)."""
    lo_cdf = dist.cdf(lo, **params)
    hi_cdf = dist.cdf(hi, **params)
    u_scaled = lo_cdf + u * (hi_cdf - lo_cdf)
    return dist.ppf(u_scaled, **params)


def beta_scaled_ppf(u: np.ndarray, a: float, b: float, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Beta support is naturally bounded [0,1]; scaling introduces no boundary-spike risk,
    so no clip/truncate distinction applies here."""
    x = stats.beta.ppf(u, a=a, b=b)
    return x * (hi - lo) + lo


def zero_inflated_poisson_ppf(u: np.ndarray, pi0: float, lam: float, cap: int) -> np.ndarray:
    """Manual inverse-CDF for a Zero-Inflated Poisson (scipy has no built-in ZIP).
    P(X=0) = pi0 + (1-pi0)*Poisson.pmf(0,lam); P(X=k>0) = (1-pi0)*Poisson.pmf(k,lam).
    Using the SAME latent `u` for both the zero-inflation gate and the count draw keeps
    the copula correlation intact (Req 1.3)."""
    k = np.arange(0, cap + 1)
    pmf = (1 - pi0) * stats.poisson.pmf(k, lam)
    pmf[0] += pi0
    pmf = pmf / pmf.sum()
    cdf = np.cumsum(pmf)
    idx = np.searchsorted(cdf, u, side="right")
    return np.clip(idx, 0, cap).astype(int)


def poisson_floor_ppf(u: np.ndarray, lam: float, floor: int) -> np.ndarray:
    return (stats.poisson.ppf(u, mu=lam) + floor).astype(int)


# ======================================================================================
# FEATURE GENERATION (Groups A-E, Requirement 2)
# ======================================================================================

def generate_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n))
    latent = sample_latent_factors(n, rng)
    f1, f2, f3 = latent[:, 0], latent[:, 1], latent[:, 2]

    # ---------------- Req 2.3: sector_risk_code, Factor-3-driven categorical ----------
    u_sector = factor_to_quantile(f3, rng)
    idx = np.searchsorted(SECTOR_CUM_PROBS, u_sector, side="right")
    idx = np.clip(idx, 0, len(SECTOR_NAMES) - 1)
    df["sector_risk_code"] = np.array(SECTOR_NAMES)[idx]
    df["sector_risk_weight"] = df["sector_risk_code"].map(SECTOR_RISK_WEIGHT).astype(int)

    # ---------------- Req 2.4: energy / FX / captive power, sector-deterministic -----
    load_group = df["sector_risk_code"].map(SECTOR_LOAD_GROUP)
    load_mu = load_group.map(lambda g: LOAD_SHEDDING_PARAMS[g][0]).to_numpy()
    load_sigma = load_group.map(lambda g: LOAD_SHEDDING_PARAMS[g][1]).to_numpy()
    df["load_shedding_impact_score"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        rng.uniform(size=n), stats.norm, 0.0, 10.0, loc=load_mu, scale=load_sigma
    )

    fx_group = df["sector_risk_code"].map(SECTOR_FX_GROUP)
    fx_a = fx_group.map(lambda g: FX_DEPENDENCY_PARAMS[g][0]).to_numpy()
    fx_b = fx_group.map(lambda g: FX_DEPENDENCY_PARAMS[g][1]).to_numpy()
    df["fx_import_dependency_ratio"] = rng.beta(fx_a, fx_b)  # Beta: naturally bounded

    is_industrial = df["sector_risk_code"].map(SECTOR_INDUSTRIAL).to_numpy()
    captive_draw = rng.uniform(size=n) < np.where(is_industrial, CAPTIVE_POWER_PROB, 0.0)
    df["captive_power_dependency"] = captive_draw.astype(int)
    cap_mu, cap_sigma = CAPTIVE_POWER_LOGNORM
    capacity = lognormal_ppf(rng.uniform(size=n), cap_mu, cap_sigma)
    df["captive_power_capacity_kw"] = np.where(captive_draw, capacity, 0.0)

    # ---------------- Factor 3 (continued): KIBOR sensitivity, seasonal CF -----------
    u_kibor = factor_to_quantile(f3, rng)
    df["kibor_sensitivity_pct"] = beta_scaled_ppf(u_kibor, 5, 2, 0, 1)

    u_seasonal = factor_to_quantile(f3, rng)
    df["seasonal_cash_flow_index"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        u_seasonal, stats.lognorm, 1.0, 5.0, s=0.35, scale=np.exp(0.50)
    )

    # ---------------- Factor 1: Formality/Governance ----------------------------------
    u_filer = factor_to_quantile(f1, rng)  # ordered worst->best so high-F1 -> "Filer"
    df["tax_filer_status"] = categorical_from_quantile(
        u_filer, ["Non-Filer", "Late-Filer", "Filer"], [0.50, 0.12, 0.38]
    )
    df["is_non_filer"] = (df["tax_filer_status"] == "Non-Filer").astype(int)

    u_declared = factor_to_quantile(f1, rng)
    df["declared_income_ratio"] = beta_scaled_ppf(u_declared, 2, 5, 0, 1)
    # --- Req 4.1: deterministic logical-coherence cap, applied AFTER latent sampling ---
    filer_ceiling = df["tax_filer_status"].map(DECLARED_INCOME_CEILING_BY_FILER_STATUS)
    df["declared_income_ratio"] = np.minimum(df["declared_income_ratio"], filer_ceiling)

    u_doc = factor_to_quantile(f1, rng)
    df["documentation_tier"] = categorical_from_quantile(
        u_doc, ["Bank-Statement-Only", "Unaudited Financials", "Audited Financials"],
        [0.40, 0.45, 0.15]
    )

    u_legal = factor_to_quantile(f1, rng)
    df["legal_structure"] = categorical_from_quantile(
        u_legal, ["Sole Proprietorship", "Partnership/AOP", "Pvt Ltd (SECP)"],
        [0.55, 0.25, 0.20]
    )

    u_labor = factor_to_quantile(f1, rng)
    df["labor_formalization_ratio"] = beta_scaled_ppf(u_labor, 1.5, 4, 0, 1)

    u_digital = factor_to_quantile(f1, rng)
    df["digital_payment_footprint"] = beta_scaled_ppf(u_digital, 1.5, 5, 0, 1)

    u_ecib = factor_to_quantile(f1, rng)
    # Req 2.2: TRUNCATED — a bureau score has a fixed, real scale ceiling [300, 850].
    df["ecib_score"] = truncated_ppf(u_ecib, stats.norm, 300, 850, loc=650, scale=90)
    nohit_prob = np.clip(0.10 + 0.20 * (1 - u_ecib), 0.0, 1.0)  # thinner file -> less formal
    df["ecib_no_hit_flag"] = (rng.uniform(size=n) < nohit_prob).astype(int)
    df.loc[df["ecib_no_hit_flag"] == 1, "ecib_score"] = np.nan

    # ---------------- Factor 2: Liquidity Discipline ----------------------------------
    u_current = factor_to_quantile(f2, rng)
    df["current_ratio"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        u_current, stats.lognorm, 0.3, 4.0, s=0.45, scale=np.exp(0.15)
    )

    u_volatility = factor_to_quantile(f2, rng, invert=True)
    df["bank_statement_volatility"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        u_volatility, stats.lognorm, 0.15, 2.5, s=0.60, scale=np.exp(-0.50)
    )

    u_late = factor_to_quantile(f2, rng, invert=True)
    df["late_payment_count_12m"] = zero_inflated_poisson_ppf(u_late, 0.55, 1.2, cap=12)

    u_utility = factor_to_quantile(f2, rng, invert=True)
    df["utility_default_count_12m"] = zero_inflated_poisson_ppf(u_utility, 0.65, 0.8, cap=10)

    u_wc = factor_to_quantile(f2, rng, invert=True)
    df["working_capital_cycle_days"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        u_wc, stats.norm, 10, 200, loc=75, scale=30
    )

    u_multibank = factor_to_quantile(f2, rng, invert=True)
    df["multi_banking_behavior"] = poisson_floor_ppf(u_multibank, 2.3, floor=1)

    # ---------------- Req 1.4: independent features (no Section-4 assignment) --------
    df["debt_to_equity_ratio"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        independent_quantile(n, rng), stats.lognorm, 0.0, 8.0, s=0.70, scale=np.exp(0.50)
    )
    df["revenue_growth_yoy_pct"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        independent_quantile(n, rng), stats.norm, -40.0, 80.0, loc=8.0, scale=18.0
    )
    # Req 2.2: TRUNCATED — years-in-business has a real plausible-age ceiling; a naive
    # clip would stack every business older than 40 into a single spike AT 40, which is
    # not representative of any real population (verified against actual generator output).
    df["years_in_business"] = truncated_ppf(
        independent_quantile(n, rng), stats.gamma, 0.5, 40.0, a=2.0, scale=4.5
    )
    df["existing_loan_exposure_pkr"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        independent_quantile(n, rng), stats.lognorm, 0.0, 150_000_000, s=1.1, scale=np.exp(14.5)
    )
    df["collateral_coverage_ratio"] = beta_scaled_ppf(  # Beta: naturally bounded
        independent_quantile(n, rng), 2, 3, 0, 2.5
    )
    df["buyer_concentration_top3_pct"] = beta_scaled_ppf(  # Beta: naturally bounded
        independent_quantile(n, rng), 2, 2, 0, 1
    )
    zero_mask = rng.uniform(size=n) < 0.70
    group_exposure = lognormal_ppf(rng.uniform(size=n), 14.0, 1.3)
    df["group_associate_exposure_pkr"] = np.where(zero_mask, 0.0, group_exposure)

    df["guarantor_net_worth_pkr"] = clipped_ppf(  # Req 2.2: soft trim, ordinary clip
        independent_quantile(n, rng), stats.lognorm, 0.0, 500_000_000, s=1.0, scale=np.exp(15.5)
    )
    df["business_premises_owned"] = (rng.uniform(size=n) < 0.40).astype(int)
    df["trade_body_membership"] = (rng.uniform(size=n) < 0.25).astype(int)

    # ---------------- Req 2.7: auxiliary field, NOT one of the named 30 --------------
    # annual_bank_turnover_pkr is required to derive sbp_enterprise_tier (#17) and to
    # anchor recommend_credit_limit_pkr on cash-flow capacity (Section 5), but is not
    # itself a named taxonomy feature. Req 2.2/3.1: TRUNCATED at the real SME-scope
    # ceiling of 800,000,000 PKR (Req 3.1) rather than clipped, so every generated
    # record is, by construction, a valid SME. Parameters (mu=17.2, sigma=1.5) are
    # chosen so the resulting Micro/SE/ME mix is ~50% / ~36% / ~13%, consistent with
    # Lahore's SME base being overwhelmingly micro/small — re-tune against the bank's
    # actual portfolio composition once available.
    df["annual_bank_turnover_pkr"] = truncated_ppf(
        independent_quantile(n, rng), stats.lognorm, 500_000, 800_000_000, s=1.5, scale=np.exp(17.2)
    )

    def assign_tier(t: float) -> str:
        for tier, (lo, hi) in SBP_TIER_TURNOVER_BREAKS.items():
            if lo <= t < hi:
                return tier
        return "ME"

    df["sbp_enterprise_tier"] = df["annual_bank_turnover_pkr"].apply(assign_tier)
    df["is_startup_flag"] = (df["years_in_business"] <= 5).astype(int)

    return df


# ======================================================================================
# LABEL GENERATION (Requirement 5, Dual-Head Architecture)
# ======================================================================================

def zscore(s: pd.Series) -> np.ndarray:
    x = s.to_numpy(dtype=float)
    mu, sigma = np.nanmean(x), np.nanstd(x)
    sigma = sigma if sigma > 1e-9 else 1.0
    return (x - mu) / sigma


def generate_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # -------------------------- Req 5.1: risk index -------------------------------------
    # Feature list and weight signs match the guide's Section-5 directional list exactly:
    # debt-to-equity(+), bank statement volatility(+), late-payment count(+), utility
    # defaults(+), load-shedding impact(+), non-filer status(+), buyer concentration(+),
    # collateral coverage(-), current ratio(-), years in business(-), declared-income
    # ratio(-). Magnitudes are a relative-importance judgment call, re-tunable once real
    # infection-ratio drivers are validated against production data.
    risk_index = (
          0.90 * zscore(df["debt_to_equity_ratio"])
        + 1.10 * zscore(df["bank_statement_volatility"])
        + 1.30 * zscore(df["late_payment_count_12m"])
        + 0.60 * zscore(df["utility_default_count_12m"])
        + 0.50 * zscore(df["load_shedding_impact_score"])
        + 1.00 * zscore(df["is_non_filer"].astype(float))
        + 0.70 * zscore(df["buyer_concentration_top3_pct"])
        - 1.00 * zscore(df["collateral_coverage_ratio"])
        - 0.80 * zscore(df["current_ratio"])
        - 0.60 * zscore(df["years_in_business"])
        - 0.90 * zscore(df["declared_income_ratio"])
    )
    z_risk = zscore(pd.Series(risk_index))

    LOGIT_SCALE = 0.85  # controls spread of PD across the portfolio

    # -------------------------- Req 5.2: flip-adjusted calibration ----------------------
    # The Req-5.3 flip is symmetric regardless of the pre-noise label, so on an imbalanced
    # ~90/10 book it pushes the realized rate UP by roughly flip*(1-2*pre_noise_rate).
    # Algebra: let f = LABEL_FLIP_RATE, target = TARGET_DEFAULT_RATE (the POST-noise rate
    # we actually want). p_final = p_pre*(1-f) + (1-p_pre)*f = f + p_pre*(1-2f)
    #   => p_pre = (target - f) / (1 - 2f)
    # We calibrate the logistic intercept so the PRE-noise population mean hits this
    # back-solved p_pre, so the POST-noise realized rate (what validate_dataset reports)
    # lands on the 8-12% band the guide actually specifies.
    pre_noise_target = (TARGET_DEFAULT_RATE - LABEL_FLIP_RATE) / (1.0 - 2.0 * LABEL_FLIP_RATE)

    def mean_pd_gap(intercept: float) -> float:
        p = 1.0 / (1.0 + np.exp(-(LOGIT_SCALE * z_risk + intercept)))
        return p.mean() - pre_noise_target

    intercept = brentq(mean_pd_gap, -10.0, 10.0, xtol=1e-10)
    pd_score = 1.0 / (1.0 + np.exp(-(LOGIT_SCALE * z_risk + intercept)))

    # -------------------------- Req 5.4: LABEL-DERIVED DIAGNOSTIC FIELD -----------------
    # `risk_index_score` IS the pre-noise default probability used to draw the label below.
    # *** TARGET LEAKAGE WARNING ***
    # Including `risk_index_score` (or any column derived from it, see recommend_credit_
    # limit_pkr below) as a PREDICTOR of `default_probability_12m` is target leakage: the
    # model will "predict" the label by reading off the value used to generate it, not by
    # learning a recoverable risk pattern. EXCLUDE this column from any feature set used
    # to train a classifier against `default_probability_12m`. It is legitimate ONLY as an
    # input to a downstream Stage-2 credit-limit model, mirroring real underwriting where
    # a risk score feeds a limit decision.
    df["risk_index_score"] = pd_score

    default_draw = (rng.uniform(size=len(df)) < pd_score).astype(int)
    flip_mask = rng.uniform(size=len(df)) < LABEL_FLIP_RATE  # Req 5.3
    df["default_probability_12m"] = np.where(flip_mask, 1 - default_draw, default_draw)

    # -------------------------- Req 5.5: credit limit head -------------------------------
    TURNOVER_FINANCING_RATE = 0.20  # standard working-capital rule-of-thumb fraction
    collateral_multiplier = 0.70 + 0.30 * np.clip(df["collateral_coverage_ratio"], 0.0, 2.0)
    risk_multiplier = 1.15 - np.clip(pd_score, 0.01, 0.60)  # *** also leakage-adjacent:
    # recommend_credit_limit_pkr is a function of pd_score, so it must ALSO be excluded
    # from any feature set predicting default_probability_12m (Req 5.4 applies here too).

    raw_limit = np.maximum(
        TURNOVER_FINANCING_RATE
        * df["annual_bank_turnover_pkr"].to_numpy()
        * collateral_multiplier.to_numpy()
        * risk_multiplier,
        0.0,
    )

    # ---- Req 3.1/5.5: deterministic SBP hard caps (post-processing business rule) ------
    secured_ceiling = df["sbp_enterprise_tier"].map(SBP_TIER_SECURED_CEILING).to_numpy(dtype=float)
    is_secured = (df["collateral_coverage_ratio"].to_numpy() >= 0.10)
    per_obligor_cap = np.where(is_secured, secured_ceiling, SBP_CLEAN_LENDING_CAP)

    has_group_exposure = df["group_associate_exposure_pkr"].to_numpy() > 0.0
    # Reg R-5's per-party ceiling already reads as an aggregate "single bank or all
    # banks/DFIs/MFBs" figure, so the group/associate cap uses the SAME tier ceiling
    # (no separate multiplier) rather than a distinct allowance (Req 5.5).
    aggregate_room = np.where(
        has_group_exposure,
        np.maximum(secured_ceiling - df["group_associate_exposure_pkr"].to_numpy(), 0.0),
        np.inf,
    )

    capped_limit = np.minimum.reduce([raw_limit, per_obligor_cap, aggregate_room])
    df["recommend_credit_limit_pkr"] = np.round(np.maximum(capped_limit, 0.0), -3)
    df["credit_limit_hit_obligor_cap"] = (raw_limit > per_obligor_cap).astype(int)
    df["credit_limit_hit_group_cap"] = (raw_limit > aggregate_room).astype(int)

    # -------------------------- Req 5.6: exact-zero handling -----------------------------
    # A capped limit of exactly 0 (typically a fully-consumed group-exposure cap) is a
    # legitimate full-decline outcome, not a bug — flagged explicitly below rather than
    # forced to a positive floor. Any Stage-2 regression trained on recommend_credit_
    # limit_pkr MUST use an objective supporting an exact-zero mass point (e.g.
    # reg:tweedie, variance power ~1.3-1.5); a strictly-positive objective (e.g.
    # reg:gamma) will error or emit NaN on these rows unless a small positive floor
    # (e.g. 1,000 PKR) is applied to this column first.
    df["credit_limit_is_full_decline"] = (df["recommend_credit_limit_pkr"] <= 0).astype(int)

    return df


# ======================================================================================
# VALIDATION (Requirement 6.3)
# ======================================================================================

def validate_dataset(df: pd.DataFrame) -> None:
    print("=" * 96)
    print(SBP_REGULATION_REMINDER)  # Req 3.2: printed reminder
    print("=" * 96)
    print(f"CREDITSENSE SYNTHETIC DATASET VALIDATION REPORT   (n = {len(df):,} records)")
    print("=" * 96)

    numeric_cols = [
        "current_ratio", "debt_to_equity_ratio", "revenue_growth_yoy_pct", "years_in_business",
        "bank_statement_volatility", "existing_loan_exposure_pkr", "late_payment_count_12m",
        "collateral_coverage_ratio", "kibor_sensitivity_pct", "load_shedding_impact_score",
        "fx_import_dependency_ratio", "seasonal_cash_flow_index", "declared_income_ratio",
        "labor_formalization_ratio", "utility_default_count_12m", "ecib_score",
        "multi_banking_behavior", "digital_payment_footprint", "buyer_concentration_top3_pct",
        "group_associate_exposure_pkr", "guarantor_net_worth_pkr", "working_capital_cycle_days",
        "annual_bank_turnover_pkr", "recommend_credit_limit_pkr",
    ]
    stats_rows = []
    for col in numeric_cols:
        s = df[col]
        stats_rows.append({
            "feature": col, "mean": s.mean(), "median": s.median(),
            "min": s.min(), "max": s.max(), "missing_%": 100 * s.isna().mean(),
        })
    report = pd.DataFrame(stats_rows).set_index("feature")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print("\n--- Continuous / Discrete Feature Summary ---")
        print(report)

    print("\n--- Categorical Distributions ---")
    for col in ["sector_risk_code", "tax_filer_status", "documentation_tier",
                "legal_structure", "sbp_enterprise_tier"]:
        print(f"\n{col}:")
        print((df[col].value_counts(normalize=True) * 100).round(2).astype(str) + "%")

    print("\n--- Binary Flag Rates ---")
    for col in ["is_non_filer", "ecib_no_hit_flag", "captive_power_dependency",
                "business_premises_owned", "trade_body_membership", "is_startup_flag",
                "credit_limit_is_full_decline"]:
        print(f"  {col:35s}: {100 * df[col].mean():5.2f}%")

    print("\n--- Correlation Structure Check (>= 3 required, Req 6.3) ---")
    print(f"  1) corr(current_ratio, bank_statement_volatility) "
          f"[expect negative, same latent Factor 2]: "
          f"{df['current_ratio'].corr(df['bank_statement_volatility']):.3f}")
    print(f"  2) corr(declared_income_ratio, is_non_filer) "
          f"[expect negative, same latent Factor 1]: "
          f"{df['declared_income_ratio'].corr(df['is_non_filer']):.3f}")
    print(f"  3) corr(late_payment_count_12m, bank_statement_volatility) "
          f"[expect positive, same latent Factor 2]: "
          f"{df['late_payment_count_12m'].corr(df['bank_statement_volatility']):.3f}")

    print("\n--- Req 4.1 Sanity Check: declared_income_ratio ceiling by filer status ---")
    for status, ceiling in DECLARED_INCOME_CEILING_BY_FILER_STATUS.items():
        sub = df.loc[df["tax_filer_status"] == status, "declared_income_ratio"]
        print(f"  {status:12s} ceiling={ceiling:.2f}  observed max={sub.max():.3f}  "
              f"(PASS if observed max <= ceiling)")

    print("\n--- Req 2.2 Truncation Sanity Check (no boundary spike) ---")
    for col, hi in [("years_in_business", 40.0), ("annual_bank_turnover_pkr", 800_000_000.0),
                    ("ecib_score", 850.0)]:
        at_ceiling = 100 * (df[col] >= hi * 0.999).mean()
        print(f"  {col:28s}: {at_ceiling:.3f}% of rows within 0.1% of the {hi:,.0f} ceiling "
              f"(should be near 0%, NOT a visible spike)")

    print("\n--- Label Head 1: default_probability_12m ---")
    default_rate = df["default_probability_12m"].mean()
    print(f"  Realized (post-noise) default rate: {default_rate * 100:.2f}%  "
          f"(target band: 8-12%)")
    print(f"  Missingness: {100 * df['default_probability_12m'].isna().mean():.2f}%")
    print("  Leakage warning: risk_index_score and recommend_credit_limit_pkr (and its "
          "derived cap-flag columns) are label-derived and MUST be excluded from any "
          "feature set predicting default_probability_12m.")

    print("\n--- Label Head 2: recommend_credit_limit_pkr ---")
    lim = df["recommend_credit_limit_pkr"]
    print(f"  mean={lim.mean():,.0f}  median={lim.median():,.0f}  "
          f"min={lim.min():,.0f}  max={lim.max():,.0f}  missing={100 * lim.isna().mean():.2f}%")
    print(f"  % hitting the per-obligor SBP cap: {100 * df['credit_limit_hit_obligor_cap'].mean():.2f}%")
    print(f"  % hitting the group-exposure cap:  {100 * df['credit_limit_hit_group_cap'].mean():.2f}%")
    print(f"  % full-decline (exact zero):       {100 * df['credit_limit_is_full_decline'].mean():.2f}%")

    print("\n--- Overall Missingness (all columns) ---")
    miss = (df.isna().mean() * 100).round(2)
    miss = miss[miss > 0]
    print(miss.to_string() if len(miss) else "  No missing values outside ecib_score no-hit flag.")

    print("\n" + "=" * 96)
    print("Validation complete.")
    print("=" * 96)


# ======================================================================================
# MAIN (Requirement 6.1, 6.4)
# ======================================================================================

def build_dataset(n: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    # Req 6.1: ONE Generator instance, threaded explicitly through every call below —
    # no function in this script reads or mutates global numpy random state.
    rng = np.random.default_rng(seed)
    df = generate_features(n, rng)
    df = generate_labels(df, rng)

    # applicant_id: a stable, human-readable row identifier. Assigned last, from row
    # order alone (no RNG draw), so it never perturbs the feature/label generation
    # above and stays identical across runs for a fixed seed. Required so the Postgres
    # `applicants` table (Phase 2) has a primary key to seed from.
    df.insert(0, "applicant_id", [f"SME-{i + 1:06d}" for i in range(len(df))])
    return df


DEFAULT_CSV_PATH = Path("src/creditsense/data/raw/creditsense_synthetic_portfolio.csv")
DEFAULT_METADATA_PATH = Path("src/creditsense/data/raw/creditsense_synthetic_portfolio.metadata.json")
DATASET_VERSION = "1.1.0"  # bumped from 1.0.0 (fix 1-A): added applicant_id


def write_metadata(df: pd.DataFrame, metadata_path: Path) -> None:
    metadata = {
        "dataset_name": "CreditSense synthetic SME portfolio",
        "dataset_version": DATASET_VERSION,
        "created_date": date.today().isoformat(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "random_seed": RANDOM_SEED,
        "source_generator": "src/creditsense/data/generators/portfolio.py",
        "csv_file": "creditsense_synthetic_portfolio.csv",
        "validation": {
            "test_command": "pytest tests/synthetic_data -v",
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_applicant_ids": int(df["applicant_id"].duplicated().sum()),
        },
        "scope_decisions": {
            "ocr_noise": "skipped (see tests/fixtures/messy_applications.json for a small "
                         "hand-built substitute used to test FinancialAnalystAgent)",
            "currency_format_variation": "skipped (see tests/fixtures/messy_applications.json)",
            "missing_collateral_values": "skipped (see tests/fixtures/messy_applications.json)",
            "urdu_english_notes": "skipped (see tests/fixtures/messy_applications.json)",
        },
        "columns": list(df.columns),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(
    n: int = N_RECORDS,
    seed: int = RANDOM_SEED,
    csv_path: Path = DEFAULT_CSV_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    dataset = build_dataset(n=n, seed=seed)
    validate_dataset(dataset)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(csv_path, index=False)
    write_metadata(dataset, metadata_path)
    print(f"\nSaved {len(dataset):,} rows to {csv_path}")
    print(f"Saved metadata to {metadata_path}")
    return dataset


if __name__ == "__main__":
    main()
