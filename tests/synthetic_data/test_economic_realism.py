"""
CreditSense — Economic Realism & Regulatory Coherence Tests
=============================================================
Verifies that the generated data doesn't just match its distribution specs (see
test_distribution_spec_adherence.py) but also makes real-world economic and
regulatory sense: correlated risk factors move together in the right direction,
SBP caps are never breached, derived fields stay logically consistent with the
fields they're derived from, and the label heads carry genuine, recoverable signal.
"""

import numpy as np
import pandas as pd
import pytest

import creditsense_generator as gen


# ======================================================================================
# SECTION 4 — Latent-Factor Correlation Structure ("informality clusters together")
# ======================================================================================

def test_current_ratio_and_bank_statement_volatility_are_negatively_correlated(dataset):
    # Both driven by latent Factor 2 (Liquidity Discipline), opposite directions:
    # disciplined cash management means a healthier current ratio AND steadier inflows.
    r = dataset["current_ratio"].corr(dataset["bank_statement_volatility"])
    assert r < -0.15, f"Expected a meaningfully negative correlation, got r={r:.3f}"


def test_late_payment_count_and_bank_statement_volatility_are_positively_correlated(dataset):
    # Both driven by latent Factor 2: a business with volatile cash inflows is also
    # more likely to miss payments — this is the core "liquidity discipline" cluster.
    r = dataset["late_payment_count_12m"].corr(dataset["bank_statement_volatility"])
    assert r > 0.15, f"Expected a meaningfully positive correlation, got r={r:.3f}"


def test_declared_income_ratio_is_lower_for_non_filers_than_filers(dataset):
    # Both driven by latent Factor 1 (Formality): a Non-Filer, by definition, has not
    # declared income to FBR, so their declared-vs-turnover ratio should sit well below
    # a compliant Filer's — this is the central "informality gap" the guide calls out.
    non_filer_mean = dataset.loc[dataset["tax_filer_status"] == "Non-Filer", "declared_income_ratio"].mean()
    filer_mean = dataset.loc[dataset["tax_filer_status"] == "Filer", "declared_income_ratio"].mean()
    assert non_filer_mean < filer_mean, (
        f"Non-Filer mean declared_income_ratio ({non_filer_mean:.3f}) should be lower "
        f"than Filer mean ({filer_mean:.3f})"
    )


def test_ecib_score_is_lower_on_average_for_non_filers_than_filers(dataset):
    # Both driven by latent Factor 1: informal businesses should skew toward weaker
    # (or missing) bureau profiles relative to compliant, formally-documented ones.
    non_filer_mean = dataset.loc[dataset["tax_filer_status"] == "Non-Filer", "ecib_score"].mean()
    filer_mean = dataset.loc[dataset["tax_filer_status"] == "Filer", "ecib_score"].mean()
    assert non_filer_mean < filer_mean, (
        f"Non-Filer mean ecib_score ({non_filer_mean:.1f}) should be lower than "
        f"Filer mean ({filer_mean:.1f})"
    )


def test_audited_financials_are_more_common_among_filers_than_non_filers(dataset):
    # Categorical association check for the Factor-1 cluster: proper audited
    # bookkeeping should be far more prevalent among tax-compliant Filers.
    filer_audited_rate = (dataset.loc[dataset["tax_filer_status"] == "Filer", "documentation_tier"] == "Audited Financials").mean()
    non_filer_audited_rate = (dataset.loc[dataset["tax_filer_status"] == "Non-Filer", "documentation_tier"] == "Audited Financials").mean()
    assert filer_audited_rate > non_filer_audited_rate, (
        f"Audited-financials rate should be higher among Filers ({filer_audited_rate:.2%}) "
        f"than Non-Filers ({non_filer_audited_rate:.2%})"
    )


def test_factor_1_and_factor_2_features_are_not_simply_independent_of_each_other(dataset):
    # Sanity check on the ~0.45 Factor1-Factor2 correlation surfacing in the data: an
    # informality proxy (is_non_filer) and a liquidity-discipline proxy (late payments)
    # should show a positive association, not zero, confirming rows weren't sampled i.i.d.
    r = dataset["is_non_filer"].astype(float).corr(dataset["late_payment_count_12m"].astype(float))
    assert r > 0.05, (
        f"Expected a detectable positive correlation from the Factor1-Factor2 copula "
        f"link (r=0.45), got r={r:.3f} — data may have been sampled independently"
    )


# ======================================================================================
# SECTION 1/2/17 — SBP Enterprise Tiering & Regulatory Cap Compliance
# ======================================================================================

def test_no_borrower_exceeds_the_800_million_sme_scope_ceiling(dataset):
    # Per the current (06-Nov-2025 / eff. 01-Jan-2026) SBP PR-SME circular, turnover
    # above PKR 800M falls outside the SME definition entirely.
    assert dataset["annual_bank_turnover_pkr"].max() <= 800_000_000 + 1.0


def test_every_recommended_credit_limit_respects_its_tier_secured_ceiling_when_secured(dataset):
    secured = dataset[dataset["collateral_coverage_ratio"] >= 0.10]
    ceilings = secured["sbp_enterprise_tier"].map(gen.SBP_TIER_SECURED_CEILING)
    violations = secured.loc[secured["recommend_credit_limit_pkr"] > ceilings + 1.0]
    assert len(violations) == 0, (
        f"{len(violations)} secured-facility rows exceed their tier's Reg. R-5 ceiling"
    )


def test_every_recommended_credit_limit_respects_the_flat_clean_lending_cap_when_unsecured(dataset):
    unsecured = dataset[dataset["collateral_coverage_ratio"] < 0.10]
    violations = unsecured.loc[unsecured["recommend_credit_limit_pkr"] > gen.SBP_CLEAN_LENDING_CAP + 1.0]
    assert len(violations) == 0, (
        f"{len(violations)} unsecured/clean-facility rows exceed the Reg. R-9 flat cap "
        f"of PKR {gen.SBP_CLEAN_LENDING_CAP:,}"
    )


def test_group_exposure_cap_flag_is_only_ever_set_when_group_exposure_is_actually_nonzero(dataset):
    flagged = dataset[dataset["credit_limit_hit_group_cap"] == 1]
    assert (flagged["group_associate_exposure_pkr"] > 0).all(), (
        "credit_limit_hit_group_cap should never fire for a borrower with zero "
        "group/associate exposure — there is nothing to cap against"
    )


def test_recommended_credit_limit_never_exceeds_20_percent_financing_rate_of_turnover_by_a_wide_margin(dataset):
    # Sense check on the underlying business formula (Section 5): even with the maximum
    # collateral uplift, the limit shouldn't runaway to an economically absurd multiple
    # of the borrower's own verified turnover.
    ratio = dataset["recommend_credit_limit_pkr"] / dataset["annual_bank_turnover_pkr"].clip(lower=1)
    assert ratio.max() < 1.0, (
        f"Some recommended credit limits exceed 100% of the borrower's annual turnover "
        f"(max ratio={ratio.max():.2f}), which is not a economically sound working-capital limit"
    )


def test_micro_tier_borrowers_never_exceed_30_million_turnover(dataset):
    micro = dataset[dataset["sbp_enterprise_tier"] == "Micro"]
    assert micro["annual_bank_turnover_pkr"].max() < 30_000_000


def test_medium_enterprise_tier_borrowers_have_turnover_strictly_above_150_million(dataset):
    me = dataset[dataset["sbp_enterprise_tier"] == "ME"]
    assert me["annual_bank_turnover_pkr"].min() >= 150_000_000


# ======================================================================================
# Req 4.1 — Cross-Feature Logical Coherence
# ======================================================================================

def test_non_filers_never_show_a_declared_income_ratio_above_the_50_percent_ceiling(dataset):
    non_filers = dataset[dataset["tax_filer_status"] == "Non-Filer"]
    assert non_filers["declared_income_ratio"].max() <= 0.50 + 1e-9, (
        "A Non-Filer showing a declared-income ratio above 50% is not economically "
        "coherent — by definition they have not declared income to FBR"
    )


def test_late_filers_never_show_a_declared_income_ratio_above_the_75_percent_ceiling(dataset):
    late_filers = dataset[dataset["tax_filer_status"] == "Late-Filer"]
    assert late_filers["declared_income_ratio"].max() <= 0.75 + 1e-9


# ======================================================================================
# Sector-Conditioning Plausibility (Group B)
# ======================================================================================

def test_industrial_sectors_show_higher_average_load_shedding_impact_than_services_sectors(dataset):
    industrial_sectors = [name for name, ind in gen.SECTOR_INDUSTRIAL.items() if ind]
    services_sectors = [name for name, ind in gen.SECTOR_INDUSTRIAL.items() if not ind]
    industrial_mean = dataset.loc[dataset["sector_risk_code"].isin(industrial_sectors), "load_shedding_impact_score"].mean()
    services_mean = dataset.loc[dataset["sector_risk_code"].isin(services_sectors), "load_shedding_impact_score"].mean()
    assert industrial_mean > services_mean, (
        f"Energy-intensive industrial sectors ({industrial_mean:.2f}) should show higher "
        f"average load-shedding impact than services sectors ({services_mean:.2f})"
    )


def test_it_tech_services_shows_near_zero_captive_power_dependency(dataset):
    # A services sector with no heavy machinery has no plausible reason to run a
    # captive diesel/gas generator.
    rate = dataset.loc[dataset["sector_risk_code"] == "IT/Tech Services", "captive_power_dependency"].mean()
    assert rate == 0.0


def test_textile_garments_shows_a_meaningfully_higher_captive_power_rate_than_it_services(dataset):
    textile_rate = dataset.loc[dataset["sector_risk_code"] == "Textile/Garments", "captive_power_dependency"].mean()
    it_rate = dataset.loc[dataset["sector_risk_code"] == "IT/Tech Services", "captive_power_dependency"].mean()
    assert textile_rate > it_rate


# ======================================================================================
# Missingness Design (ecib_score MNAR pattern)
# ======================================================================================

def test_ecib_no_hit_rate_is_higher_for_non_filers_than_filers_reflecting_thinner_bureau_files(dataset):
    non_filer_nohit = dataset.loc[dataset["tax_filer_status"] == "Non-Filer", "ecib_no_hit_flag"].mean()
    filer_nohit = dataset.loc[dataset["tax_filer_status"] == "Filer", "ecib_no_hit_flag"].mean()
    assert non_filer_nohit > filer_nohit, (
        f"Non-Filers ({non_filer_nohit:.2%} no-hit) should show a higher bureau "
        f"no-hit rate than Filers ({filer_nohit:.2%}) — informality should correlate "
        f"with a thinner or absent credit file, not a random one"
    )


# ======================================================================================
# Label Head Signal Quality (Section 5) — does the label carry recoverable signal?
# ======================================================================================

def test_default_rate_increases_monotonically_across_risk_index_score_quartiles(dataset):
    # The single most important economic-sense check on the label: businesses the
    # generator itself judged riskier should actually default more often. A flat or
    # non-monotonic relationship would mean the label carries no learnable signal.
    quartile = pd.qcut(dataset["risk_index_score"], 4, labels=["Q1_lowest_risk", "Q2", "Q3", "Q4_highest_risk"])
    default_rate_by_quartile = dataset.groupby(quartile, observed=True)["default_probability_12m"].mean()
    rates = default_rate_by_quartile.tolist()
    assert rates == sorted(rates), (
        f"Default rate should rise monotonically from the lowest- to the "
        f"highest-risk quartile, got {dict(default_rate_by_quartile)}"
    )
    assert rates[-1] > rates[0] * 3, (
        "The highest-risk quartile's default rate should be substantially higher than "
        "the lowest-risk quartile's — otherwise the risk index carries too weak a "
        "signal to be useful for underwriting"
    )


def test_non_filers_default_at_a_higher_rate_than_filers(dataset):
    non_filer_rate = dataset.loc[dataset["tax_filer_status"] == "Non-Filer", "default_probability_12m"].mean()
    filer_rate = dataset.loc[dataset["tax_filer_status"] == "Filer", "default_probability_12m"].mean()
    assert non_filer_rate > filer_rate, (
        f"Non-Filer default rate ({non_filer_rate:.2%}) should exceed Filer default "
        f"rate ({filer_rate:.2%}) — informality is a genuine risk driver, not noise"
    )


def test_higher_collateral_coverage_is_associated_with_a_lower_default_rate(dataset):
    tercile = pd.qcut(dataset["collateral_coverage_ratio"], 3, labels=["low", "mid", "high"], duplicates="drop")
    default_rate_by_tercile = dataset.groupby(tercile, observed=True)["default_probability_12m"].mean()
    assert default_rate_by_tercile.iloc[0] > default_rate_by_tercile.iloc[-1], (
        f"Low-collateral borrowers should default more often than high-collateral "
        f"borrowers, got {dict(default_rate_by_tercile)}"
    )


# ======================================================================================
# Req 5.4/5.5 — Target-Leakage Guardrails (structural, not statistical)
# ======================================================================================

def test_risk_index_score_is_documented_as_excluded_from_default_prediction_feature_sets():
    # This is a structural/documentation guardrail, not a statistical one: confirms the
    # leakage warning actually lives in the generator's source, not just in a chat
    # message that nobody re-reads before training a model.
    import inspect
    source = inspect.getsource(gen.generate_labels)
    assert "LEAKAGE" in source.upper(), (
        "generate_labels() must carry an explicit, in-source target-leakage warning "
        "on risk_index_score per Requirement 5.4"
    )


def test_risk_index_score_is_perfectly_predictive_of_the_pre_noise_label_confirming_leakage_risk(dataset):
    # Demonstrates WHY the leakage warning matters: risk_index_score alone separates
    # defaulters from non-defaulters far too well to be a legitimate model input,
    # because it IS (up to the 4% flip) the mechanism that generated the label.
    defaulters = dataset.loc[dataset["default_probability_12m"] == 1, "risk_index_score"]
    non_defaulters = dataset.loc[dataset["default_probability_12m"] == 0, "risk_index_score"]
    assert defaulters.mean() > non_defaulters.mean() * 1.5, (
        "risk_index_score should be dramatically higher, on average, for defaulters "
        "than non-defaulters — confirming it is label-derived and must be excluded "
        "from any default-prediction feature set"
    )


# ======================================================================================
# General Structural Sanity (applies across the whole portfolio)
# ======================================================================================

def test_no_duplicate_rows_in_the_generated_portfolio(dataset):
    # At n=50,000 with continuous latent draws, an exact duplicate row is virtually
    # impossible unless something in the generator collapsed to a constant.
    assert dataset.duplicated().sum() == 0


def test_no_monetary_field_contains_a_negative_value(dataset):
    monetary_cols = [
        "existing_loan_exposure_pkr", "group_associate_exposure_pkr", "guarantor_net_worth_pkr",
        "annual_bank_turnover_pkr", "recommend_credit_limit_pkr",
    ]
    for col in monetary_cols:
        assert (dataset[col] >= 0).all(), f"{col} contains a negative value, which has no economic meaning"


def test_sector_risk_weight_column_is_internally_consistent_with_sector_risk_code(dataset):
    expected = dataset["sector_risk_code"].map(gen.SECTOR_RISK_WEIGHT)
    assert (dataset["sector_risk_weight"] == expected).all()
