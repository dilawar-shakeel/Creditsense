"""
CreditSense — Distribution Spec-Adherence Tests
=================================================
Verifies that every generated feature matches the distribution family, parameters, and
bounds specified in Section 3 of the CreditSense_Feature_Synthetic_Data_XGBoost_Guide.md
("Synthetic Data Specification" table). One or more descriptively-named tests per
feature (#1-30), plus the auxiliary turnover field and the two label heads.

These tests check "did we implement the spec correctly" — NOT "is the data realistic",
which is covered separately in test_economic_realism.py.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from creditsense.data.generators import portfolio as gen
from stat_helpers import theoretical_mean_clipped, theoretical_mean_truncated

from conftest import CATEGORICAL_PROPORTION_TOLERANCE_PCT, MOMENT_RELATIVE_TOLERANCE


def assert_close_relative(actual: float, expected: float, tol: float = MOMENT_RELATIVE_TOLERANCE, label: str = ""):
    rel_err = abs(actual - expected) / abs(expected)
    assert rel_err <= tol, (
        f"{label}: actual={actual:.4f} vs theoretical={expected:.4f} "
        f"(relative error {rel_err:.1%} exceeds tolerance {tol:.1%})"
    )


def assert_proportion_close(observed_pct: float, expected_pct: float, label: str,
                             tol_pct: float = CATEGORICAL_PROPORTION_TOLERANCE_PCT):
    assert abs(observed_pct - expected_pct) <= tol_pct, (
        f"{label}: observed {observed_pct:.2f}% vs spec {expected_pct:.2f}% "
        f"(diff {abs(observed_pct - expected_pct):.2f}pp exceeds {tol_pct}pp tolerance)"
    )


# ======================================================================================
# GROUP A — Baseline Financial Ratios (#1-9)
# ======================================================================================

def test_01_current_ratio_is_lognormal_mu015_sigma045_clipped_to_03_40(dataset):
    s = dataset["current_ratio"]
    assert s.min() >= 0.3 - 1e-9 and s.max() <= 4.0 + 1e-9, "current_ratio must stay within the clipped [0.3, 4.0] bound"
    expected_mean = theoretical_mean_clipped(stats.lognorm, 0.3, 4.0, s=0.45, scale=np.exp(0.15))
    assert_close_relative(s.mean(), expected_mean, label="current_ratio mean")


def test_02_debt_to_equity_ratio_is_lognormal_mu050_sigma070_clipped_to_0_8(dataset):
    s = dataset["debt_to_equity_ratio"]
    assert s.min() >= 0.0 and s.max() <= 8.0 + 1e-9, "debt_to_equity_ratio must stay within the clipped [0, 8.0] bound"
    expected_mean = theoretical_mean_clipped(stats.lognorm, 0.0, 8.0, s=0.70, scale=np.exp(0.50))
    assert_close_relative(s.mean(), expected_mean, label="debt_to_equity_ratio mean")


def test_03_revenue_growth_yoy_is_normal_mu8_sigma18_clipped_to_neg40_80(dataset):
    s = dataset["revenue_growth_yoy_pct"]
    assert s.min() >= -40.0 - 1e-9 and s.max() <= 80.0 + 1e-9, "revenue_growth_yoy_pct must stay within [-40, 80]"
    expected_mean = theoretical_mean_clipped(stats.norm, -40.0, 80.0, loc=8.0, scale=18.0)
    assert_close_relative(s.mean(), expected_mean, label="revenue_growth_yoy_pct mean")


def test_04_sector_risk_code_matches_the_eleven_specified_sector_weights(dataset):
    observed = dataset["sector_risk_code"].value_counts(normalize=True) * 100
    for name, prob in zip(gen.SECTOR_NAMES, gen.SECTOR_PROBS):
        assert_proportion_close(observed.get(name, 0.0), prob * 100, label=f"sector share: {name}")


def test_04b_sector_risk_code_has_exactly_the_eleven_specified_categories(dataset):
    assert set(dataset["sector_risk_code"].unique()) == set(gen.SECTOR_NAMES)


def test_05_years_in_business_is_gamma_shape2_scale45_truncated_to_05_40_no_boundary_spike(dataset):
    s = dataset["years_in_business"]
    assert s.min() >= 0.5 - 1e-6 and s.max() <= 40.0 + 1e-6, "years_in_business must stay within [0.5, 40]"
    expected_mean = theoretical_mean_truncated(stats.gamma, 0.5, 40.0, a=2.0, scale=4.5)
    assert_close_relative(s.mean(), expected_mean, label="years_in_business mean")
    # Truncation-specific check: no artificial pile-up of mass at the upper boundary,
    # which is what a naive clip (instead of a proper truncated inverse-CDF) would cause.
    at_ceiling_pct = 100 * (s >= 40.0 * 0.999).mean()
    assert at_ceiling_pct < 0.5, (
        f"years_in_business shows {at_ceiling_pct:.2f}% of rows piled at the 40-year "
        f"ceiling — indicates clipping was used instead of proper truncation"
    )


def test_06_bank_statement_volatility_is_lognormal_muNeg050_sigma060_clipped_015_25(dataset):
    s = dataset["bank_statement_volatility"]
    assert s.min() >= 0.15 - 1e-9 and s.max() <= 2.5 + 1e-9
    expected_mean = theoretical_mean_clipped(stats.lognorm, 0.15, 2.5, s=0.60, scale=np.exp(-0.50))
    assert_close_relative(s.mean(), expected_mean, label="bank_statement_volatility mean")


def test_07_existing_loan_exposure_is_lognormal_mu145_sigma11_clipped_0_150M(dataset):
    s = dataset["existing_loan_exposure_pkr"]
    assert s.min() >= 0.0 and s.max() <= 150_000_000 + 1e-3
    expected_mean = theoretical_mean_clipped(stats.lognorm, 0.0, 150_000_000, s=1.1, scale=np.exp(14.5))
    assert_close_relative(s.mean(), expected_mean, label="existing_loan_exposure_pkr mean")


def test_08_late_payment_count_is_zero_inflated_poisson_pi055_lambda12_capped_12(dataset):
    s = dataset["late_payment_count_12m"]
    assert s.min() >= 0 and s.max() <= 12, "late_payment_count_12m must be a non-negative integer capped at 12"
    zero_pct = 100 * (s == 0).mean()
    # P(X=0) = pi0 + (1-pi0)*exp(-lambda) = 0.55 + 0.45*exp(-1.2)
    expected_zero_pct = 100 * (0.55 + 0.45 * np.exp(-1.2))
    assert_proportion_close(zero_pct, expected_zero_pct, label="late_payment_count_12m zero-rate")
    expected_mean = (1 - 0.55) * 1.2  # E[X] for ZIP, cap-induced truncation negligible here
    assert_close_relative(s.mean(), expected_mean, tol=0.15, label="late_payment_count_12m mean")


def test_09_collateral_coverage_ratio_is_beta_2_3_scaled_to_0_25(dataset):
    s = dataset["collateral_coverage_ratio"]
    assert s.min() >= 0.0 and s.max() <= 2.5 + 1e-9, "collateral_coverage_ratio must stay within [0, 2.5]"
    expected_mean = stats.beta.mean(2, 3) * 2.5  # Beta support is naturally bounded, no clip needed
    assert_close_relative(s.mean(), expected_mean, label="collateral_coverage_ratio mean")


# ======================================================================================
# GROUP B — Macro/Rate & Sector-Structural Exposure (#10-14)
# ======================================================================================

def test_10_kibor_sensitivity_is_beta_5_2_scaled_to_0_1(dataset):
    s = dataset["kibor_sensitivity_pct"]
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9
    assert_close_relative(s.mean(), stats.beta.mean(5, 2), label="kibor_sensitivity_pct mean")


def test_11_load_shedding_score_matches_guides_explicit_textile_steel_high_group_mu7(dataset):
    high_group_sectors = ["Textile/Garments", "Steel/Re-rolling"]  # guide's explicit example
    sub = dataset.loc[dataset["sector_risk_code"].isin(high_group_sectors), "load_shedding_impact_score"]
    assert_close_relative(sub.mean(), 7.0, tol=0.15, label="load_shedding_impact_score, Textile/Steel group")


def test_11b_load_shedding_score_matches_guides_explicit_it_services_lowest_group_mu1(dataset):
    sub = dataset.loc[dataset["sector_risk_code"] == "IT/Tech Services", "load_shedding_impact_score"]
    assert_close_relative(sub.mean(), 1.0, tol=0.30, label="load_shedding_impact_score, IT/Services group")


def test_11c_load_shedding_score_matches_guides_explicit_retail_trade_low_group_mu3(dataset):
    sub = dataset.loc[dataset["sector_risk_code"] == "Retail/Trade", "load_shedding_impact_score"]
    assert_close_relative(sub.mean(), 3.0, tol=0.15, label="load_shedding_impact_score, Retail/Trade group")


def test_11d_load_shedding_score_matches_guides_explicit_construction_food_mid_group_mu5(dataset):
    mid_group_sectors = ["Construction/Bldg Materials", "Food Processing"]  # guide's explicit example
    sub = dataset.loc[dataset["sector_risk_code"].isin(mid_group_sectors), "load_shedding_impact_score"]
    assert_close_relative(sub.mean(), 5.0, tol=0.15, label="load_shedding_impact_score, Construction/Food group")


def test_11e_load_shedding_score_stays_within_clipped_0_10_bound(dataset):
    s = dataset["load_shedding_impact_score"]
    assert s.min() >= 0.0 and s.max() <= 10.0 + 1e-9


def test_12_captive_power_dependency_is_bernoulli_p035_industrial_sectors_only(dataset):
    industrial_sectors = [name for name, ind in gen.SECTOR_INDUSTRIAL.items() if ind]
    industrial_rate = 100 * dataset.loc[dataset["sector_risk_code"].isin(industrial_sectors), "captive_power_dependency"].mean()
    assert_proportion_close(industrial_rate, 35.0, label="captive_power_dependency rate, industrial sectors")


def test_12b_captive_power_dependency_is_always_zero_for_non_industrial_sectors(dataset):
    non_industrial = [name for name, ind in gen.SECTOR_INDUSTRIAL.items() if not ind]
    rate = dataset.loc[dataset["sector_risk_code"].isin(non_industrial), "captive_power_dependency"].mean()
    assert rate == 0.0, "Non-industrial sectors must never show captive power dependency"


def test_12c_captive_power_capacity_is_lognormal_mu40_sigma08_when_dependent_else_zero(dataset):
    dependent = dataset.loc[dataset["captive_power_dependency"] == 1, "captive_power_capacity_kw"]
    not_dependent = dataset.loc[dataset["captive_power_dependency"] == 0, "captive_power_capacity_kw"]
    assert (not_dependent == 0.0).all(), "captive_power_capacity_kw must be exactly 0 where dependency flag is 0"
    assert_close_relative(dependent.mean(), np.exp(4.0 + 0.8 ** 2 / 2), tol=0.15, label="captive_power_capacity_kw mean")


def test_13_fx_import_dependency_matches_guides_explicit_auto_parts_pharma_high_group(dataset):
    high_fx_sectors = ["Auto Parts/Engineering", "Pharma/Surgical"]  # guide's explicit example
    sub = dataset.loc[dataset["sector_risk_code"].isin(high_fx_sectors), "fx_import_dependency_ratio"]
    assert_close_relative(sub.mean(), stats.beta.mean(3, 4), tol=0.15, label="fx_import_dependency_ratio, Auto/Pharma group")


def test_13b_fx_import_dependency_matches_guides_explicit_retail_food_low_group(dataset):
    low_fx_sectors = ["Retail/Trade", "Food Processing"]  # guide's explicit example
    sub = dataset.loc[dataset["sector_risk_code"].isin(low_fx_sectors), "fx_import_dependency_ratio"]
    assert_close_relative(sub.mean(), stats.beta.mean(1.5, 5), tol=0.15, label="fx_import_dependency_ratio, Retail/Food group")


def test_13c_fx_import_dependency_ratio_stays_within_0_1_bound(dataset):
    s = dataset["fx_import_dependency_ratio"]
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9


def test_14_seasonal_cash_flow_index_is_lognormal_mu050_sigma035_clipped_1_5(dataset):
    s = dataset["seasonal_cash_flow_index"]
    assert s.min() >= 1.0 - 1e-9 and s.max() <= 5.0 + 1e-9
    expected_mean = theoretical_mean_clipped(stats.lognorm, 1.0, 5.0, s=0.35, scale=np.exp(0.50))
    assert_close_relative(s.mean(), expected_mean, label="seasonal_cash_flow_index mean")


# ======================================================================================
# GROUP C — Documentation, Formalization & Tax Behavior (#15-20)
# ======================================================================================

def test_15_tax_filer_status_matches_filer38_latefiler12_nonfiler50(dataset):
    observed = dataset["tax_filer_status"].value_counts(normalize=True) * 100
    assert_proportion_close(observed.get("Filer", 0), 38.0, label="tax_filer_status: Filer")
    assert_proportion_close(observed.get("Late-Filer", 0), 12.0, label="tax_filer_status: Late-Filer")
    assert_proportion_close(observed.get("Non-Filer", 0), 50.0, label="tax_filer_status: Non-Filer")


def test_16_declared_income_ratio_full_population_marginal_is_beta_2_5_scaled_to_0_1(dataset):
    # NOTE: we deliberately test the FULL population here, not a tax_filer_status subset.
    # By construction of the quantile-transform copula, declared_income_ratio's own
    # marginal is exactly Beta(2,5) regardless of its correlation with tax_filer_status
    # (correlation reshuffles the JOINT distribution, not each variable's own marginal).
    # The Req-4.1 coherence cap only clips the upper tail (P(Beta(2,5) > 0.5) ~= 11%,
    # P(Beta(2,5) > 0.75) ~= 0.5%), so its effect on the population-wide mean is small
    # and the generous tolerance below absorbs it.
    s = dataset["declared_income_ratio"]
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9
    assert_close_relative(s.mean(), stats.beta.mean(2, 5), tol=0.10, label="declared_income_ratio (full population) mean")


def test_17_sbp_enterprise_tier_is_deterministically_derived_from_turnover_for_every_row(dataset):
    def expected_tier(t):
        for tier, (lo, hi) in gen.SBP_TIER_TURNOVER_BREAKS.items():
            if lo <= t < hi:
                return tier
        return "ME"
    mismatches = (dataset["sbp_enterprise_tier"] != dataset["annual_bank_turnover_pkr"].apply(expected_tier)).sum()
    assert mismatches == 0, f"{mismatches} rows have an sbp_enterprise_tier inconsistent with their turnover"


def test_17b_is_startup_flag_is_deterministically_years_in_business_lte_5(dataset):
    expected = (dataset["years_in_business"] <= 5).astype(int)
    assert (dataset["is_startup_flag"] == expected).all(), "is_startup_flag must exactly equal (years_in_business <= 5)"


def test_18_documentation_tier_matches_audited15_unaudited45_bankstatement40(dataset):
    observed = dataset["documentation_tier"].value_counts(normalize=True) * 100
    assert_proportion_close(observed.get("Audited Financials", 0), 15.0, label="documentation_tier: Audited")
    assert_proportion_close(observed.get("Unaudited Financials", 0), 45.0, label="documentation_tier: Unaudited")
    assert_proportion_close(observed.get("Bank-Statement-Only", 0), 40.0, label="documentation_tier: Bank-Statement-Only")


def test_19_legal_structure_matches_soleprop55_partnership25_pvtltd20(dataset):
    observed = dataset["legal_structure"].value_counts(normalize=True) * 100
    assert_proportion_close(observed.get("Sole Proprietorship", 0), 55.0, label="legal_structure: Sole Proprietorship")
    assert_proportion_close(observed.get("Partnership/AOP", 0), 25.0, label="legal_structure: Partnership/AOP")
    assert_proportion_close(observed.get("Pvt Ltd (SECP)", 0), 20.0, label="legal_structure: Pvt Ltd (SECP)")


def test_20_labor_formalization_ratio_is_beta_15_4_scaled_to_0_1(dataset):
    s = dataset["labor_formalization_ratio"]
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9
    assert_close_relative(s.mean(), stats.beta.mean(1.5, 4), label="labor_formalization_ratio mean")


# ======================================================================================
# GROUP D — Behavioral & Bureau Signals (#21-24)
# ======================================================================================

def test_21_utility_default_count_is_zero_inflated_poisson_pi065_lambda08_capped_10(dataset):
    s = dataset["utility_default_count_12m"]
    assert s.min() >= 0 and s.max() <= 10
    zero_pct = 100 * (s == 0).mean()
    expected_zero_pct = 100 * (0.65 + 0.35 * np.exp(-0.8))
    assert_proportion_close(zero_pct, expected_zero_pct, label="utility_default_count_12m zero-rate")


def test_22_ecib_score_is_normal_mu650_sigma90_truncated_to_300_850(dataset):
    s = dataset["ecib_score"].dropna()
    assert s.min() >= 300.0 - 1e-6 and s.max() <= 850.0 + 1e-6
    expected_mean = theoretical_mean_truncated(stats.norm, 300, 850, loc=650, scale=90)
    assert_close_relative(s.mean(), expected_mean, tol=0.10, label="ecib_score mean")
    at_ceiling_pct = 100 * (s >= 850.0 * 0.999).mean()
    assert at_ceiling_pct < 1.0, "ecib_score shows a boundary pile-up, indicating clipping rather than truncation"


def test_22b_ecib_no_hit_flag_occurs_for_approximately_20_percent_of_borrowers(dataset):
    assert_proportion_close(100 * dataset["ecib_no_hit_flag"].mean(), 20.0, tol_pct=3.0, label="ecib_no_hit_flag rate")


def test_22c_ecib_score_is_null_if_and_only_if_no_hit_flag_is_set(dataset):
    null_mask = dataset["ecib_score"].isna()
    flag_mask = dataset["ecib_no_hit_flag"] == 1
    assert (null_mask == flag_mask).all(), "ecib_score must be NaN exactly where ecib_no_hit_flag == 1, and only there"


def test_23_multi_banking_behavior_is_poisson_lambda23_with_floor_1(dataset):
    s = dataset["multi_banking_behavior"]
    assert s.min() >= 1, "multi_banking_behavior must never be below the floor of 1"
    assert_close_relative(s.mean(), 2.3 + 1, label="multi_banking_behavior mean")


def test_24_digital_payment_footprint_is_beta_15_5_scaled_to_0_1(dataset):
    s = dataset["digital_payment_footprint"]
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9
    assert_close_relative(s.mean(), stats.beta.mean(1.5, 5), label="digital_payment_footprint mean")


# ======================================================================================
# GROUP E — Structural & Relationship Risk (#25-30)
# ======================================================================================

def test_25_buyer_concentration_top3_is_beta_2_2_scaled_to_0_1(dataset):
    s = dataset["buyer_concentration_top3_pct"]
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9
    assert_close_relative(s.mean(), stats.beta.mean(2, 2), label="buyer_concentration_top3_pct mean")


def test_26_group_associate_exposure_is_zero_inflated_lognormal_pi070_mu140_sigma13(dataset):
    s = dataset["group_associate_exposure_pkr"]
    zero_pct = 100 * (s == 0.0).mean()
    assert_proportion_close(zero_pct, 70.0, label="group_associate_exposure_pkr zero-rate")
    nonzero = s[s > 0.0]
    assert_close_relative(nonzero.mean(), np.exp(14.0 + 1.3 ** 2 / 2), tol=0.15, label="group_associate_exposure_pkr mean (nonzero subset)")


def test_27_guarantor_net_worth_is_lognormal_mu155_sigma10_clipped_0_500M(dataset):
    s = dataset["guarantor_net_worth_pkr"]
    assert s.min() >= 0.0 and s.max() <= 500_000_000 + 1e-3
    expected_mean = theoretical_mean_clipped(stats.lognorm, 0.0, 500_000_000, s=1.0, scale=np.exp(15.5))
    assert_close_relative(s.mean(), expected_mean, label="guarantor_net_worth_pkr mean")


def test_28_business_premises_owned_is_bernoulli_p040(dataset):
    assert_proportion_close(100 * dataset["business_premises_owned"].mean(), 40.0, label="business_premises_owned rate")


def test_29_trade_body_membership_is_bernoulli_p025(dataset):
    assert_proportion_close(100 * dataset["trade_body_membership"].mean(), 25.0, label="trade_body_membership rate")


def test_30_working_capital_cycle_days_is_normal_mu75_sigma30_clipped_10_200(dataset):
    s = dataset["working_capital_cycle_days"]
    assert s.min() >= 10.0 - 1e-9 and s.max() <= 200.0 + 1e-9
    expected_mean = theoretical_mean_clipped(stats.norm, 10, 200, loc=75, scale=30)
    assert_close_relative(s.mean(), expected_mean, label="working_capital_cycle_days mean")


# ======================================================================================
# AUXILIARY FIELD (not one of the named 30, but required by Section 2 / #17)
# ======================================================================================

def test_31_annual_bank_turnover_is_lognormal_mu172_sigma15_truncated_to_500k_800M(dataset):
    s = dataset["annual_bank_turnover_pkr"]
    assert s.min() >= 500_000 - 1e-3 and s.max() <= 800_000_000 + 1e-3
    expected_mean = theoretical_mean_truncated(stats.lognorm, 500_000, 800_000_000, s=1.5, scale=np.exp(17.2))
    assert_close_relative(s.mean(), expected_mean, tol=0.15, label="annual_bank_turnover_pkr mean")
    at_ceiling_pct = 100 * (s >= 800_000_000 * 0.999).mean()
    assert at_ceiling_pct < 0.5, "annual_bank_turnover_pkr shows a boundary pile-up at 800M, expected smooth truncation"


# ======================================================================================
# LABEL HEADS (Section 5)
# ======================================================================================

def test_32_default_probability_12m_is_binary_with_no_missing_values(dataset):
    s = dataset["default_probability_12m"]
    assert set(s.unique()) <= {0, 1}
    assert s.isna().sum() == 0


def test_33_default_probability_12m_realized_rate_falls_within_the_8_to_12_percent_band(dataset):
    rate = dataset["default_probability_12m"].mean()
    assert 0.08 <= rate <= 0.12, f"Realized default rate {rate:.2%} falls outside the specified 8-12% band"


def test_34_risk_index_score_is_a_valid_probability_between_0_and_1(dataset):
    s = dataset["risk_index_score"]
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s.isna().sum() == 0


def test_35_recommend_credit_limit_is_non_negative_with_no_missing_values(dataset):
    s = dataset["recommend_credit_limit_pkr"]
    assert s.min() >= 0.0
    assert s.isna().sum() == 0
