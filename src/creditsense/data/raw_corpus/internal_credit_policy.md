<!--
SOURCE: Simulated. Authored for this project to model a realistic bank/DFI internal
credit policy manual implementing the real SBP Prudential Regulations for SME
Financing (see sbp_prudential_sme_regulations.md). Not a real bank's policy document.
Clauses that implement a specific SBP regulation cite it by number (e.g. "per
Regulation R-5") — these citations are the multi-hop retrieval test cases (a question
about internal policy correctly surfaces both this clause and the SBP regulation it
points to). See MANIFEST.md.
-->

# Part A: Credit Risk Governance

## Policy P-1: Credit Risk Appetite Statement

The Bank's SME credit risk appetite is reviewed annually by the Board Risk Committee
and expressed as: maximum SME portfolio non-performing loan (NPL) ratio of 8%,
maximum single-sector concentration of 25% of the SME book, and minimum portfolio
risk-adjusted return on capital (RAROC) of 14%. Any credit proposal that would breach
an approved appetite limit requires escalation to the Management Committee on SME
Finance established under Regulation R-4.

## Policy P-2: Delegated Credit Authority

Branch Managers may approve SME facilities up to PKR 5 million within their assigned
sector limits. Regional Credit Committees approve up to PKR 50 million. Facilities above
PKR 50 million, or any facility approaching the per-party exposure ceiling in Regulation
R-5, require Head Office Credit Committee approval regardless of amount.

## Policy P-3: Segregation of Duties

The officer who originates an SME credit proposal shall not be the officer who approves
it. Risk assessment, conducted under the framework required by Regulation R-8, shall be
performed by an officer independent of the originating relationship team.

## Policy P-4: Annual Policy Review

This manual is reviewed at least annually by the SME Finance Group and approved by
the Board, consistent with the strategy-setting obligation in Regulation R-1. Interim
amendments required by a new SBP circular take effect within 30 days of the circular's
notification date.

## Policy P-5: Exception Reporting

Any facility approved outside the standard criteria in this manual (a "policy exception")
must be logged in the quarterly exception report reviewed by the Management
Committee on SME Finance per Regulation R-4(i), with the compensating control that
justified the exception.

# Part B: Underwriting Standards by Enterprise Tier

## Policy P-6: Tier Classification at Onboarding

Enterprise tier (Micro, Small, Medium, or Start-up) is assigned at onboarding using the
annual sales turnover bands in Part-I of the SBP Prudential Regulations for SME
Financing, and re-verified at each annual review or renewal. A business active five
years or less is classified Start-up regardless of turnover, per the SBP definition.

## Policy P-7: Minimum Documentation Tier — Bank-Statement-Only

Applicants providing only 12 months of bank statements (no audited or unaudited
financials) may be underwritten for facilities up to PKR 3 million, using bank statement
volatility and cash-flow-based income estimation models as contemplated by Regulation
R-15, subject to a minimum 6-month banking relationship.

## Policy P-8: Minimum Documentation Tier — Unaudited Financials

Applicants providing management-prepared (unaudited) financial statements may be
underwritten up to PKR 30 million, provided the statements are cross-checked against
bank statement turnover within a 15% tolerance.

## Policy P-9: Minimum Documentation Tier — Audited Financials

Applicants providing statutory audited financial statements (required for all Pvt Ltd
(SECP) entities above the Small Enterprise threshold) may be underwritten without a
tier-based facility cap, subject to the per-party exposure limit in Regulation R-5.

## Policy P-10: Credit Scoring Model Governance

Any digital credit scoring model used under Regulation R-15 shall be validated annually
against realized default outcomes, with model inputs, weights and validation results
retained for SBP on-site inspection.

## Policy P-11: Turnaround Time Escalation

Applications approaching the 15 working day TAT ceiling in Regulation R-16 are flagged
to the Regional Credit Committee at day 10; applications breaching TAT are logged for
the Management Committee's remedial review under Regulation R-16(iv).

# Part C: Sector Risk Overlays

The following sector overlays adjust standard underwriting parameters for the eleven
SME sectors tracked in the Bank's portfolio risk model. Each overlay reflects sector-
specific operating conditions and is reviewed annually against realized sector default
rates.

## Policy P-12: Sector Overlay — Textile/Garments

Textile/Garments carries elevated exposure to load-shedding and energy cost
volatility. Facilities to this sector require a documented captive power assessment
where load-shedding impact score exceeds the portfolio's 75th percentile, and are
subject to a 10% reduction in the standard collateral coverage discount.

## Policy P-13: Sector Overlay — Steel/Re-rolling

Steel/Re-rolling combines high energy intensity with high FX import dependency
(imported scrap/billets). Facilities require an FX hedging policy statement from the
applicant where fx_import_dependency_ratio exceeds 0.5, consistent with the risk
management expectations in Regulation R-8.

## Policy P-14: Sector Overlay — Construction/Building Materials

Construction/Building Materials is subject to seasonal cash flow cycles and elevated
buyer concentration risk. Underwriting requires disclosure of the top-3 buyer
concentration percentage and a minimum collateral coverage ratio of 0.75 for facilities
above PKR 10 million.

## Policy P-15: Sector Overlay — Retail/Trade

Retail/Trade, the Bank's largest SME sector by exposure count, is underwritten
primarily on bank statement volatility and digital payment footprint given typically
thin formal documentation, consistent with the bank-statement-only tier described in
Policy P-7.

## Policy P-16: Sector Overlay — Auto Parts/Engineering

Auto Parts/Engineering carries high FX dependency for imported components.
Facilities above PKR 20 million require a 12-month landed-cost sensitivity analysis
against KIBOR and exchange rate movements.

## Policy P-17: Sector Overlay — Leather/Sports Goods

Leather/Sports Goods is predominantly export-oriented; facilities are eligible for
preferential pricing under the Bank's export refinance scheme where export sales
exceed 40% of declared turnover, subject to standard SBP export documentation.

## Policy P-18: Sector Overlay — Food Processing

Food Processing requires food safety/health certification as a standing condition
precedent to disbursement, in addition to the standard documentation tier
requirements in Policy P-7 through P-9.

## Policy P-19: Sector Overlay — Rice/Agri-processing

Rice/Agri-processing financing is seasonal, concentrated around harvest cycles.
Working capital facilities may be structured with a seasonal drawdown/repayment
schedule rather than the standard monthly amortization profile.

## Policy P-20: Sector Overlay — Pharma/Surgical

Pharma/Surgical carries high FX dependency but low energy intensity. DRAP
(Drug Regulatory Authority of Pakistan) registration is a standing condition precedent
for any facility exceeding PKR 5 million.

## Policy P-21: Sector Overlay — IT/Tech Services

IT/Tech Services typically presents thin collateral but strong cash-flow visibility.
Facilities may rely more heavily on the clean facility allowance in Regulation R-9,
subject to the PKR 50 million clean exposure ceiling stated there.

## Policy P-22: Sector Overlay — Other Services

Other Services is underwritten on a case-by-case basis using the closest analogous
sector overlay above, pending a dedicated overlay once sufficient portfolio history
accumulates.

# Part D: Collateral and Security Policy

## Policy P-23: Standard Security Requirements

All facilities other than clean facilities under Regulation R-9 shall be secured per
Regulation R-13. Standard acceptable security includes registered mortgage over
commercial/industrial/residential property, pledge of stock and hypothecation of plant
& machinery, subject to the valuation criteria in Annexure-II of the SBP Prudential
Regulations for SME Financing (Regulation R-17 Annexure-II).

## Policy P-24: Collateral Coverage Ratio Floor

Secured facilities require a minimum collateral coverage ratio of 0.50, except where a
sector overlay in Part C states a higher floor. Facilities falling below this floor require
Regional Credit Committee approval with documented compensating controls.

## Policy P-25: Guarantor Requirements

Where a facility relies in part on a personal guarantee, the guarantor's declared net
worth must be independently verified and shall not be counted toward satisfying the
collateral coverage floor in Policy P-24 — a guarantee supports the clean-facility
allowance in Regulation R-9, not the secured-facility collateral requirement.

## Policy P-26: Valuation Refresh Cycle

Collateral valuations follow the Full-Scope/Desktop Evaluation cycle prescribed in
Annexure-II to Regulation R-17: a Full-Scope Valuation at origination, Desktop
Evaluations in years two and three, and a fresh Full-Scope Valuation in year four.

## Policy P-27: FSV Benefit Application

Provisioning calculations may take the Forced Sale Value benefit described in
Annexure-II to Regulation R-17 only where the underlying collateral documentation
(perfected charge, current insurance, evaluator panel membership) is complete and
current at the classification date.

# Part E: Exposure Aggregation and Group Lending

## Policy P-28: Group Exposure Identification

At onboarding and at each annual review, Relationship Managers shall identify all
group/associate companies sharing common ownership or control with the applicant, so
that aggregate exposure can be measured against the per-party ceiling in Regulation
R-5.

## Policy P-29: Aggregate Exposure Calculation

Group and associate company exposure is aggregated with the applicant's own exposure
for the purpose of the Regulation R-5 ceiling. Where aggregate exposure would exceed
the applicable tier ceiling, the incremental facility is declined regardless of the
individual applicant's standalone creditworthiness.

## Policy P-30: Related Party Screening

Every SME credit proposal is screened against the Bank's related-party register to
confirm compliance with the restriction in Regulation R-11 before submission to any
approving authority.

## Policy P-31: Liquid Asset Deduction

Where an applicant holds liquid assets under the Bank's perfected lien (bank deposits,
certificates of deposit/investment, Pakistan Investment Bonds, Treasury Bills, National
Savings Scheme securities), these may be deducted from gross exposure when
calculating headroom against the Regulation R-5 per-party ceiling, per the deduction
allowance stated in that regulation.

# Part F: Monitoring, Classification and Restructuring

## Policy P-32: Early Warning Indicators

Facilities are flagged for enhanced monitoring on any of: two or more late payments in
a rolling 12-month window, a utility default, working capital cycle days exceeding the
sector's 75th percentile, or a decline in digital payment footprint exceeding 30%
quarter-on-quarter — consistent with the monitoring obligation in Regulation R-10.

## Policy P-33: Classification Trigger Alignment

Internal classification triggers align exactly with the day-count thresholds in Annexure-I
to Regulation R-17 (Substandard at 90 days, Doubtful at 180 days, Loss at one year),
with no internal grace period beyond what Annexure-I allows.

## Policy P-34: Restructuring Approval Authority

Any restructuring or rescheduling of an SME facility requires Regional Credit Committee
approval and must meet the minimum cash recovery conditions in Regulation R-18
before the classification category may be improved.

## Policy P-35: Restructuring Re-default Handling

Where a restructured facility re-defaults, it is reclassified to its pre-restructuring
category immediately, and any mark-up reversal required by Regulation R-18(v) is
processed in the same reporting cycle.

## Policy P-36: Portfolio MIS Reporting

The SME portfolio Management Information System required by Regulation R-19(ii)
reports, at minimum: sector concentration against the 25% appetite limit in Policy P-1,
NPL ratio by sector and by documentation tier, and TAT compliance against the
Regulation R-16 ceiling.

# Part G: KYC/AML and Digital Onboarding

## Policy P-37: KYC Documentation Standard

Minimum KYC for SME onboarding follows the Bank's AML/CFT policy issued by Banking
Policy & Regulations Department, as referenced in the Preface to the SBP Prudential
Regulations for SME Financing, and is completed before any facility is disbursed
regardless of documentation tier.

## Policy P-38: Digital Onboarding Data Reuse

Data collected during digital account opening is reused across the credit assessment
workflow rather than re-collected, per Regulation R-14(a)(ii), reducing turnaround time
against the ceiling in Regulation R-16.

## Policy P-39: Third-Party Data Sourcing

Where statutory or regulatory data is available from an authorized third-party source
(e.g. tax filer status, e-CIB credit report), it is sourced directly rather than requested
from the applicant, per Regulation R-14(a)(iii).

## Policy P-40: Bilingual Disclosure

Loan/financing terms are disclosed to the applicant in both English and Urdu per
Regulation R-12, with the disclosure acknowledgement retained in the credit file
regardless of documentation tier.

## Policy P-41: Complaint Handling Integration

All SME-related complaints are logged and tracked through the Sunwai portal
integration required by Regulation R-19(iv), with status visible to the Management
Committee on SME Finance.
