"""
P3.9 (stretch) — a small fairness sanity check across sectors.

Not a formal fairness audit; a quick table an interviewer's question can point at.
For each sector: the actual default rate, the approval rate the trained model would
give (at its shipped cutoff), and the model's average predicted risk. Flags any sector
whose approval rate is unusually far from the portfolio average, which is worth
explaining rather than something automatically wrong — sectors differ in real risk, so
some spread is expected.

Requires a trained bundle (creditsense.ml.train.main()) to already exist.

Run: python -m creditsense.ml.fairness_check
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditsense.ml.models import CreditSenseBundle

DEFAULT_CSV_PATH = Path("src/creditsense/data/raw/creditsense_synthetic_portfolio.csv")
DEFAULT_BUNDLE_PATH = Path("src/creditsense/ml/artifacts/creditsense_bundle.joblib")
DEFAULT_REPORT_PATH = Path("src/creditsense/ml/artifacts/fairness_by_sector.md")

# A sector's approval rate more than this many percentage points from the portfolio
# average gets flagged for a look — not proof of a problem, a prompt to explain it.
FLAG_THRESHOLD_PCT = 15.0


def build_report(frame: pd.DataFrame, bundle: CreditSenseBundle) -> str:
    stage1_frame = bundle.stage1_preprocessor.transform(frame)
    probabilities = bundle.stage1_model.predict_proba(stage1_frame)[:, 1]
    approved = probabilities < bundle.stage1_threshold

    working = pd.DataFrame(
        {
            "sector_risk_code": frame["sector_risk_code"].to_numpy(),
            "actual_default": frame["default_probability_12m"].to_numpy(),
            "predicted_risk": probabilities,
            "approved": approved,
        }
    )
    by_sector = working.groupby("sector_risk_code")
    overall_approval_rate = float(working["approved"].mean()) * 100

    lines = ["# Stage 1 fairness sanity check by sector (P3.9)", ""]
    lines.append(f"Portfolio-wide approval rate: **{overall_approval_rate:.2f}%**")
    lines.append("")
    lines.append(
        "| Sector | Rows | Actual default rate | Approval rate | Avg predicted risk | Flag |"
    )
    lines.append("|---|---|---|---|---|---|")
    for sector, group in by_sector:
        default_rate = group["actual_default"].mean() * 100
        approval_rate = group["approved"].mean() * 100
        avg_risk = group["predicted_risk"].mean() * 100
        gap = abs(approval_rate - overall_approval_rate)
        flag = "check" if gap > FLAG_THRESHOLD_PCT else ""
        lines.append(
            f"| {sector} | {len(group):,} | {default_rate:.2f}% | "
            f"{approval_rate:.2f}% | {avg_risk:.2f}% | {flag} |"
        )
    lines.append("")
    lines.append(
        f"Flag threshold: approval rate more than {FLAG_THRESHOLD_PCT:.0f} percentage "
        "points from the portfolio average. A flag is a prompt to explain the gap "
        "(sectors carry genuinely different risk), not proof of an unfair model."
    )
    return "\n".join(lines) + "\n"


def main(
    csv_path: Path = DEFAULT_CSV_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    frame = pd.read_csv(csv_path)
    bundle = CreditSenseBundle.load(bundle_path)
    report_path.write_text(build_report(frame, bundle), encoding="utf-8")
    print(f"Wrote fairness-by-sector report to {report_path}")


if __name__ == "__main__":
    main()
