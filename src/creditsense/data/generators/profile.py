"""
P1.6 — Distribution sanity check, human-readable version.

The 74 tests in tests/synthetic_data/ already assert per-sector realism numerically;
this just writes what they check into a short markdown report a person can read without
running pytest. Reuses nothing new — it is a summary view over the same CSV.

Run: python -m creditsense.data.generators.profile
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CSV_PATH = Path("src/creditsense/data/raw/creditsense_synthetic_portfolio.csv")
DEFAULT_REPORT_PATH = Path("src/creditsense/data/raw/data_profile.md")

# The five ratios most relevant to underwriting realism, checked per sector.
KEY_RATIOS = [
    "current_ratio",
    "debt_to_equity_ratio",
    "collateral_coverage_ratio",
    "bank_statement_volatility",
    "revenue_growth_yoy_pct",
]


def build_report(frame: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# CreditSense Synthetic Portfolio — Data Profile (P1.6)")
    lines.append("")
    lines.append(
        f"Row count: **{len(frame):,}** | Sectors: **{frame['sector_risk_code'].nunique()}** "
        f"| Overall default rate: **{frame['default_probability_12m'].mean() * 100:.2f}%**"
    )
    lines.append("")
    lines.append("## Default rate by sector")
    lines.append("")
    lines.append("| Sector | Rows | Default rate |")
    lines.append("|---|---|---|")
    by_sector = frame.groupby("sector_risk_code")
    default_rate = by_sector["default_probability_12m"].mean().sort_values(ascending=False)
    counts = by_sector.size()
    for sector, rate in default_rate.items():
        lines.append(f"| {sector} | {counts[sector]:,} | {rate * 100:.2f}% |")
    lines.append("")

    for ratio in KEY_RATIOS:
        lines.append(f"## {ratio} — quartiles by sector")
        lines.append("")
        lines.append("| Sector | p25 | p50 (median) | p75 |")
        lines.append("|---|---|---|---|")
        quartiles = by_sector[ratio].quantile([0.25, 0.5, 0.75]).unstack()
        for sector in quartiles.index:
            p25, p50, p75 = quartiles.loc[sector, [0.25, 0.5, 0.75]]
            lines.append(f"| {sector} | {p25:.3f} | {p50:.3f} | {p75:.3f} |")
        lines.append("")

    lines.append(
        "Full statistical validation (74 tests: distribution shape, sector "
        "conditioning, monotonic constraints, leakage warnings) lives in "
        "`tests/synthetic_data/` — run with `pytest tests/synthetic_data -v`."
    )
    return "\n".join(lines) + "\n"


def main(
    csv_path: Path = DEFAULT_CSV_PATH, report_path: Path = DEFAULT_REPORT_PATH
) -> None:
    frame = pd.read_csv(csv_path)
    report_path.write_text(build_report(frame), encoding="utf-8")
    print(f"Wrote data profile to {report_path}")


if __name__ == "__main__":
    main()
