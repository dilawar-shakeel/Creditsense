# CreditSense Synthetic Portfolio — Data Profile (P1.6)

Row count: **50,000** | Sectors: **11** | Overall default rate: **10.18%**

## Default rate by sector

| Sector | Rows | Default rate |
|---|---|---|
| Steel/Re-rolling | 2,482 | 11.48% |
| Food Processing | 4,029 | 11.17% |
| Other Services | 2,532 | 10.82% |
| Textile/Garments | 9,118 | 10.67% |
| Construction/Bldg Materials | 5,069 | 10.44% |
| Rice/Agri-processing | 2,499 | 10.40% |
| Auto Parts/Engineering | 4,870 | 10.12% |
| Leather/Sports Goods | 3,043 | 10.09% |
| Retail/Trade | 10,872 | 9.54% |
| Pharma/Surgical | 2,460 | 9.19% |
| IT/Tech Services | 3,026 | 8.39% |

## current_ratio — quartiles by sector

| Sector | p25 | p50 (median) | p75 |
|---|---|---|---|
| Auto Parts/Engineering | 0.872 | 1.182 | 1.593 |
| Construction/Bldg Materials | 0.866 | 1.160 | 1.569 |
| Food Processing | 0.856 | 1.149 | 1.568 |
| IT/Tech Services | 0.860 | 1.165 | 1.595 |
| Leather/Sports Goods | 0.865 | 1.172 | 1.582 |
| Other Services | 0.847 | 1.153 | 1.581 |
| Pharma/Surgical | 0.861 | 1.172 | 1.574 |
| Retail/Trade | 0.849 | 1.168 | 1.587 |
| Rice/Agri-processing | 0.847 | 1.156 | 1.579 |
| Steel/Re-rolling | 0.854 | 1.149 | 1.592 |
| Textile/Garments | 0.862 | 1.169 | 1.581 |

## debt_to_equity_ratio — quartiles by sector

| Sector | p25 | p50 (median) | p75 |
|---|---|---|---|
| Auto Parts/Engineering | 1.044 | 1.656 | 2.658 |
| Construction/Bldg Materials | 1.036 | 1.662 | 2.645 |
| Food Processing | 1.030 | 1.619 | 2.597 |
| IT/Tech Services | 1.044 | 1.668 | 2.674 |
| Leather/Sports Goods | 1.061 | 1.673 | 2.661 |
| Other Services | 1.018 | 1.648 | 2.570 |
| Pharma/Surgical | 1.012 | 1.653 | 2.634 |
| Retail/Trade | 1.025 | 1.670 | 2.660 |
| Rice/Agri-processing | 1.030 | 1.679 | 2.803 |
| Steel/Re-rolling | 1.058 | 1.679 | 2.644 |
| Textile/Garments | 1.046 | 1.695 | 2.667 |

## collateral_coverage_ratio — quartiles by sector

| Sector | p25 | p50 (median) | p75 |
|---|---|---|---|
| Auto Parts/Engineering | 0.609 | 0.938 | 1.354 |
| Construction/Bldg Materials | 0.605 | 0.966 | 1.381 |
| Food Processing | 0.608 | 0.974 | 1.363 |
| IT/Tech Services | 0.596 | 0.948 | 1.357 |
| Leather/Sports Goods | 0.629 | 0.974 | 1.378 |
| Other Services | 0.590 | 0.951 | 1.366 |
| Pharma/Surgical | 0.600 | 0.962 | 1.360 |
| Retail/Trade | 0.612 | 0.970 | 1.358 |
| Rice/Agri-processing | 0.618 | 0.975 | 1.351 |
| Steel/Re-rolling | 0.623 | 0.953 | 1.331 |
| Textile/Garments | 0.605 | 0.961 | 1.365 |

## bank_statement_volatility — quartiles by sector

| Sector | p25 | p50 (median) | p75 |
|---|---|---|---|
| Auto Parts/Engineering | 0.403 | 0.610 | 0.896 |
| Construction/Bldg Materials | 0.403 | 0.609 | 0.904 |
| Food Processing | 0.404 | 0.610 | 0.926 |
| IT/Tech Services | 0.407 | 0.611 | 0.919 |
| Leather/Sports Goods | 0.400 | 0.593 | 0.889 |
| Other Services | 0.409 | 0.620 | 0.921 |
| Pharma/Surgical | 0.407 | 0.593 | 0.902 |
| Retail/Trade | 0.402 | 0.602 | 0.908 |
| Rice/Agri-processing | 0.407 | 0.617 | 0.913 |
| Steel/Re-rolling | 0.420 | 0.622 | 0.925 |
| Textile/Garments | 0.405 | 0.607 | 0.902 |

## revenue_growth_yoy_pct — quartiles by sector

| Sector | p25 | p50 (median) | p75 |
|---|---|---|---|
| Auto Parts/Engineering | -3.561 | 7.929 | 19.837 |
| Construction/Bldg Materials | -3.668 | 8.476 | 20.216 |
| Food Processing | -4.849 | 8.024 | 19.972 |
| IT/Tech Services | -4.347 | 7.854 | 20.143 |
| Leather/Sports Goods | -4.737 | 8.001 | 19.899 |
| Other Services | -4.166 | 7.911 | 19.740 |
| Pharma/Surgical | -4.269 | 8.558 | 20.039 |
| Retail/Trade | -4.678 | 7.877 | 19.849 |
| Rice/Agri-processing | -4.461 | 8.141 | 19.705 |
| Steel/Re-rolling | -4.005 | 8.270 | 19.809 |
| Textile/Garments | -4.637 | 7.794 | 19.994 |

Full statistical validation (74 tests: distribution shape, sector conditioning, monotonic constraints, leakage warnings) lives in `tests/synthetic_data/` — run with `pytest tests/synthetic_data -v`.
