# Zeckathon PS 03 — Liquid Fund Flow Forecasting & Cash-Buffer Policy

> **Status: research prototype — source reconciliation and final forecast validation remain open.** Do not present this repository as fully production-validated.

## Objective
Forecast monthly net flows for India's Liquid and Overnight mutual fund categories and evaluate a cash-buffer policy that balances idle-cash yield drag against redemption shortfall risk.

## Repository layout

- `src/` — modelling/rebuild script.
- `data/raw_transcribed/` — source-traceable observations transcribed from browser-readable source tables; not raw official downloads.
- `data/processed/` — month/category calendar and merged modelling panel.
- `results/` — validation/test metrics, cash-buffer sensitivity tables, workbook and charts.
- `docs/` — audit, methodology and limitations.

## Current methodology

- Fixed test window: October 2024–September 2026 (24 target months).
- Candidate models: Seasonal Naive, ETS with trend, AR(1) SARIMAX, Ridge with lagged features, and Ridge with lagged features plus rate/macro features.
- Forecast timing safeguard: AMFI flow features use observations through target month `t-2`; the spread feature is lagged more conservatively.
- Cash-buffer sensitivity is evaluated only for months with observed rate spreads. The liquidity-demand proxy is `max(-net_flow, 0)`; gross redemptions are treated as a separate stress measure.

## Key current findings

The current selected models are Seasonal Naive for Liquid and RidgeSpreadMacro for Overnight based on the common pre-test validation period. On the fixed test window, Liquid Seasonal Naive MAE is about ₹33,199 crore and Overnight RidgeSpreadMacro MAE is about ₹14,325 crore. These are initial results, not a guarantee of future accuracy.

## Important limitations

1. The AMFI calendar spans April 2019–September 2026, but four category-month cells remain missing/unverified (both categories in May and July 2021).
2. Some older observations are transcribed from browser-readable tables, and two are cumulative-table residual reconstructions. They have not all been reconciled against downloaded raw official AMFI files.
3. The T-bill-minus-WACR spread is a market-rate proxy, not a fund-specific realized cash-versus-investment yield differential. Joint observed spread coverage currently ends in June 2026.
4. Prediction intervals are not demonstrated to be calibrated: Overnight coverage was 75% for both nominal 80% and 95% intervals on a 24-month test.
5. A final three-month forecast is not included because recursive future feature generation has not been fully implemented and validated.
6. The included script is a research rebuild script and should be reviewed before being treated as a clean one-command reproduction pipeline.

## Data sources

- AMFI monthly reports: https://www.amfiindia.com/research-information/amfi-monthly
- SEBI cumulative mutual-fund tables used for explicitly flagged residual reconstructions: https://www.sebi.gov.in/statistics/mutual-fund/objective/apr-2019-dec-2019.html and https://www.sebi.gov.in/statistics/mutual-fund/objective/apr-jan-2021.html
- WACR monthly series (RBI DBIE series republished): https://measure.thisindianlife.today/data/rates-call-monthly.json
- 15–91-day T-bill SGL yield series (RBI DBIE series republished): https://measure.thisindianlife.today/data/rates-tbill-sgl.json

## Before final submission

- Reconcile transcribed/derived values against raw source files.
- Fix or document the four missing cells.
- Review the script for reproducibility in a clean environment.
- Recompute all metrics from the final data and code.
- Add the final presentation/report only after results are frozen.
