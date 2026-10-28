# Zeckathon PS 03 — Liquid Fund Flow Forecasting & Cash-Buffer Policy

> **Research prototype.** This repository is not production-validated. Source reconciliation, economic interpretation of the cash-buffer proxy, and forecast stability remain limitations.

## Objective

Forecast monthly net flows for India's Liquid and Overnight mutual-fund categories, then evaluate a simple cash-buffer policy under an explicitly labelled market-rate spread proxy.

## Reproduce from a clean checkout

Tested environment is intended to be **Ubuntu + Python 3.11** through GitHub Actions. A successful run is not claimed until the workflow completes successfully on this branch.

```bash
git clone https://github.com/Adarsh4777/zeckathon-ps03-liquidity-forecasting.git
cd zeckathon-ps03-liquidity-forecasting
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/reproduce_submission.py
```

The script resolves all paths from its own repository location, not the current working directory. Its sole modelling input is:

- `data/processed/ps03_amfi_rate_merged_full_calendar_2019_2026.csv`

It intentionally does not fetch external data at runtime. Input observations and source URLs are committed for traceability.

## Canonical outputs

The canonical pipeline regenerates the following under `results/`:

- `ps03_data_quality_audit_canonical.csv` — row counts, missing numeric cells, source URLs/statuses, duplicates, and flow-identity diagnostics.
- `ps03_missing_category_months_canonical.csv` — explicit list of missing/unverified category-month cells.
- `ps03_rolling_backtest_predictions_canonical.csv` — per-month predictions and forecast cut-off.
- `ps03_model_metrics_canonical.csv` — candidate model metrics on a common validation/test date set.
- `ps03_selected_models_canonical.csv` and `ps03_selected_test_metrics_canonical.csv` — models selected by validation MAE and their test results.
- `ps03_cash_buffer_policy_backtest_real_spread.csv` and `ps03_cash_buffer_policy_summary_real_spread.csv` — observed-monthly-spread cash-buffer evaluation, not the July 2026 snapshot sensitivity.
- `ps03_forecast_liquid_canonical.png`, `ps03_forecast_overnight_canonical.png`, and `research_workbook.xlsx` — figures and workbook rebuilt from those outputs.

GitHub Actions runs the same command on pushes, pull requests, and manual dispatch, then uploads generated results as a workflow artifact. The artifact is the run output; it is not automatically committed back to the repository. Do not describe results as reproduced until the workflow is green and the generated artifact has been reviewed.

## Evaluation protocol

- Validation window: March 2023–September 2024.
- Test window: October 2024–September 2026.
- Five candidates: Seasonal Naive, ETS with additive trend, AR(1) SARIMAX, Ridge with lagged features, and Ridge with lagged features plus the lagged rate-spread feature.
- For each category and window, metrics are computed over the same dates where actuals and predictions from every candidate are available. Selection is by validation MAE only.
- Flow predictors use observations through target month `t-2`; the spread feature is lagged to target month `t-3` or earlier.
- The cash-buffer evaluation uses observed monthly T-bill SGL yield minus WACR. It does **not** use the July 2026 snapshot as if it were a historical series. The target-month observed spread is used only for ex-post scoring, not for the buffer decision.

## Data quality and limitations

- Calendar span: April 2019–September 2026; 90 months × 2 categories = 180 possible category-month cells. Four cells remain missing/unverified: Liquid and Overnight for May 2021 and July 2021. The pipeline writes these cells explicitly and fails if the count changes unexpectedly.
- Two cumulative-table residual reconstruction rows have small known flow-identity differences (December 2019 Liquid: ₹0.03 crore; January 2021 Overnight: ₹0.04 crore). The audit reports these documented exceptions; discrepancies above ₹0.05 crore fail the run.
- Older observations were transcribed from browser-readable source tables rather than downloaded raw official files. Source URLs and transcription notes are stored with the data; this is traceability, not independent source verification.
- WACR has observations through September 2026, while the T-bill SGL series ends in June 2026. No missing T-bill rates are forward-filled; the cash-buffer score therefore ends in June 2026.
- `max(-net_flow, 0)` is only a monthly net-outflow proxy. It is not intramonth liquidity demand and can be small or zero even when gross redemptions are large. Gross redemptions are reported separately as a stress indicator.
- The rate spread is a market proxy, not a fund-specific realized cash-versus-investment yield differential. Cash-buffer costs depend on an illustrative shortfall penalty and should not be interpreted as an operational optimum.
- Prediction intervals are not claimed to be calibrated or production-ready. No final three-month recursive forecast is published.
- Metrics in older, non-canonical result files may reflect previous analysis runs. Use the files explicitly listed under **Canonical outputs** after a successful workflow run; do not mix old tables with canonical outputs.

## Source references

- AMFI monthly reports: https://www.amfiindia.com/research-information/amfi-monthly
- SEBI cumulative mutual-fund tables used for flagged residual reconstructions: https://www.sebi.gov.in/statistics/mutual-fund/objective/apr-2019-dec-2019.html and https://www.sebi.gov.in/statistics/mutual-fund/objective/apr-jan-2021.html
- Monthly WACR series (RBI DBIE series republished): https://measure.thisindianlife.today/data/rates-call-monthly.json
- Monthly 15–91-day T-bill SGL yield series (RBI DBIE series republished): https://measure.thisindianlife.today/data/rates-tbill-sgl.json
