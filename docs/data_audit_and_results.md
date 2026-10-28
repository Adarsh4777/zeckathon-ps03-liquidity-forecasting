# PS03 Data Extension & Historical Yield-Spread Update

## What was actually added

- Extended older AMFI panel rows combine direct values from browser-readable AMFI Monthly Composite Report tables, August 2020 values from a readable mirror of that report, and two clearly labelled cumulative-table residual reconstructions (December 2019 and January 2021) supported by SEBI's official cumulative tables. The residual rows use cumulative mobilisations/redemptions/net flows less observed month rows; closing and average AUM are taken from the respective SEBI table.
- The older source subset is still sparse. No unavailable May/July 2021 observations are filled, interpolated or treated as real values.
- The original continuous AMFI table from October 2021–September 2026 is retained.
- A historical market-rate series has been transcribed from browser-readable JSON data pages published by Measure of India, which identify RBI DBIE series: monthly average weighted-average call/notice-money rate (WACR), and monthly weighted SGL trade yield on Treasury bills with 15–91 days residual maturity.
- Rates are source observations, not a fund-specific realized yield spread. The calculated spread is `15–91 day T-bill SGL yield − WACR`, in percentage points.

## Coverage / gaps

- Calendar span: 2019-04 to 2026-09 (90 months × 2 categories = 180 category-month cells).
- Actual category rows present: 176 of 180.
- Rows still missing/unverified: 4.
- Months with at least one missing category row: 2.
- Missing/unverified month labels: 2021-05, 2021-07.
- This is an **extended but not fully continuous** archive before October 2021. Missing older AMFI source rows remain blank in `ps03_amfi_full_calendar_2019_2026.csv`.
- Direct monthly report rows reconcile within about ₹0.01 crore. Two cumulative-derived rows do not reconcile exactly to mobilisations minus redemptions: December 2019 Liquid differs by ₹0.03 crore and January 2021 Overnight by ₹0.04 crore. These residual differences are retained and flagged rather than silently forced to zero.

## Historical rate spread coverage

- WACR observations: 60 months, October 2021–September 2026.
- 15–91-day T-bill SGL yield observations: 57 months, October 2021–June 2026.
- Joint spread observations: 57 months, October 2021–June 2026.
- The T-bill source has no observation after June 2026 in the fetched/rendered JSON; July–September are left missing, never forward-filled.
- Historical rates are included in a new CSV and merged to the full calendar. For forecast features for target month `t`, the spread feature uses the latest joint spread observation no later than `t-3`, a conservative publication-lag convention. The actual spread in target month is used only ex post to score historical cash-buffer cost.
- The local runtime cannot resolve external source hosts; the values in `ps03_historical_rates_spread_2021_2026.csv` were manually transcribed from the browser-rendered JSON output, with source URLs preserved. So the files are source-traceable but not raw-file downloads.

## Model protocol

- Fixed test window retained: October 2024–September 2026 (24 target months).
- Model selection uses common validation March 2023–September 2024 only, so all compared candidate models are evaluated on the same 19 target months.
- Flow features use AMFI rows through target `t-2`; yield-spread feature uses a jointly observed monthly spread no later than `t-3`.
- Older missing AMFI reports are not imputed. Seasonal-naive uses the prior-year same month only when observed; ETS is fitted to the recent continuous segment; SARIMAX receives missing endogenous months as missing observations; Ridge uses rows with complete required observed feature lags only.
- Validation residuals from the same common March 2023–September 2024 window are used to construct fixed test intervals. The test outcomes are not used to calibrate their own intervals.

## Test model results (models selected from validation)

 category selected_model_by_pretest_MAE  n_test_predictions test_start test_end    MAE_cr   RMSE_cr  WAPE_pct  PI80_coverage_pct  PI80_n  PI95_coverage_pct  PI95_n                                                        interval_calibration_source
   Liquid                 SeasonalNaive                  24    2024-10  2026-09 33,198.87 40,501.14     52.01              83.33      24             100.00      24 Common validation period Mar2023-Sep2024 residuals only; fixed before Oct2024 test
Overnight              RidgeSpreadMacro                  24    2024-10  2026-09 14,324.99 18,084.12     75.07              75.00      24              75.00      24 Common validation period Mar2023-Sep2024 residuals only; fixed before Oct2024 test

## Full metrics by validation/test window

 category            model     window  n   start     end    MAE_cr   RMSE_cr  WAPE_pct  sMAPE_pct  filtered_MAPE_abs_actual_ge_5000_pct
   Liquid    SeasonalNaive       test 24 2024-10 2026-09 33,198.87 40,501.14     52.01      82.76                                 74.86
   Liquid RidgeSpreadMacro       test 24 2024-10 2026-09 42,438.90 48,609.32     66.49     103.89                                 88.93
   Liquid     RidgeLagOnly       test 24 2024-10 2026-09 44,910.06 52,656.22     70.36      97.84                                 87.86
   Liquid     ETS_AddTrend       test 24 2024-10 2026-09 63,646.58 77,941.20     99.71     177.28                                 99.64
   Liquid      SARIMAX_AR1       test 24 2024-10 2026-09 69,321.29 84,169.41    108.61     168.73                                104.60
Overnight RidgeSpreadMacro       test 24 2024-10 2026-09 14,324.99 18,084.12     75.07     116.65                                 70.63
Overnight    SeasonalNaive       test 24 2024-10 2026-09 15,804.12 19,157.35     82.82     119.48                                 78.97
Overnight     RidgeLagOnly       test 24 2024-10 2026-09 16,111.28 20,392.31     84.43     120.81                                 78.07
Overnight     ETS_AddTrend       test 24 2024-10 2026-09 19,345.27 23,704.60    101.38     176.06                                 98.40
Overnight      SARIMAX_AR1       test 24 2024-10 2026-09 20,281.31 24,979.41    106.29     174.87                                108.95
   Liquid    SeasonalNaive validation 24 2022-10 2024-09 36,001.02 44,684.36     74.48     109.22                                105.37
   Liquid RidgeSpreadMacro validation 23 2022-11 2024-09 38,351.18 46,234.23     77.31     128.59                                 94.02
   Liquid     RidgeLagOnly validation 24 2022-10 2024-09 38,942.61 49,823.71     80.56     112.57                                109.35
   Liquid      SARIMAX_AR1 validation 24 2022-10 2024-09 53,057.46 63,951.28    109.77     184.04                                114.25
   Liquid     ETS_AddTrend validation 19 2023-03 2024-09 58,097.55 67,749.60    102.54     183.41                                104.26
Overnight     ETS_AddTrend validation 19 2023-03 2024-09 10,150.92 12,368.36    103.12     138.09                                100.09
Overnight      SARIMAX_AR1 validation 24 2022-10 2024-09 10,200.33 14,430.00    104.46     161.03                                104.28
Overnight     RidgeLagOnly validation 24 2022-10 2024-09 11,747.54 16,902.12    120.31     143.93                                 98.34
Overnight RidgeSpreadMacro validation 23 2022-11 2024-09 11,884.25 20,151.03    120.50     140.54                                100.21
Overnight    SeasonalNaive validation 24 2022-10 2024-09 16,826.78 19,939.67    172.33     152.89                                130.21

## Cash-buffer historical cost: observed-spread subset

Backtest is restricted to target months where the actual monthly yield spread is observed (maximum through June 2026), and uses only prior observations to choose each buffer. Cost proxy:

`buffer × max(actual spread, 0) / 100 / 12 + 3 × max(actual net outflow proxy − buffer, 0)`

The outflow proxy is `max(-net_flow, 0)`. Gross redemptions are shown separately as stress demand, not substituted for net outflow. The historical spread is a market-rate proxy rather than the realized yield gap for a specific fund. This matters materially for interpreting the optimum.

 category                       policy  n_months_scored first_month last_month  mean_buffer_cr  mean_idle_drag_cr  mean_shortfall_cr  total_penalized_shortfall_cost_cr  total_cost_proxy_cr  mean_cost_proxy_cr                                                                                                            yield_spread_source                                                                                      outflow_measure
   Liquid         Last_year_same_month               21     2024-10    2026-06       29,431.28               0.93           5,374.72                         338,607.18           338,626.72           16,125.08 observed monthly T-bill SGL 15-91d minus WACR; no future spread used for decision; actual month spread only used to score cost max(-net flow, 0); gross redemptions shown as separate stress proxy, not substituted for net outflow
   Liquid Trailing_95th_percentile_24m               21     2024-10    2026-06      112,073.57               2.12           3,022.01                         190,386.70           190,431.27            9,068.16 observed monthly T-bill SGL 15-91d minus WACR; no future spread used for decision; actual month spread only used to score cost max(-net flow, 0); gross redemptions shown as separate stress proxy, not substituted for net outflow
   Liquid Trailing_99th_percentile_24m               21     2024-10    2026-06      146,245.09               2.59               0.00                               0.00                54.46                2.59 observed monthly T-bill SGL 15-91d minus WACR; no future spread used for decision; actual month spread only used to score cost max(-net flow, 0); gross redemptions shown as separate stress proxy, not substituted for net outflow
Overnight         Last_year_same_month               21     2024-10    2026-06        7,074.72               0.21           5,620.72                         354,105.12           354,109.58           16,862.36 observed monthly T-bill SGL 15-91d minus WACR; no future spread used for decision; actual month spread only used to score cost max(-net flow, 0); gross redemptions shown as separate stress proxy, not substituted for net outflow
Overnight Trailing_95th_percentile_24m               21     2024-10    2026-06       26,567.73               0.58           1,667.28                         105,038.67           105,050.88            5,002.42 observed monthly T-bill SGL 15-91d minus WACR; no future spread used for decision; actual month spread only used to score cost max(-net flow, 0); gross redemptions shown as separate stress proxy, not substituted for net outflow
Overnight Trailing_99th_percentile_24m               21     2024-10    2026-06       31,028.26               0.65             885.66                          55,796.64            55,810.30            2,657.63 observed monthly T-bill SGL 15-91d minus WACR; no future spread used for decision; actual month spread only used to score cost max(-net flow, 0); gross redemptions shown as separate stress proxy, not substituted for net outflow

## Source links

- AMFI official monthly reports: https://www.amfiindia.com/research-information/amfi-monthly
- Per-month AMFI PDF source URLs are recorded row-by-row. Cumulative-derived rows cite the relevant SEBI summary tables; August 2020 cites the readable AMFI-report mirror and also records the intended official AMFI URL in the note.
- Monthly WACR source JSON (RBI DBIE series as republished): https://measure.thisindianlife.today/data/rates-call-monthly.json
- Monthly Treasury bill SGL yield source JSON (RBI DBIE series as republished): https://measure.thisindianlife.today/data/rates-tbill-sgl.json

## Limitations to fix before submission

1. Retrieve the original raw AMFI archive files in a network-enabled environment and reconcile the manually transcribed/derived legacy subset with official Excel/PDF downloads. May and July 2021 monthly category rows remain unavailable here.
2. Add complete official macro series only with release dates and historical vintages where available; a month-average observation must not leak into a forecast issued before that month has closed and been published.
3. The fixed 24-month performance sample is small. Do not present 80%/95% interval coverage as calibrated merely because the sample coverage happens to be near target.
4. The T-bill-minus-WACR spread is a market proxy. A fund-specific cash-versus-instrument return spread or independently sourced historical returns would be stronger.
