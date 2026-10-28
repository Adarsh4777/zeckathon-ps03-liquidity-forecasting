#!/usr/bin/env python3
"""Reproducible PS03 analysis pipeline.

Run from any working directory:
    python src/reproduce_submission.py

Inputs are repository-relative. Outputs are rebuilt under results/ on every run.
All monthly forecast targets use information available through target month t-2.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TEST_START, TEST_END = "2024-10", "2026-09"
VAL_START, VAL_END = "2023-03", "2024-09"
CATEGORIES = ["Liquid", "Overnight"]
MODELS = ["SeasonalNaive", "ETS_AddTrend", "SARIMAX_AR1", "RidgeLagOnly", "RidgeSpreadMacro"]
FLOW_COLS = ["funds_mobilized_cr", "gross_redemptions_cr", "net_flow_cr",
             "closing_net_assets_cr", "average_net_assets_cr"]


def read_inputs():
    panel_path = DATA / "processed/ps03_amfi_rate_merged_full_calendar_2019_2026.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Required input not found: {panel_path}")
    d = pd.read_csv(panel_path, parse_dates=["month"])
    required = {"month", "category", *FLOW_COLS}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Processed panel lacks required columns: {sorted(missing)}")
    d["month"] = d["month"].dt.to_period("M").dt.to_timestamp()
    d = d[d.category.isin(CATEGORIES)].copy()
    if d.duplicated(["month", "category"]).any():
        dup = d.loc[d.duplicated(["month", "category"], keep=False), ["month", "category"]]
        raise ValueError(f"Duplicate month/category rows found:\n{dup.to_string(index=False)}")
    return d.sort_values(["category", "month"]).reset_index(drop=True)


def make_panel(d, category):
    g = d[d.category == category].set_index("month").sort_index()
    months = pd.date_range(g.index.min(), g.index.max(), freq="MS")
    g = g.reindex(months)
    g.index.name = "month"
    g["category"] = category
    if "yield_spread_tbill_minus_wacr_pct_points" not in g:
        g["yield_spread_tbill_minus_wacr_pct_points"] = np.nan
    return g


def feature_row(g, i, macro):
    # Target month i is forecast using AMFI flow observations no later than i-2.
    if i < 4:
        return None
    vals = []
    for lag in (2, 3, 4):
        v = g["net_flow_cr"].iloc[i-lag]
        a = g["average_net_assets_cr"].iloc[i-lag]
        if pd.isna(v) or pd.isna(a) or a <= 0:
            return None
        vals.append(100.0 * v / a)
    spread = g["yield_spread_tbill_minus_wacr_pct_points"].iloc[i-3]
    if macro and pd.isna(spread):
        return None
    month = g.index[i].month
    vals += [np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)]
    if macro:
        vals.append(float(spread))
    return np.asarray(vals, dtype=float)


def forecast_one(g, i, model):
    cutoff = i - 2
    y = g["net_flow_cr"].to_numpy(dtype=float)
    if cutoff < 0:
        return np.nan
    if model == "SeasonalNaive":
        return float(y[i-12]) if i >= 12 and np.isfinite(y[i-12]) else np.nan
    train_y = y[:cutoff+1]
    if model == "ETS_AddTrend":
        good = np.flatnonzero(np.isfinite(train_y))
        if len(good) < 18 or not np.all(np.diff(good[-min(len(good), 24):]) == 1):
            return np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(train_y[good[-min(len(good), 36):]],
                    trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
                return float(fit.forecast(2)[-1])
        except Exception:
            return np.nan
    if model == "SARIMAX_AR1":
        good = np.flatnonzero(np.isfinite(train_y))
        if len(good) < 18 or not np.all(np.diff(good[-min(len(good), 24):]) == 1):
            return np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(train_y[good[-min(len(good), 36):]], order=(1,0,0),
                              trend="c", enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False, maxiter=100)
                return float(fit.forecast(2)[-1])
        except Exception:
            return np.nan
    macro = model == "RidgeSpreadMacro"
    X, Y = [], []
    # Only labels observed by the forecast origin (through target i-2) enter training.
    for j in range(4, cutoff+1):
        f = feature_row(g, j, macro)
        target = y[j]
        a = g["average_net_assets_cr"].iloc[j-2]
        if f is not None and np.isfinite(target) and np.isfinite(a) and a > 0:
            X.append(f)
            Y.append(100.0 * target / a)
    f = feature_row(g, i, macro)
    if f is None or len(Y) < 12:
        return np.nan
    try:
        model_fit = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        model_fit.fit(np.asarray(X), np.asarray(Y))
        denom = g["average_net_assets_cr"].iloc[i-2]
        if not np.isfinite(denom) or denom <= 0:
            return np.nan
        return float(model_fit.predict(f.reshape(1, -1))[0] * denom / 100.0)
    except Exception:
        return np.nan


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[ok], pred[ok]
    if not len(a):
        return {"n": 0, "MAE_cr": np.nan, "RMSE_cr": np.nan, "WAPE_pct": np.nan, "sMAPE_pct": np.nan}
    e = a-p
    denom = np.sum(np.abs(a))
    smden = np.abs(a)+np.abs(p)
    sm = np.where(smden > 0, 200*np.abs(e)/smden, 0)
    return {"n": len(a), "MAE_cr": float(np.mean(np.abs(e))),
            "RMSE_cr": float(np.sqrt(np.mean(e**2))),
            "WAPE_pct": float(100*np.sum(np.abs(e))/denom) if denom else np.nan,
            "sMAPE_pct": float(np.mean(sm))}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = read_inputs()
    audit = []
    for cat in CATEGORIES:
        raw = d[d.category == cat]
        expected = pd.date_range(d.month.min(), d.month.max(), freq="MS")
        present = set(raw.month)
        numeric_missing = int(raw[FLOW_COLS].isna().sum().sum())
        identity = raw["funds_mobilized_cr"] - raw["gross_redemptions_cr"] - raw["net_flow_cr"]
        source_missing = int(raw["source_url"].isna().sum() + raw["source_url"].astype(str).str.strip().eq("").sum()) if "source_url" in raw else len(raw)
        status_counts = raw["observation_status"].fillna("MISSING_STATUS").value_counts().to_dict() if "observation_status" in raw else {}
        duplicate_sources = int(raw.duplicated(["month", "category", "source_url"]).sum()) if "source_url" in raw else len(raw)
        audit.append({"category": cat, "expected_months": len(expected),
                      "observed_rows": len(raw), "missing_months": len(set(expected)-present),
                      "duplicate_month_category": int(raw.duplicated(["month","category"]).sum()),
                      "missing_numeric_cells": numeric_missing, "rows_missing_source_url": source_missing,
                      "duplicate_month_category_source_rows": duplicate_sources,
                      "observation_status_counts": str(status_counts),
                      "max_abs_identity_diff_cr": float(identity.abs().max()),
                      "identity_rows_over_0_02_cr": int((identity.abs()>0.02).sum())})
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(RESULTS / "ps03_data_quality_audit_canonical.csv", index=False)
    all_months = pd.date_range(d.month.min(), d.month.max(), freq="MS")
    expected_keys = {(m, cat) for m in all_months for cat in CATEGORIES}
    observed_keys = set(zip(d.month, d.category))
    missing_keys = sorted(expected_keys - observed_keys)
    pd.DataFrame([{"month": m.strftime("%Y-%m"), "category": cat,
                   "status": "missing_or_unverified"} for m, cat in missing_keys]).to_csv(
                       RESULTS / "ps03_missing_category_months_canonical.csv", index=False)
    if len(d) != len(observed_keys):
        raise ValueError("Duplicate month/category keys detected after input validation.")
    if len(missing_keys) != 4:
        raise ValueError(f"Expected 4 known missing/unverified category-month cells; found {len(missing_keys)}.")
    if audit_df["max_abs_identity_diff_cr"].max() > 0.05:
        raise ValueError("Net-flow identity discrepancy exceeds ₹0.05 crore; inspect the data before submission.")

    preds = []
    panels = {}
    for cat in CATEGORIES:
        g = make_panel(d, cat)
        panels[cat] = g
        for i, month in enumerate(g.index):
            if month.strftime("%Y-%m") < VAL_START:
                continue
            actual = g["net_flow_cr"].iloc[i]
            if not np.isfinite(actual):
                continue
            for model in MODELS:
                p = forecast_one(g, i, model)
                preds.append({"month": month.strftime("%Y-%m"), "category": cat,
                              "model": model, "actual_net_flow_cr": actual,
                              "forecast_net_flow_cr": p, "error_cr": actual-p if np.isfinite(p) else np.nan,
                              "abs_error_cr": abs(actual-p) if np.isfinite(p) else np.nan,
                              "flow_cutoff_month": g.index[i-2].strftime("%Y-%m") if i >= 2 else ""})
    pred = pd.DataFrame(preds)
    pred.to_csv(RESULTS / "ps03_rolling_backtest_predictions_canonical.csv", index=False, float_format="%.4f")

    # A single common date set per category: dates with actuals and finite predictions for every model.
    rows, selected, selected_test = [], [], []
    for cat in CATEGORIES:
        for window, start, end in [("validation", VAL_START, VAL_END), ("test", TEST_START, TEST_END)]:
            z = pred[(pred.category == cat) & pred.month.between(start, end)]
            pivot = z.pivot(index="month", columns="model", values="forecast_net_flow_cr")
            actuals = z.drop_duplicates("month").set_index("month")["actual_net_flow_cr"]
            common = pivot.dropna(subset=MODELS, how="any").index.intersection(actuals.dropna().index)
            for model in MODELS:
                q = metrics(actuals.loc[common], pivot.loc[common, model]) if len(common) else metrics([], [])
                rows.append({"category": cat, "model": model, "window": window,
                             "n_common": len(common), "start": min(common) if len(common) else "",
                             "end": max(common) if len(common) else "", **q})
    metric = pd.DataFrame(rows)
    metric.to_csv(RESULTS / "ps03_model_metrics_canonical.csv", index=False, float_format="%.4f")
    metric.rename(columns={"n_common": "n"}).to_csv(
        RESULTS / "ps03_expanded_model_metrics.csv", index=False, float_format="%.4f")
    common_val = metric[metric.window == "validation"].rename(columns={
        "n_common": "n_validation_common", "start": "validation_common_start",
        "end": "validation_common_end"})
    common_val.to_csv(RESULTS / "ps03_model_selection_common_validation.csv",
                      index=False, float_format="%.4f")
    val = metric[metric.window == "validation"]
    for cat in CATEGORIES:
        vc = val[val.category == cat].sort_values(["MAE_cr", "model"])
        chosen = vc.iloc[0]["model"] if len(vc) else ""
        tm = metric[(metric.category == cat) & (metric.window == "test") & (metric.model == chosen)]
        selected.append({"category": cat, "selected_model_by_common_validation_MAE": chosen})
        if len(tm):
            selected_test.append({"category": cat, "selected_model_by_common_validation_MAE": chosen,
                                  **tm.iloc[0].to_dict()})
    pd.DataFrame(selected).to_csv(RESULTS / "ps03_selected_models_canonical.csv", index=False)
    pd.DataFrame(selected_test).to_csv(RESULTS / "ps03_selected_test_metrics_canonical.csv", index=False, float_format="%.4f")
    selected_test_df = pd.DataFrame(selected_test)
    if len(selected_test_df):
        selected_test_df = selected_test_df.rename(columns={
            "selected_model_by_common_validation_MAE": "selected_model_by_pretest_MAE",
            "n_common": "n_test_predictions", "start": "test_start", "end": "test_end"})
        selected_test_df.to_csv(RESULTS / "ps03_expanded_selected_model_test_metrics.csv",
                                index=False, float_format="%.4f")

    # Ex-post cost scoring uses only months with observed spread. Decisions use data through t-2.
    spread_col = "yield_spread_tbill_minus_wacr_pct_points"
    buffer_rows = []
    for cat, g in panels.items():
        y = g["net_flow_cr"].to_numpy(dtype=float)
        spr = g[spread_col].to_numpy(dtype=float)
        for i in range(36, len(g)):
            month = g.index[i].strftime("%Y-%m")
            if not (TEST_START <= month <= TEST_END) or not np.isfinite(spr[i]) or not np.isfinite(y[i]):
                continue
            known = np.maximum(-y[:i-1], 0.0)  # through i-2 only
            demand = max(-y[i], 0.0)
            q95 = float(np.quantile(known[np.isfinite(known)], .95, method="higher")) if np.isfinite(known).any() else 0.0
            q99 = float(np.quantile(known[np.isfinite(known)], .99, method="higher")) if np.isfinite(known).any() else 0.0
            policies = {"SameMonthLastYear": max(-y[i-12], 0.0) if i >= 12 and np.isfinite(y[i-12]) else 0.0,
                        "TrailingQ95": q95, "TrailingQ99": q99}
            for policy, buf in policies.items():
                drag = max(float(spr[i]), 0.0) / 100.0 * buf * 30/365
                short = max(demand-buf, 0.0)
                buffer_rows.append({"month": month, "category": cat, "policy": policy,
                    "buffer_cr": buf, "actual_net_outflow_proxy_cr": demand,
                    "actual_gross_redemptions_cr": g["gross_redemptions_cr"].iloc[i],
                    "observed_spread_pct_points": spr[i], "idle_yield_drag_cr": drag,
                    "shortfall_cr": short, "penalized_shortfall_cost_cr": 3*short,
                    "total_cost_proxy_cr": drag+3*short,
                    "decision_cutoff": g.index[i-2].strftime("%Y-%m"),
                    "scoring_note": "Observed target-month spread used ex post only; spread is a market proxy."})
    buf = pd.DataFrame(buffer_rows)
    buf.to_csv(RESULTS / "ps03_cash_buffer_policy_backtest_real_spread.csv", index=False, float_format="%.4f")
    if len(buf):
        summ = buf.groupby(["category","policy"], as_index=False).agg(
            scored_months=("month","nunique"), first_month=("month","min"), last_month=("month","max"),
            mean_buffer_cr=("buffer_cr","mean"), total_yield_drag_cr=("idle_yield_drag_cr","sum"),
            total_shortfall_cr=("shortfall_cr","sum"), shortfall_months=("shortfall_cr",lambda s:int((s>0).sum())),
            total_cost_proxy_cr=("total_cost_proxy_cr","sum"))
        summ.to_csv(RESULTS / "ps03_cash_buffer_policy_summary_real_spread.csv", index=False, float_format="%.4f")

    # Rebuild figures from canonical outputs.
    for cat in CATEGORIES:
        z = pred[(pred.category == cat) & pred.month.between(TEST_START, TEST_END)]
        plt.figure(figsize=(11,5))
        for model in ["SeasonalNaive", "RidgeLagOnly", "RidgeSpreadMacro"]:
            q = z[z.model == model].dropna(subset=["forecast_net_flow_cr"])
            if len(q):
                plt.plot(q.month, q.forecast_net_flow_cr, label=model)
        actual = z.drop_duplicates("month").dropna(subset=["actual_net_flow_cr"])
        plt.plot(actual.month, actual.actual_net_flow_cr, label="Actual", linewidth=2)
        plt.xticks(rotation=45, ha="right"); plt.ylabel("Net flow (INR crore)")
        plt.title(f"{cat}: actual vs rolling-origin forecasts"); plt.legend()
        plt.tight_layout(); plt.savefig(RESULTS / f"ps03_forecast_{cat.lower()}_canonical.png", dpi=160); plt.close()

    with pd.ExcelWriter(RESULTS / "research_workbook.xlsx", engine="openpyxl") as writer:
        audit_df.to_excel(writer, sheet_name="Data Audit", index=False)
        pred.to_excel(writer, sheet_name="Rolling Predictions", index=False)
        metric.to_excel(writer, sheet_name="Model Metrics", index=False)
        pd.DataFrame(selected).to_excel(writer, sheet_name="Selected Models", index=False)
        buf.to_excel(writer, sheet_name="Cash Buffer Backtest", index=False)
        if len(buf):
            summ.to_excel(writer, sheet_name="Cash Buffer Summary", index=False)

    print("Canonical PS03 pipeline completed.")
    print(f"Input: {DATA / 'processed/ps03_amfi_rate_merged_full_calendar_2019_2026.csv'}")
    print(f"Outputs: {RESULTS}")
    print(f"Model validation window: {VAL_START} to {VAL_END}; test window: {TEST_START} to {TEST_END}")
    print("Review ps03_data_quality_audit_canonical.csv, ps03_missing_category_months_canonical.csv, and ps03_selected_test_metrics_canonical.csv before submission.")


if __name__ == "__main__":
    main()
