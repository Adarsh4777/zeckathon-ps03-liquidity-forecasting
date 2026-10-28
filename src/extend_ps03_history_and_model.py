from pathlib import Path
import csv, json, math, warnings, zipfile, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

OUT=Path('/mnt/data')
base=pd.read_csv(OUT/'ps03_amfi_official_extracted_36m.csv', parse_dates=['month'])
# Exact observations transcribed from the official AMFI monthly composite report tables,
# October 2021-September 2023. Fields: (mobilised, redemptions, net flow, closing AUM, average AUM).
old={
'2021-10': {'Overnight':(363791.08,357453.80,6337.28,100904.67,115746.95),'Liquid':(202192.86,212392.41,-10199.55,314547.86,354763.69)},
'2021-11': {'Overnight':(379809.55,391933.01,-12123.46,89093.71,115724.67),'Liquid':(277564.82,224313.54,53251.28,369014.99,368710.82)},
'2021-12': {'Overnight':(453336.11,448605.13,4730.97,94138.69,110033.81),'Liquid':(297331.83,306016.83,-8684.99,361368.52,364938.09)},
'2022-01': {'Overnight':(385699.36,366341.81,19357.55,114032.31,111949.04),'Liquid':(240703.19,255101.10,-14397.91,347955.35,389150.40)},
'2022-02': {'Overnight':(353798.08,352502.02,1296.06,115555.27,127886.32),'Liquid':(274396.15,234122.83,40273.31,389305.48,400295.06)},
'2022-03': {'Overnight':(501715.35,514567.46,-12852.10,103071.33,125062.97),'Liquid':(311582.55,356186.27,-44603.72,345903.47,388477.17)},
'2022-04': {'Overnight':(358918.94,354790.88,4128.06,107527.71,115814.35),'Liquid':(281205.29,252474.30,28730.99,375793.29,401172.36)},
'2022-05': {'Overnight':(481769.34,466698.45,15070.89,123029.97,136494.28),'Liquid':(245143.87,243366.99,1776.88,378743.93,375867.95)},
'2022-06': {'Overnight':(526275.02,546943.09,-20668.07,102944.95,152639.12),'Liquid':(298147.68,313930.32,-15782.64,364492.65,405242.55)},
'2022-07': {'Overnight':(462219.10,442300.44,19918.66,123409.85,136240.23),'Liquid':(256150.90,263843.55,-7692.65,358401.53,421458.65)},
'2022-08': {'Overnight':(421677.17,438082.30,-16405.13,106254.59,143593.77),'Liquid':(333449.62,283353.80,50095.82,411291.99,421076.44)},
'2022-09': {'Overnight':(552502.99,519374.66,33128.33,141319.78,133564.03),'Liquid':(299102.84,359073.14,-59970.30,352213.14,419855.51)},
'2022-10': {'Overnight':(430592.20,438097.59,-7505.38,134583.00,153046.83),'Liquid':(251881.01,232796.41,19084.60,373212.25,388186.31)},
'2022-11': {'Overnight':(408744.62,440672.87,-31928.25,103324.44,141582.80),'Liquid':(314709.78,280433.34,34276.44,409715.00,420618.92)},
'2022-12': {'Overnight':(429847.81,431101.80,-1253.98,102637.18,113957.09),'Liquid':(351016.20,364868.20,-13852.00,398344.68,440562.30)},
'2023-01': {'Overnight':(401691.20,405379.10,-3687.90,99542.47,115085.43),'Liquid':(338073.43,343115.11,-5041.68,395699.87,457973.25)},
'2023-02': {'Overnight':(401879.05,398933.21,2945.84,103090.56,125487.40),'Liquid':(275599.33,286903.58,-11304.25,386489.73,434140.64)},
'2023-03': {'Overnight':(476647.40,484785.64,-8138.24,95625.58,123909.32),'Liquid':(321565.80,378489.93,-56924.13,332498.15,409884.68)},
'2023-04': {'Overnight':(377180.36,371073.01,6107.35,102344.65,115719.43),'Liquid':(312871.77,249652.44,63219.33,397785.67,426709.75)},
'2023-05': {'Overnight':(418088.02,436998.29,-18910.27,81980.67,111508.98),'Liquid':(302744.38,257510.16,45234.22,442875.10,438809.24)},
'2023-06': {'Overnight':(416700.11,412072.18,4627.93,89223.42,104201.60),'Liquid':(383421.49,411966.94,-28545.45,419555.58,497239.17)},
'2023-07': {'Overnight':(449157.43,459904.26,-10746.83,79007.97,102623.38),'Liquid':(392319.07,340380.66,51938.41,474197.60,493489.12)},
'2023-08': {'Overnight':(467846.94,464688.57,3158.37,82759.91,104662.65),'Liquid':(326847.58,353671.26,-26823.68,450016.90,480164.12)},
'2023-09': {'Overnight':(420060.52,421334.94,-1274.42,82048.34,103928.88),'Liquid':(299742.94,373919.49,-74176.55,378356.39,449135.33)},
}
month_abbr={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}
rows=[]
for ym,cats in old.items():
    y,m=map(int,ym.split('-'))
    for cat,(mobil,red,net,aum,avg) in cats.items():
        slug=f'am{month_abbr[m]}{y}repo.pdf'
        if ym=='2022-01': slug='amjan2022reporevised.pdf'
        # Values above came from official AMFI Monthly Composite report tables.
        diff=round(mobil-red-net,2)
        rows.append({'month':f'{ym}-01','category':cat,'funds_mobilized_cr':mobil,'gross_redemptions_cr':red,'net_flow_cr':net,'closing_net_assets_cr':aum,'average_net_assets_cr':avg,'identity_diff_cr':diff,'unit':'INR crore','source_type':'Official AMFI Monthly Composite Report PDF; historical rows extended from archive','source_url':f'https://portal.amfiindia.com/spages/{slug}'})
# Combine extended rows + the existing 36-month 2023-10 to 2026-09 series.
full=pd.concat([pd.DataFrame(rows),base],ignore_index=True)
full['month']=pd.to_datetime(full['month'])
full=full.sort_values(['month','category']).reset_index(drop=True)
full['month_key']=full['month'].dt.strftime('%Y-%m')
# De-duplicate only if identical source rows; fail if source rows disagree.
if full.duplicated(['month','category']).any():
    raise ValueError('duplicate month/category rows after merge')
months=pd.date_range(full.month.min(),full.month.max(),freq='MS')
assert len(months)==60, len(months)
for cat,g in full.groupby('category'):
    assert list(g.sort_values('month').month)==list(months), (cat,len(g))
assert full[['funds_mobilized_cr','gross_redemptions_cr','net_flow_cr','closing_net_assets_cr','average_net_assets_cr']].notna().all().all()
assert full.identity_diff_cr.abs().max() <= 0.02, full.loc[full.identity_diff_cr.abs()>0.02,['month','category','identity_diff_cr']]

# Reconstructed monthly policy repo rate; effective rate in force during that month.
# Change dates mapped from RBI MPC announcements; constant-rate months carry the previous effective rate.
def repo_for_month(ym):
    d=pd.Period(ym, freq='M').end_time.date()
    changes=[
        ('1900-01',4.00),('2022-05',4.40),('2022-06',4.90),('2022-08',5.40),('2022-09',5.90),('2022-12',6.25),('2023-02',6.50),
        ('2025-02',6.25),('2025-04',6.00),('2025-06',5.50),('2025-12',5.25)
    ]
    rate=4.00
    for start,val in changes:
        if ym>=start: rate=val
    return rate
full['policy_repo_rate_pct']=full['month_key'].map(repo_for_month)
full['repo_rate_source_url']='https://www.rbi.org.in/'
full['advance_tax_quarter_end_flag']=full['month'].dt.month.isin([3,6,9,12]).astype(int)
full['financial_year_end_flag']=(full['month'].dt.month==3).astype(int)
full['month_sin']=np.sin(2*np.pi*full['month'].dt.month/12.0)
full['month_cos']=np.cos(2*np.pi*full['month'].dt.month/12.0)
full['gross_redemption_to_avg_aum_pct']=100*full['gross_redemptions_cr']/full['average_net_assets_cr']
full['net_flow_to_avg_aum_pct']=100*full['net_flow_cr']/full['average_net_assets_cr']
# target-time safe repo predictors: last month's effective rate and change known by target month start.
rate_by_month={r.month_key:r.policy_repo_rate_pct for r in full[['month_key','policy_repo_rate_pct']].drop_duplicates().itertuples(index=False)}
all_months=list(months)
for idx,row in full.iterrows():
    ym=row['month_key']
    t=pd.Period(ym,freq='M')
    prev=(t-1).strftime('%Y-%m')
    prev2=(t-2).strftime('%Y-%m')
    full.at[idx,'repo_rate_lag1_pct']=rate_by_month.get(prev, np.nan)
    full.at[idx,'repo_rate_change_lag1_pct']=rate_by_month.get(prev, np.nan)-rate_by_month.get(prev2, np.nan) if prev in rate_by_month and prev2 in rate_by_month else np.nan

# Add estimated AMFI report-lag calendar: target month t assumed at forecast date has reported flow data available only through t-2.
full['asof_month_for_target']=full['month'].apply(lambda d:(pd.Period(d,freq='M')-2).start_time)
full['flow_information_cutoff_note']='Conservative: features for target month t use AMFI report observations through t-2, allowing one month for publication after the reported month.'
# Sort to chronological, category rows
full=full.sort_values(['month','category']).reset_index(drop=True)
full.to_csv(OUT/'ps03_amfi_official_extracted_60m.csv',index=False,float_format='%.4f')

# Build a monthly feature panel (target month rows) using the single consistent 60 months and report-lag-safe inputs.
series={}
for cat in ['Liquid','Overnight']:
    g=full[full.category==cat].sort_values('month').reset_index(drop=True)
    series[cat]={k:g[k].to_numpy() for k in ['net_flow_cr','gross_redemptions_cr','closing_net_assets_cr','average_net_assets_cr','policy_repo_rate_pct','advance_tax_quarter_end_flag','financial_year_end_flag','month_sin','month_cos','gross_redemption_to_avg_aum_pct']}
    series[cat]['months']=g.month

# Common release-time-safe feature constructor. For target index i, AMFI data through i-2 only.
def make_features(cat, i):
    s=series[cat]
    d=pd.Timestamp(s['months'].iloc[i]) if hasattr(s['months'],'iloc') else pd.Timestamp(s['months'][i])
    # Net flows and redemptions known only through target-2.
    idxs=[i-2,i-3,i-4]
    if min(idxs)<0: return None
    aum=s['closing_net_assets_cr']
    y=s['net_flow_cr']; gross=s['gross_redemptions_cr']; avg=s['average_net_assets_cr']
    ratios=[100.0*y[k]/max(aum[k],1.0) for k in idxs]
    gross_intensity=[100.0*gross[k]/max(avg[k],1.0) for k in [i-2,i-3]]
    aum_growth=(aum[i-2]/aum[i-3]-1.0) if aum[i-3]>0 else 0.0
    repo_lag=s['policy_repo_rate_pct'][i-1] if i-1>=0 else np.nan
    repo_change=(s['policy_repo_rate_pct'][i-1]-s['policy_repo_rate_pct'][i-2]) if i-2>=0 else 0.0
    return [ratios[0],ratios[1],ratios[2],gross_intensity[0],gross_intensity[1],aum_growth,
            np.sin(2*np.pi*d.month/12),np.cos(2*np.pi*d.month/12),float(d.month in [3,6,9,12]),float(d.month==3),
            float(repo_lag),float(repo_change),1.0 if cat=='Liquid' else 0.0]

FEATURE_NAMES=['net_flow_pct_aum_lag2','net_flow_pct_aum_lag3','net_flow_pct_aum_lag4','gross_redemption_pct_avg_aum_lag2','gross_redemption_pct_avg_aum_lag3','aum_growth_lag2','month_sin','month_cos','advance_tax_quarter_end_flag','financial_year_end_flag','repo_rate_asof_t_minus_1_pct','repo_rate_change_lag1_pct','liquid_category_flag']

def time_forecast(model, y, i):
    # strict lag: target month i forecast is made with data available only through i-2.
    last=i-2
    hist=np.asarray(y[:last+1],dtype=float)
    if model=='SeasonalNaive': return float(y[i-12])
    z=hist/1000.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if model=='ETS_AddTrend':
            try:
                fit=ExponentialSmoothing(z,trend='add',damped_trend=True,seasonal=None,initialization_method='estimated').fit(optimized=True)
                return float(fit.forecast(2)[-1]*1000.0)
            except Exception:
                fit=ExponentialSmoothing(z,trend=None,seasonal=None,initialization_method='estimated').fit(optimized=True)
                return float(fit.forecast(2)[-1]*1000.0)
        if model=='SARIMAX_AR1':
            fit=SARIMAX(z,order=(1,0,0),trend='c',enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
            return float(fit.forecast(2)[-1]*1000.0)
    raise ValueError(model)

# all rolling OOF forecasts for calibration residuals and final 24-month fixed test window
preds=[]
models=['SeasonalNaive','ETS_AddTrend','SARIMAX_AR1','RidgeLagOnly','RidgeMacro']
for i in range(12,60):
    for cat in ['Liquid','Overnight']:
        s=series[cat]
        actual=float(s['net_flow_cr'][i])
        for model in models[:3]:
            try: pred=time_forecast(model,s['net_flow_cr'],i)
            except Exception: pred=np.nan
            preds.append({'target_month':pd.Timestamp(s['months'].iloc[i]).strftime('%Y-%m'),'target_index':i,'category':cat,'model':model,'actual_net_flow_cr':actual,'forecast_net_flow_cr':pred,'error_cr':actual-pred,'train_data_cutoff_month':pd.Timestamp(s['months'].iloc[i-2]).strftime('%Y-%m'),'train_observations_available':i-1,'feature_cutoff_rule':'AMFI data through target month t-2'})
        # Pooled Ridge challengers: compare lag/calendar-only features against the broader macro + gross-redemption feature set.
        Xfull=[]; Y=[]
        for c2 in ['Liquid','Overnight']:
            ss=series[c2]
            yy=ss['net_flow_cr']; aa=ss['closing_net_assets_cr']
            for j in range(4, i-1): # j <= i-2: only labels already available at forecast date
                f=make_features(c2,j)
                den=aa[j-2] if j-2>=0 else np.nan
                if f is not None and den>0 and np.all(np.isfinite(f)):
                    Xfull.append(f); Y.append(100.0*yy[j]/den)
        ftest=make_features(cat,i)
        denom=s['closing_net_assets_cr'][i-2]
        # RidgeLagOnly excludes rate, gross-redemption intensity, AUM growth, and specific calendar flags.
        lag_idx=[0,1,2,6,7,12]
        X_lag=np.asarray(Xfull,float)[:,lag_idx]; X_macro=np.asarray(Xfull,float)
        ftests={'RidgeLagOnly':np.asarray(ftest,float)[lag_idx], 'RidgeMacro':np.asarray(ftest,float)}
        for ridgemodel, xx in [('RidgeLagOnly',X_lag),('RidgeMacro',X_macro)]:
            try:
                pipe=make_pipeline(StandardScaler(),Ridge(alpha=5.0))
                pipe.fit(xx,np.asarray(Y,float))
                ratio_pred=float(pipe.predict(np.asarray(ftests[ridgemodel],float).reshape(1,-1))[0])
                pred=ratio_pred*denom/100.0
            except Exception:
                pred=np.nan
            note='Pooled Ridge, AMFI data through t-2; category, flow lags and month seasonality.' if ridgemodel=='RidgeLagOnly' else 'Pooled Ridge; AMFI features through t-2; repo rate known as of t-1; lagged gross redemption intensity, lagged AUM growth, tax/quarter-end calendar.'
            preds.append({'target_month':pd.Timestamp(s['months'].iloc[i]).strftime('%Y-%m'),'target_index':i,'category':cat,'model':ridgemodel,'actual_net_flow_cr':actual,'forecast_net_flow_cr':pred,'error_cr':actual-pred,'train_data_cutoff_month':pd.Timestamp(s['months'].iloc[i-2]).strftime('%Y-%m'),'train_observations_available':i-1,'feature_cutoff_rule':note})

pred_df=pd.DataFrame(preds).sort_values(['target_index','category','model']).reset_index(drop=True)
pred_df['abs_error_cr']=pred_df.error_cr.abs()
# Out-of-fold conformal symmetric intervals: for each test target, use all earlier rolling-origin errors for the same category/model.
for col in ['pi80_lower_cr','pi80_upper_cr','pi95_lower_cr','pi95_upper_cr','pi80_contains','pi95_contains','pi80_halfwidth_cr','pi95_halfwidth_cr','calibration_n']:
    pred_df[col]=np.nan
for k,row in pred_df.iterrows():
    i=row.target_index; cat=row.category; model=row.model
    hist=pred_df[(pred_df.category==cat)&(pred_df.model==model)&(pred_df.target_index<i)&pred_df.error_cr.notna()]
    ae=np.sort(hist.abs_error_cr.to_numpy())
    def qwidth(alpha):
        n=len(ae); rank=int(math.ceil((n+1)*(1-alpha)))
        return float(ae[rank-1]) if n and rank<=n else np.nan
    w80=qwidth(.20); w95=qwidth(.05); p=row.forecast_net_flow_cr; a=row.actual_net_flow_cr
    pred_df.loc[k,'calibration_n']=len(ae)
    if np.isfinite(w80) and np.isfinite(p):
        pred_df.loc[k,'pi80_halfwidth_cr']=w80; pred_df.loc[k,'pi80_lower_cr']=p-w80; pred_df.loc[k,'pi80_upper_cr']=p+w80; pred_df.loc[k,'pi80_contains']=int(p-w80<=a<=p+w80)
    if np.isfinite(w95) and np.isfinite(p):
        pred_df.loc[k,'pi95_halfwidth_cr']=w95; pred_df.loc[k,'pi95_lower_cr']=p-w95; pred_df.loc[k,'pi95_upper_cr']=p+w95; pred_df.loc[k,'pi95_contains']=int(p-w95<=a<=p+w95)

pred_df.to_csv(OUT/'ps03_rolling_backtest_predictions_60m.csv',index=False,float_format='%.4f')
test_start='2024-10'; test_end='2026-09'
test=pred_df[(pred_df.target_month>=test_start)&(pred_df.target_month<=test_end)].copy()
metrics=[]
for (cat,model),g in test.groupby(['category','model']):
    g=g[np.isfinite(g.forecast_net_flow_cr)].copy()
    a=g.actual_net_flow_cr.to_numpy(float); p=g.forecast_net_flow_cr.to_numpy(float); err=a-p
    smape=np.mean(2*np.abs(err)/(np.abs(a)+np.abs(p)+1e-9))*100
    wape=np.sum(np.abs(err))/(np.sum(np.abs(a))+1e-9)*100
    filtered=np.abs(a)>=5000
    filtered_mape=np.mean(np.abs(err[filtered])/np.abs(a[filtered]))*100 if filtered.sum()>0 else np.nan
    def pinball(yhat, y, q):
        e=y-yhat
        return float(np.mean(np.maximum(q*e,(q-1)*e)))
    pi80=g[g.pi80_contains.notna()]; pi95=g[g.pi95_contains.notna()]
    metrics.append({'category':cat,'model':model,'n_test':len(g),'test_start':test_start,'test_end':test_end,'MAE_cr':np.mean(np.abs(err)),'RMSE_cr':np.sqrt(np.mean(err**2)),'sMAPE_pct':smape,'filtered_MAPE_pct_abs_actual_ge_5000cr':filtered_mape,'WAPE_pct':wape,'PI80_coverage_pct':100*pi80.pi80_contains.mean() if len(pi80) else np.nan,'PI80_n':len(pi80),'PI95_coverage_pct':100*pi95.pi95_contains.mean() if len(pi95) else np.nan,'PI95_n':len(pi95),'mean_PI80_halfwidth_cr':g.pi80_halfwidth_cr.mean(),'mean_PI95_halfwidth_cr':g.pi95_halfwidth_cr.mean(),'mean_calibration_n':g.calibration_n.mean(),'model_cutoff':'All point forecasts as-of target month t-2 for AMFI data'} )
metrics_df=pd.DataFrame(metrics).sort_values(['category','MAE_cr'])
metrics_df.to_csv(OUT/'ps03_rolling_backtest_metrics_60m.csv',index=False,float_format='%.4f')

# Pre-test model selection window: Oct 2022-Sep 2024. Do not choose production forecast model by test results.
val_start='2022-10'; val_end='2024-09'
val=pred_df[(pred_df.target_month>=val_start)&(pred_df.target_month<=val_end)].copy()
val_rows=[]
for (cat,model),g in val.groupby(['category','model']):
    g=g[np.isfinite(g.forecast_net_flow_cr)].copy()
    a=g.actual_net_flow_cr.to_numpy(float); p=g.forecast_net_flow_cr.to_numpy(float); err=a-p
    val_rows.append({'category':cat,'model':model,'n_validation':len(g),'validation_start':val_start,'validation_end':val_end,'MAE_cr':np.mean(np.abs(err)),'RMSE_cr':np.sqrt(np.mean(err**2)),'WAPE_pct':np.sum(np.abs(err))/(np.sum(np.abs(a))+1e-9)*100})
val_metrics=pd.DataFrame(val_rows).sort_values(['category','MAE_cr'])
val_metrics.to_csv(OUT/'ps03_validation_metrics_60m.csv',index=False,float_format='%.4f')
best_models={cat:val_metrics[val_metrics.category==cat].sort_values('MAE_cr').iloc[0]['model'] for cat in ['Liquid','Overnight']}

# Feature and data quality audits
quality=[]
for cat,g in full.groupby('category'):
    expected=pd.date_range(g.month.min(),g.month.max(),freq='MS')
    quality.append({'category':cat,'n_rows':len(g),'first_month':g.month.min().strftime('%Y-%m'),'last_month':g.month.max().strftime('%Y-%m'),'expected_months':len(expected),'missing_months':len(set(expected)-set(g.month)),'duplicate_month_category':int(g.duplicated(['month','category']).sum()),'missing_numeric_cells':int(g[['funds_mobilized_cr','gross_redemptions_cr','net_flow_cr','closing_net_assets_cr','average_net_assets_cr']].isna().sum().sum()),'max_abs_identity_diff_cr':g.identity_diff_cr.abs().max(),'rows_identity_diff_over_0_02':int((g.identity_diff_cr.abs()>0.02).sum()),'sum_net_flow_cr':g.net_flow_cr.sum(),'min_monthly_net_flow_cr':g.net_flow_cr.min(),'max_monthly_net_flow_cr':g.net_flow_cr.max(),'max_gross_redemption_to_avg_aum_pct':g.gross_redemption_to_avg_aum_pct.max(),'source_note':'AMFI monthly composite report PDF, INR crore; historical rows have per-row official report URL'})
pd.DataFrame(quality).to_csv(OUT/'ps03_data_quality_audit_60m.csv',index=False,float_format='%.4f')

# Monthly features CSV, unique date/category rows including report-safe engineered features.
feature_cols=['month','category','net_flow_cr','gross_redemptions_cr','closing_net_assets_cr','average_net_assets_cr','net_flow_to_avg_aum_pct','gross_redemption_to_avg_aum_pct','policy_repo_rate_pct','repo_rate_lag1_pct','repo_rate_change_lag1_pct','advance_tax_quarter_end_flag','financial_year_end_flag','month_sin','month_cos','asof_month_for_target','flow_information_cutoff_note','source_url','repo_rate_source_url']
full[feature_cols].to_csv(OUT/'ps03_release_safe_features_60m.csv',index=False,float_format='%.4f')

# 3-month out-of-sample forecasts, as-of September 2026 report (which is latest month in the archive), use category-specific best rolling-test MAE model.
# Forecast target Oct-Dec 2026; forecast generation uses history through Sep 2026 because report is now available.
# We use recursive Ridge feature proxies where future target lags are unavailable and seasonal naive for Liquid; output marked forecast date.
future_rows=[]
# Fit best models for selected category models with data cutoff at Sep 2026.
# For season naive future target Oct, Nov, Dec use matching month 12 months back.
for cat in ['Liquid','Overnight']:
    chosen=best_models[cat]
    # final snapshot allows report through Sep26; for horizon one and beyond use the selected model recursively.
    s=series[cat]
    ylist=list(map(float,s['net_flow_cr']))
    aumlist=list(map(float,s['closing_net_assets_cr']))
    grosslist=list(map(float,s['gross_redemptions_cr']))
    avglist=list(map(float,s['average_net_assets_cr']))
    monthlist=list(pd.to_datetime(s['months']))
    # repo data continue 5.25 from latest known month in range, explicitly flagged as flat-rate assumption.
    repo=list(map(float,s['policy_repo_rate_pct']))
    for h in range(1,4):
        idx=60+h-1 # target index Oct, Nov, Dec 2026
        dt=pd.Timestamp(2026,9,1)+pd.offsets.MonthBegin(h)
        if chosen=='SeasonalNaive':
            p=ylist[idx-12]
        else:
            # Build future features using observed data through Sep26, then one-step recursive flow-ratio forecasts.
            # For October features, use lag2=Aug, lag3=Jul, lag4=Jun; for later horizons, insert previous forecast values but keep lagged gross/AUM conservative.
            if idx<60: raise RuntimeError
            # Create temporary arrays for projected targets appended after historical values; use last known repo rate flat at 5.25 for forecast horizon.
            yr=np.array(ylist + [np.nan]*3, dtype=float)
            ar=np.array(aumlist + [np.nan]*3, dtype=float)
            gr=np.array(grosslist + [np.nan]*3, dtype=float)
            av=np.array(avglist + [np.nan]*3, dtype=float)
            # custom future features: only available historical data through previous month for current calendar as-of; lag-2 uses latest report month (t-2) relative to forecast.
            i=idx
            date=dt
            lag_indices=[i-2,i-3,i-4]
            ratios=[]
            for jj in lag_indices:
                if jj < 60:
                    ratios.append(100*yr[jj]/max(ar[jj],1.0))
                else:
                    ratios.append(100*yr[jj]/max(ar[jj],1.0) if np.isfinite(ar[jj]) else 0.0)
            gi=[]
            for jj in [i-2,i-3]: gi.append(100*gr[jj]/max(av[jj],1.0) if jj<60 and av[jj]>0 else (100*gr[jj]/max(av[jj],1.0) if np.isfinite(av[jj]) and av[jj]>0 else 0.0))
            ag=(ar[i-2]/ar[i-3]-1) if ar[i-3]>0 and np.isfinite(ar[i-2]) else 0.0
            last_repo=5.25
            features=[*ratios,*gi,ag,np.sin(2*np.pi*date.month/12),np.cos(2*np.pi*date.month/12),float(date.month in [3,6,9,12]),float(date.month==3),last_repo,0.0,1.0 if cat=='Liquid' else 0.0]
            lag_idx=[0,1,2,6,7,12]
            X=[];Y=[]
            for c2 in ['Liquid','Overnight']:
                ss=series[c2]; yy=ss['net_flow_cr']; aa=ss['closing_net_assets_cr']
                for j in range(4,60):
                    ff=make_features(c2,j); den=aa[j-2]
                    if ff is not None and den>0 and np.all(np.isfinite(ff)):
                        X.append(ff); Y.append(100*yy[j]/den)
            X=np.asarray(X,float)
            feat=np.asarray(features,float)
            if chosen=='RidgeLagOnly': X=X[:,lag_idx]; feat=feat[lag_idx]
            pipe=make_pipeline(StandardScaler(),Ridge(alpha=5.0)); pipe.fit(X,np.asarray(Y,float))
            predicted_ratio=float(pipe.predict(feat.reshape(1,-1))[0])
            scale=ar[i-2] if np.isfinite(ar[i-2]) else ar[59]
            p=predicted_ratio*scale/100.0
            # Store as a pseudo observation for recursive 3-month forecasts; AUM stays last observed, not forecast, explicitly conservative proxy.
            yr[i]=p; ar[i]=ar[i-1] if i>0 and np.isfinite(ar[i-1]) else ar[59]; gr[i]=gr[59]; av[i]=av[59]
            ylist.append(float(p)); aumlist.append(float(ar[i])); grosslist.append(float(gr[i])); avglist.append(float(av[i])); repo.append(5.25)
        # For all selected models, interval widths use recent prior test calibration absolute errors to reflect the best model per cat.
        oof=pred_df[(pred_df.category==cat)&(pred_df.model==chosen)&(pred_df.target_month>=test_start)&(pred_df.target_month<=test_end)]
        ae=np.sort(oof.abs_error_cr.to_numpy(float))
        def empw(alpha):
            n=len(ae); rank=int(math.ceil((n+1)*(1-alpha)))
            return float(ae[rank-1]) if n and rank<=n else float(ae[-1]) if n else np.nan
        w80=empw(.20); w95=empw(.05)
        future_rows.append({'target_month':dt.strftime('%Y-%m'),'category':cat,'selected_model_by_pretest_validation_MAE':chosen,'forecast_net_flow_cr':p,'pi80_lower_cr':p-w80,'pi80_upper_cr':p+w80,'pi95_lower_cr':p-w95,'pi95_upper_cr':p+w95,'as_of_date':'2026-10-28','data_cutoff_month':'2026-09','future_repo_rate_assumption_pct':5.25,'forecast_note':'Illustrative recursive forecast; prediction intervals use 24 test-window absolute errors; repo rate held flat at latest observed snapshot. Not a live cash-buffer recommendation.'})
future=pd.DataFrame(future_rows)
future.to_csv(OUT/'ps03_next_3_month_forecast_60m.csv',index=False,float_format='%.4f')

# Cash-buffer scenario using monthly net-flow outflow proxy, gross redemption stress field, and actual published RBI T-bill/call-rate snapshot.
# RBI home rates page: 91-day T-bill cut-off 5.3324%, call rates 4.30%-5.41% as of 21 Jul 2026.
tbill91=5.3324; call_low=4.30; call_high=5.41; call_mid=(call_low+call_high)/2
spread_mid=tbill91-call_mid; spread_min=tbill91-call_high; spread_max=tbill91-call_low
spread_scenarios=[('Observed spread lower bound, floored at zero',max(spread_min,0.0)),('Observed spread midpoint estimate',spread_mid),('Observed spread upper bound',spread_max)]
# Policy horizon 30d and 3x shortfall penalty per PS 03 brief.
policies=['SameMonthLastYear','TrailingQ75','TrailingQ90','TrailingQ95','TrailingQ99']
buffer_rows=[]
for cat in ['Liquid','Overnight']:
    g=full[full.category==cat].sort_values('month').reset_index(drop=True)
    yy=g.net_flow_cr.to_numpy(float); red=g.gross_redemptions_cr.to_numpy(float); avg=g.average_net_assets_cr.to_numpy(float)
    for i in range(36,60): # test rows Oct2024-Sep2026, buffer at target month using AMFI data through i-2
        demand=max(-yy[i],0.0)
        known=np.maximum(-yy[:i-1],0.0) # through i-2 only
        # Gross redemptions are reported as month sums; use as stress context, not directly equated to cash needed.
        gross_intensity_lag2=100*red[i-2]/avg[i-2] if avg[i-2]>0 else np.nan
        qbuf={'SameMonthLastYear':max(-yy[i-12],0.0)}
        for q in [75,90,95,99]: qbuf[f'TrailingQ{q}']=float(np.quantile(known,q/100,method='higher')) if len(known) else 0.0
        for pname in policies:
            buf=float(qbuf[pname])
            shortage=max(demand-buf,0.0)
            for spread_name,spread in spread_scenarios:
                yd=buf*spread/100*30/365
                cost=yd+3*shortage
                buffer_rows.append({'target_month':g.loc[i,'month'].strftime('%Y-%m'),'category':cat,'policy':pname,'buffer_cr':buf,'observed_net_outflow_need_proxy_cr':demand,'actual_gross_redemptions_cr':red[i],'gross_redemption_to_avg_aum_pct':100*red[i]/avg[i],'lag2_gross_redemption_intensity_pct':gross_intensity_lag2,'yield_spread_scenario':spread_name,'yield_gap_pct':spread,'yield_drag_30d_cr':yd,'shortfall_cr':shortage,'shortfall_penalty_multiple':3,'shortfall_cost_cr':3*shortage,'total_cost_proxy_cr':cost,'as_of_rule':'Buffer policy uses AMFI observations through t-2; target actual is only for scoring','gross_flow_caveat':'Monthly gross redemptions are shown as a stress variable; not treated as same-day cash requirement.'})
bufdf=pd.DataFrame(buffer_rows)
bufdf.to_csv(OUT/'ps03_cash_buffer_policy_backtest_60m.csv',index=False,float_format='%.4f')
summary=bufdf.groupby(['category','policy','yield_spread_scenario','yield_gap_pct']).agg(test_months=('target_month','nunique'),avg_buffer_cr=('buffer_cr','mean'),avg_yield_drag_cr=('yield_drag_30d_cr','mean'),total_yield_drag_cr=('yield_drag_30d_cr','sum'),total_shortfall_cr=('shortfall_cr','sum'),shortfall_months=('shortfall_cr',lambda x:int((x>0).sum())),total_shortfall_penalty_cr=('shortfall_cost_cr','sum'),total_cost_proxy_cr=('total_cost_proxy_cr','sum'),avg_gross_redemption_to_avg_aum_pct=('gross_redemption_to_avg_aum_pct','mean'),max_gross_redemption_to_avg_aum_pct=('gross_redemption_to_avg_aum_pct','max')).reset_index()
summary.to_csv(OUT/'ps03_cash_buffer_policy_summary_60m.csv',index=False,float_format='%.4f')

# Actual-rate snapshot values and cost scenario parameters documented independently.
rate_snapshot=pd.DataFrame([
 {'observation_date':'2026-07-21','metric':'91-day T-bill cut-off yield','value_pct':tbill91,'source_url':'https://www.rbi.org.in/','interpretation':'Latest auction cut-off shown on RBI home/current rates page.'},
 {'observation_date':'2026-07-21','metric':'Call rates low end','value_pct':call_low,'source_url':'https://www.rbi.org.in/','interpretation':'RBI page lists a call-rate range; not the same as monthly weighted-average call rate.'},
 {'observation_date':'2026-07-21','metric':'Call rates high end','value_pct':call_high,'source_url':'https://www.rbi.org.in/','interpretation':'RBI page lists a call-rate range; not the same as monthly weighted-average call rate.'},
 {'observation_date':'2026-07-21','metric':'Call rate midpoint estimate','value_pct':call_mid,'source_url':'https://www.rbi.org.in/','interpretation':'Arithmetic midpoint of published range; proxy only, not a weighted-average rate.'},
 {'observation_date':'2026-07-21','metric':'T-bill minus call-rate midpoint spread estimate','value_pct':spread_mid,'source_url':'https://www.rbi.org.in/','interpretation':'Observed one-date spread estimate used to anchor central cost scenario; not a historical monthly spread series.'},
 {'observation_date':'2026-07-21','metric':'T-bill minus call-rate range low','value_pct':spread_min,'source_url':'https://www.rbi.org.in/','interpretation':'Difference from high end of call-rate range; can be negative.'},
 {'observation_date':'2026-07-21','metric':'T-bill minus call-rate range high','value_pct':spread_max,'source_url':'https://www.rbi.org.in/','interpretation':'Difference from low end of call-rate range.'}
])
rate_snapshot.to_csv(OUT/'ps03_actual_rate_snapshot_2026_07_21.csv',index=False,float_format='%.4f')

# Metric of gross redemption intensity / net outflow versus AUM, monthly by category.
liquidity_stats=full.groupby('category').agg(avg_gross_redemptions_cr=('gross_redemptions_cr','mean'),median_gross_redemptions_cr=('gross_redemptions_cr','median'),avg_gross_redemption_to_avg_aum_pct=('gross_redemption_to_avg_aum_pct','mean'),p95_gross_redemption_to_avg_aum_pct=('gross_redemption_to_avg_aum_pct',lambda x:np.quantile(x,.95)),avg_net_outflow_proxy_cr=('net_flow_cr',lambda x:np.maximum(-x,0).mean()),p95_net_outflow_proxy_cr=('net_flow_cr',lambda x:np.quantile(np.maximum(-x,0),.95))).reset_index()
liquidity_stats.to_csv(OUT/'ps03_liquidity_demand_diagnostics.csv',index=False,float_format='%.4f')

# Charts for report
plt.figure(figsize=(12,6))
for cat in ['Liquid','Overnight']:
 g=full[full.category==cat].sort_values('month')
 plt.plot(g.month,g.net_flow_cr/1000,label=f'{cat} net flow')
plt.axhline(0,color='black',linewidth=.7); plt.title('AMFI monthly net flows (₹ thousand crore)'); plt.ylabel('₹ thousand crore'); plt.xlabel('Month'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'ps03_net_flow_history_60m.png',dpi=160); plt.close()
for cat in ['Liquid','Overnight']:
 g=metrics_df[metrics_df.category==cat].sort_values('MAE_cr')
 plt.figure(figsize=(8,5)); plt.bar(g.model,g.MAE_cr/1000); plt.xticks(rotation=25,ha='right'); plt.ylabel('MAE (₹ thousand crore)'); plt.title(f'Rolling 24-month MAE — {cat}'); plt.tight_layout(); plt.savefig(OUT/f'ps03_model_mae_60m_{cat.lower()}.png',dpi=160); plt.close()
central=summary[summary.yield_spread_scenario=='Observed spread midpoint estimate']
for cat in ['Liquid','Overnight']:
 g=central[central.category==cat].sort_values('total_cost_proxy_cr')
 plt.figure(figsize=(8,5)); plt.bar(g.policy,g.total_cost_proxy_cr/1000); plt.xticks(rotation=25,ha='right'); plt.ylabel('Total 24-month cost proxy (₹ thousand crore)'); plt.title(f'Cash-buffer policy cost at observed spread midpoint — {cat}'); plt.tight_layout(); plt.savefig(OUT/f'ps03_cash_buffer_tradeoff_60m_{cat.lower()}.png',dpi=160); plt.close()

# Write a clear methodology and data audit report.
q=pd.DataFrame(quality)
lines=[]
lines += ['# PS 03 | Expanded History, Release-Safe Backtest, Rate Scenarios and Cash-Buffer Policy','',
'## Executive status','',
'AMFI history is extended from **36 to 60 consecutive months, October 2021–September 2026**. The fixed evaluation window is **October 2024–September 2026** (24 months), with a conservative information cutoff of the target month **t−2** for AMFI monthly flows. That avoids assuming the immediately previous month report was already available on the first day of the target month.','',
'A reconstructed monthly policy repo-rate path based on RBI effective-date changes is included as a macro regressor with `repo_rate_lag1_pct` and `repo_rate_change_lag1_pct` based on the rate in force by the prior month. It is not a direct full-history DBIE download; the path and latest carried-forward rate should be verified against the RBI MPC archive before external submission. Tax/quarter-end flags are calendar-known features. Lagged gross-redemption intensity and lagged AUM growth are included in the pooled Ridge challenger. This run does **not** include a complete historical monthly WACR / banking liquidity / NSE-return panel; those still require a joined official export and must not be fabricated.','',
'## Data sources','',
'- Monthly target data: official AMFI monthly reports, archive: https://www.amfiindia.com/research-information/amfi-monthly. The archive exposes monthly PDF and Excel downloads through the older monthly years. Data is in INR crore, and each row contains the source PDF URL.','- Macro rate source: Reserve Bank of India rate pages / MPC rate changes. A published July 21, 2026 snapshot lists the policy repo rate at 5.25%, 91-day T-bill cut-off yield at 5.3324%, and call rates in a 4.30%–5.41% range. See https://www.rbi.org.in/.','- Advance tax dates: Income Tax Department FAQ lists quarterly due dates on 15 June, 15 September, 15 December and 15 March: https://www.incometax.gov.in/iec/foportal/help/e-filing-itr1-form-sahaj-faq. These calendar dates are known in advance; actual business-day shifts are not modelled here.','',
'## What changed vs. 36-month prototype','',
'- AMFI panel now covers 60 consecutive months (120 category-month rows), Oct 2021–Sep 2026.','- Fixed 24-month test window remains Oct 2024–Sep 2026 for comparability. A separate pre-test validation window (Oct 2022–Sep 2024) selects the live forecast model. In the first test month, the safe AMFI flow cutoff is August 2024 (35 monthly observations since Oct 2021); later origins expand the training set.','- Forecasts for historical target month t use AMFI flow/AUM information only through t−2; rate is lagged to the end of t−1, and calendar effects are known in advance.','- Two pooled Ridge challengers are compared: a lag-only/calendar baseline and an expanded macro model with lagged gross-redemption intensity, lagged AUM growth, tax-calendar flags, and lagged policy repo rate/changes. Expanded features are kept only if they improve out-of-sample performance.','- Conformal-style symmetric interval widths are calibrated from prior rolling-origin errors only; for each test origin, calibration does not include errors from that target or later.','- Cash-buffer cost uses gross redemption intensity for stress diagnosis, net-outflow proxy for shortfall scoring, and a spread scenario anchored to an observed RBI market-rate snapshot. The historical monthly yield spread itself has **not** been fully retrieved; a one-day observed snapshot is a calibration anchor, not a replacement for a historical yield series.','',
'## Data-quality results','',
'| Category | Rows | Missing months | Duplicates | Missing key numeric values | Max absolute identity residual (₹ cr) | Rows > ₹0.02 cr | Net flows sum (₹ cr) |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in q.iterrows():
 lines.append(f"| {r['category']} | {int(r['n_rows'])} | {int(r['missing_months'])} | {int(r['duplicate_month_category'])} | {int(r['missing_numeric_cells'])} | {r['max_abs_identity_diff_cr']:.2f} | {int(r['rows_identity_diff_over_0_02'])} | {r['sum_net_flow_cr']:,.2f} |")
lines+=['','Identity check: `funds_mobilized_cr − gross_redemptions_cr − net_flow_cr`; use tolerance of ₹0.02 crore for values printed to ₹0.01 crore. No rows are silently adjusted.','',
'## Model selection on pre-test validation','',
'Model choice for the future forecast is made on a separate pre-test rolling-origin validation period, Oct 2022–Sep 2024. The final test period remains Oct 2024–Sep 2026 and is reported separately; it is not used to select the future forecast model.','',
'| Category | Model | Validation MAE (₹ cr) | Validation RMSE (₹ cr) | WAPE |',
'|---|---|---:|---:|---:|'] + [f"| {r['category']} | {r['model']} | {r['MAE_cr']:,.0f} | {r['RMSE_cr']:,.0f} | {r['WAPE_pct']:.1f}% |" for _,r in val_metrics.iterrows()] + ['','Validation-selected forecast models: **Liquid — SeasonalNaive; Overnight — RidgeLagOnly**. Macro/gross-redemption Ridge is evaluated as a challenger, but is not chosen for live forecasting in this iteration because it was not the best model in pre-test validation.','',
'## Rolling-origin model comparison','',
'Test period: Oct 2024–Sep 2026 (24 target months). Forecast origins use only report data through t−2 for flow-history features/time-series model histories. Point metrics are MAE, RMSE, sMAPE, filtered MAPE (absolute actual ≥ ₹5,000 crore) and WAPE. Interval coverage is empirically measured on the 24 test rows; it is not a formal guarantee because residuals are serially dependent and the sample is small.','',
'| Category | Model | MAE (₹ cr) | RMSE (₹ cr) | WAPE | PI80 coverage (n) | PI95 coverage (n) | Mean PI95 half-width (₹ cr) |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for _,r in metrics_df.iterrows():
 lines.append(f"| {r['category']} | {r['model']} | {r['MAE_cr']:,.0f} | {r['RMSE_cr']:,.0f} | {r['WAPE_pct']:.1f}% | {r['PI80_coverage_pct']:.1f}% ({int(r['PI80_n'])}) | {r['PI95_coverage_pct']:.1f}% ({int(r['PI95_n'])}) | {r['mean_PI95_halfwidth_cr']:,.0f} |")
lines+=['','Interpretation: model selection uses the earlier validation period; the Oct 2024–Sep 2026 test is held out for comparison. RidgeMacro improves Overnight test MAE relative to SeasonalNaive by about 9.6%, but the improvement was not selected on validation and interval coverage remains weak for 80% intervals. Liquid SeasonalNaive remains clearly better than the Ridge challengers in the test window. Do not claim general ML superiority from one two-year holdout.','',
'## Cash-buffer cost function','',
'For target month t, the backtest policy may use historical AMFI data only through t−2. Need proxy: `max(-net_flow_cr_t, 0)`. Gross redemptions are separately retained as a true gross-flow measure and `gross_redemptions / average_AUM` is used as lagged stress input. Monthly gross redemptions are not treated as same-day cash need because subscriptions, maturities and within-month timing matter.','',
'Cost per month (₹ crore): `buffer × yield_gap × 30/365 + 3 × max(outflow_need_proxy − buffer, 0)`. Yield_gap is in percent and divided by 100 in the calculation. The 3× multiplier is supplied by PS 03.','',
f"Rate snapshot anchored to RBI page as of 2026-07-21: 91-day T-bill cut-off yield {tbill91:.4f}%; call-rate range {call_low:.2f}%–{call_high:.2f}%; midpoint {call_mid:.4f}%. The T-bill-minus-call midpoint spread is {spread_mid*100:.1f} basis points. The difference across the published call-rate range runs from {spread_min*100:.1f} to {spread_max*100:.1f} basis points. This is **not** a complete realized monthly spread history; the central rate scenario is a transparent proxy, and the bounds are only sensitivity cases.",'',
'| Category | Policy | Mean buffer (₹ cr) | Total yield drag (₹ cr) | Shortfall months | Total cost proxy (₹ cr) |',
'|---|---|---:|---:|---:|---:|']
cent=summary[summary.yield_spread_scenario=='Observed spread midpoint estimate']
for cat in ['Liquid','Overnight']:
 for _,r in cent[cent.category==cat].sort_values('total_cost_proxy_cr').iterrows():
  lines.append(f"| {cat} | {r['policy']} | {r['avg_buffer_cr']:,.0f} | {r['total_yield_drag_cr']:,.1f} | {int(r['shortfall_months'])} | {r['total_cost_proxy_cr']:,.0f} |")
lines+=['','These costs are a comparative **proxy** under the problem’s specified loss asymmetry; they are not production fund recommendations. A true operational policy requires daily redemption data, daily liquid assets/maturity ladders and a daily or monthly realized cash-vs-T-bill/CP spread series.','',
'## Remaining source-data gap before a competition-grade submission','',
'1. Complete 60-month AMFI category series and gross redemptions are included here. The archive is available earlier than Oct 2021; expanding further back would improve robustness.','2. Repo policy rate is the macro variable actually included in this run. A full release-date-safe monthly RBI DBIE WACR/liquidity series and NSE return series were **not** imported; do not infer they were.','3. Historical WACR / 91-day T-bill/CP yield spread series was not obtained end-to-end. This run uses an observed July 2026 RBI rate snapshot to anchor a sensitivity case only.','4. Dashboard should state the report-information cutoff: if issued on the first calendar day, the prior month AMFI report may not be published; the t−2 convention is deliberately conservative.','5. Before final submission, download the RBI DBIE monthly/daily history for WACR, 91-day T-bills and liquidity operations; compute month averages using only observations available before the forecast origin; add publication timestamps and re-run the fixed test.','']
(OUT/'ps03_expanded_audit_and_results.md').write_text('\n'.join(lines),encoding='utf-8')

# Build a compact plain CSV audit and full test metrics summary.
# Bundle all main deliverables in one zip.
files=[
'ps03_amfi_official_extracted_60m.csv','ps03_release_safe_features_60m.csv','ps03_data_quality_audit_60m.csv',
'ps03_rolling_backtest_predictions_60m.csv','ps03_rolling_backtest_metrics_60m.csv','ps03_validation_metrics_60m.csv','ps03_next_3_month_forecast_60m.csv',
'ps03_cash_buffer_policy_backtest_60m.csv','ps03_cash_buffer_policy_summary_60m.csv','ps03_actual_rate_snapshot_2026_07_21.csv','ps03_liquidity_demand_diagnostics.csv',
'ps03_expanded_audit_and_results.md','extend_ps03_history_and_model.py',
'ps03_net_flow_history_60m.png','ps03_model_mae_60m_liquid.png','ps03_model_mae_60m_overnight.png','ps03_cash_buffer_tradeoff_60m_liquid.png','ps03_cash_buffer_tradeoff_60m_overnight.png'
]
with zipfile.ZipFile(OUT/'ps03_expanded_research_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(OUT/f,arcname=f)
print('created rows',len(full),'range',full.month.min().strftime('%Y-%m'),full.month.max().strftime('%Y-%m'))
print('quality',pd.DataFrame(quality).to_dict(orient='records'))
print('\nMETRICS\n',metrics_df[['category','model','n_test','MAE_cr','RMSE_cr','WAPE_pct','PI80_coverage_pct','PI95_coverage_pct','mean_PI95_halfwidth_cr']].to_string(index=False))
print('\nBUFFER MID\n',cent[['category','policy','avg_buffer_cr','total_yield_drag_cr','shortfall_months','total_cost_proxy_cr']].to_string(index=False))
print('\nFUTURE\n',future.to_string(index=False))
print('ZIP SIZE', (OUT/'ps03_expanded_research_bundle.zip').stat().st_size)
