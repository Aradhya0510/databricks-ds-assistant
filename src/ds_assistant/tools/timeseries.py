import pandas as pd, polars as pl, numpy as np, os, matplotlib.pyplot as plt
from ..common.schemas import TSDetect, TSDecomp
def _to_pandas(df: pl.DataFrame)->pd.DataFrame: return df.to_pandas(use_pyarrow_extension_array=True)
def detect_ts(df: pl.DataFrame, target:str, time_col:str|None=None)->TSDetect:
  p=_to_pandas(df)
  if time_col is None:
    c=[c for c in p.columns if "date" in c.lower() or "time" in c.lower()]
    time_col=c[0] if c else None
  if time_col is None: return TSDetect(is_time_series=False)
  ts=p[[time_col,target]].dropna()
  try:
    ts[time_col]=pd.to_datetime(ts[time_col]); ts=ts.sort_values(time_col); freq=pd.infer_freq(ts[time_col])
    miss=0.0
    if freq:
      rng=pd.date_range(ts[time_col].min(), ts[time_col].max(), freq=freq); miss=100.0*(1-len(ts[time_col].drop_duplicates())/len(rng))
    return TSDetect(is_time_series=True,time_col=time_col,freq_guess=freq,missing_timestamps_pct=miss)
  except Exception: return TSDetect(is_time_series=False)

def stl_adf_acf_pacf(df: pl.DataFrame, target:str, time_col:str, artifacts_dir:str, freq:str|None=None)->TSDecomp:
  import statsmodels.api as sm
  from statsmodels.tsa.stattools import adfuller
  from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
  p=_to_pandas(df); p[time_col]=pd.to_datetime(p[time_col]); ts=p[[time_col,target]].dropna().sort_values(time_col).set_index(time_col)[target].asfreq(freq or pd.infer_freq(p[time_col]) or "D")
  out=TSDecomp()
  try:
    stl=sm.tsa.STL(ts.fillna(method='ffill').fillna(method='bfill'), period=max(2,7)); res=stl.fit()
    fig=res.plot(); stl_path=os.path.join(artifacts_dir,"stl.png"); fig.savefig(stl_path); out.trend_seasonality_plots={"stl":stl_path}
  except Exception: pass
  try:
    a=adfuller(ts.dropna()); out.adf={"stat":float(a[0]),"pvalue":float(a[1]),"stationary":bool(a[1]<0.05)}
  except Exception: pass
  try:
    import matplotlib.pyplot as plt
    fig1=plt.figure(); plot_acf(ts.dropna(), ax=plt.gca()); acf=os.path.join(artifacts_dir,"acf.png"); plt.savefig(acf); plt.close(fig1)
    fig2=plt.figure(); plot_pacf(ts.dropna(), ax=plt.gca()); pacf=os.path.join(artifacts_dir,"pacf.png"); plt.savefig(pacf); plt.close(fig2)
    out.acf_pacf_plots={"acf":acf,"pacf":pacf}
  except Exception: pass
  return out
