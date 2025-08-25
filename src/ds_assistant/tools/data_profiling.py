import numpy as np, pandas as pd, polars as pl, os
import matplotlib.pyplot as plt
from ..common.schemas import BasicProfile, ShapeInfo, DTypeInfo, MissingInfo, NumericSummary, CardinalityInfo, ScaleCheck, ScaleStat

def _to_pandas(df: pl.DataFrame)->pd.DataFrame: return df.to_pandas(use_pyarrow_extension_array=True)

def basic_profile(df: pl.DataFrame, artifacts_dir:str)->BasicProfile:
  p=_to_pandas(df)
  shape=ShapeInfo(rows=p.shape[0], cols=p.shape[1])
  dtypes=[DTypeInfo(col=c, dtype=str(p[c].dtype)) for c in p.columns]
  missing=[MissingInfo(col=c, missing_count=int(p[c].isna().sum()), missing_rate=float(p[c].isna().mean())) for c in p.columns]
  num=p.select_dtypes(include=[np.number]).columns.tolist()
  cat=[c for c in p.columns if c not in num]
  ns=[]; 
  for c in num:
    s=p[c].astype(float)
    ns.append(NumericSummary(col=c, min=float(np.nanmin(s)), max=float(np.nanmax(s)), mean=float(np.nanmean(s)), std=float(np.nanstd(s)), skew=float(pd.Series(s).skew(skipna=True)), kurtosis=float(pd.Series(s).kurtosis(skipna=True))))
  card=[CardinalityInfo(col=c, n_unique=int(p[c].nunique(dropna=True))) for c in cat]
  hist_dir=os.path.join(artifacts_dir,"hists"); os.makedirs(hist_dir, exist_ok=True)
  hpaths={}; bpaths={}
  for c in num[:30]:
    fig=plt.figure(); pd.Series(p[c]).plot(kind='hist', bins=30); out=os.path.join(hist_dir,f"hist_{c}.png"); plt.savefig(out); plt.close(fig); hpaths[c]=out
  for c in cat[:30]:
    fig=plt.figure(); pd.Series(p[c]).value_counts(dropna=False).head(30).plot(kind='bar'); out=os.path.join(hist_dir,f"bar_{c}.png"); plt.savefig(out); plt.close(fig); bpaths[c]=out
  return BasicProfile(shape=shape,dtypes=dtypes,missing=missing,numerical_summary=ns,categorical_cardinality=card,distribution_artifacts={"hist_png_paths":hpaths,"bar_png_paths":bpaths})

def scale_check(df: pl.DataFrame)->ScaleCheck:
  p=_to_pandas(df); num=p.select_dtypes(include=[np.number]).columns.tolist()
  stats=[]; ranges=[]
  for c in num:
    s=p[c].astype(float); mn=float(np.nanmin(s)); mx=float(np.nanmax(s)); sd=float(np.nanstd(s))
    stats.append(ScaleStat(col=c, min=mn, max=mx, std=sd))
    if sd>0: ranges.append(mx-mn)
  requires = (max(ranges)/max(1e-9, min(ranges))>10) if ranges else False
  return ScaleCheck(scale_stats=stats, requires_scaling=requires)
