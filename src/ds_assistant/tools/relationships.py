import numpy as np, pandas as pd, polars as pl, os, matplotlib.pyplot as plt
from ..common.schemas import CorrVIFResult, CorrPair, VIFItem, OutliersResult

def _to_pandas(df: pl.DataFrame)->pd.DataFrame: return df.to_pandas(use_pyarrow_extension_array=True)

def corr_and_vif(df: pl.DataFrame, artifacts_dir:str, target:str|None=None)->CorrVIFResult:
  p=_to_pandas(df).select_dtypes(include=[np.number]).dropna(axis=1, how='all')
  res=CorrVIFResult(top_pairwise_corrs=[], vif=[], multicollinearity_flag=False)
  if p.shape[1]>=2:
    corr=p.corr(numeric_only=True)
    fig=plt.figure(figsize=(6,5))
    import seaborn as sns
    sns.heatmap(corr, cmap="coolwarm", center=0); path=os.path.join(artifacts_dir,"corr_heatmap.png"); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    res.corr_matrix_path=path
    tri=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)).stack().sort_values(ascending=False)
    res.top_pairwise_corrs=[CorrPair(col1=a,col2=b,corr=float(v)) for (a,b),v in tri.head(15).items()]
    try:
      from statsmodels.stats.outliers_influence import variance_inflation_factor
      import statsmodels.api as sm
      X=p.dropna(); X1=sm.add_constant(X); vifs=[]
      if X.shape[1]<=100:
        for i,col in enumerate(X.columns):
          v=float(variance_inflation_factor(X1.values, i+1)); vifs.append(VIFItem(col=col, vif=v))
        res.vif=vifs; res.multicollinearity_flag=any(v.vif>10 for v in vifs)
    except Exception: pass
  return res

def outliers(df: pl.DataFrame, method:str="iqr")->OutliersResult:
  p=_to_pandas(df).select_dtypes(include=[np.number])
  per=[]; total=0; denom=0
  for c in p.columns:
    s=p[c].astype(float)
    if method=="iqr":
      q1,q3=np.nanpercentile(s,25),np.nanpercentile(s,75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
      mask=(s<lo)|(s>hi)
    else:
      z=(s-np.nanmean(s))/(np.nanstd(s)+1e-9); mask=np.abs(z)>3
    cnt=int(np.nansum(mask)); n=int(np.sum(~np.isnan(s)))
    per.append({"col":c,"pct":100.0*cnt/max(1,n)}); total+=cnt; denom+=n
  return OutliersResult(per_col_outlier_pct=per, global_outlier_pct=100.0*total/max(1,denom), suggest_capping=(100.0*total/max(1,denom))>1.0)
