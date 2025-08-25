from ..common.schemas import ProblemSummary, Recommendation, CandidatePlan
def recommend(summary: ProblemSummary)->Recommendation:
  c=[]
  if summary.problem_type in ("binary","multiclass","regression"):
    if summary.dtype_mix.get("categorical_pct",0)>0.3 or summary.high_cardinality_cols:
      c.append(CandidatePlan(model_family="LightGBM/GBDT", why=["Mixed dtypes","handles missing/outliers"], preprocessing=["Median impute","Target/Cat encoding"], tuning_notes=["num_leaves","max_depth"], eval_metrics=["AUC-PR" if summary.problem_type!="regression" else "RMSE","F1" if summary.problem_type!="regression" else "MAE"]))
      c.append(CandidatePlan(model_family="CatBoost", why=["Native categorical"], preprocessing=["Minimal encoding"], tuning_notes=["depth","learning_rate"], eval_metrics=["AUC-ROC" if summary.problem_type!="regression" else "RMSE"]))
    else:
      c.append(CandidatePlan(model_family="ElasticNet/LogReg", why=["Wide/linear"], preprocessing=["StandardScaler","One-hot"], tuning_notes=["alpha","l1_ratio"], eval_metrics=["AUC-ROC" if summary.problem_type!="regression" else "RMSE"]))
  if summary.problem_type=="forecasting":
    c.append(CandidatePlan(model_family="SARIMAX/Prophet", why=["Seasonality/exog"], preprocessing=["Impute gaps","Difference if needed"], tuning_notes=["(p,d,q)(P,D,Q)m","holidays"], eval_metrics=["sMAPE","MAE"]))
    c.append(CandidatePlan(model_family="GBM on lags", why=["Nonlinearities"], preprocessing=["Lag/rolling features"], tuning_notes=["windows","reg"], eval_metrics=["sMAPE","MAE"]))
  primary = c[0].model_family if c else "LightGBM/GBDT"
  return Recommendation(candidates=c, primary_choice=primary)
