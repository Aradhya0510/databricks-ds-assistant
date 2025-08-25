from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ShapeInfo(BaseModel): rows:int; cols:int
class DTypeInfo(BaseModel): col:str; dtype:str
class MissingInfo(BaseModel): col:str; missing_count:int; missing_rate:float
class NumericSummary(BaseModel): col:str; min:float|None=None; max:float|None=None; mean:float|None=None; std:float|None=None; skew:float|None=None; kurtosis:float|None=None
class CardinalityInfo(BaseModel): col:str; n_unique:int

class BasicProfile(BaseModel):
  shape: ShapeInfo
  dtypes: List[DTypeInfo]
  missing: List[MissingInfo]
  numerical_summary: List[NumericSummary]
  categorical_cardinality: List[CardinalityInfo]
  distribution_artifacts: Dict[str, Dict[str, str]]|None=None

class ScaleStat(BaseModel): col:str; min:float|None=None; max:float|None=None; std:float|None=None
class ScaleCheck(BaseModel): scale_stats:List[ScaleStat]; requires_scaling:bool

class CorrPair(BaseModel): col1:str; col2:str; corr:float
class VIFItem(BaseModel): col:str; vif:float
class CorrVIFResult(BaseModel):
  corr_matrix_path: Optional[str]=None
  top_pairwise_corrs: List[CorrPair]=[]
  vif: List[VIFItem]=[]
  multicollinearity_flag: bool=False

class OutliersResult(BaseModel):
  per_col_outlier_pct: List[Dict[str, float]]
  global_outlier_pct: float
  suggest_capping: bool

class TSDetect(BaseModel):
  is_time_series: bool
  time_col: Optional[str]=None
  freq_guess: Optional[str]=None
  missing_timestamps_pct: Optional[float]=None

class TSDecomp(BaseModel):
  trend_seasonality_plots: Dict[str,str]|None=None
  adf: Dict[str,float|bool]|None=None
  acf_pacf_plots: Dict[str,str]|None=None
  seasonality_detected: Optional[bool]=None
  exog: List[Dict[str,float]]|None=None
  exog_recommended: Optional[bool]=None

class ProblemSummary(BaseModel):
  problem_type:str
  shape: ShapeInfo
  dtype_mix: Dict[str,float]
  missing_overall_pct: float
  imbalance_ratio: Optional[float]=None
  skewed_target: Optional[bool]=None
  requires_scaling: bool=False
  high_cardinality_cols: List[str]=[]
  multicollinearity_flag: bool=False
  outliers_heavy: bool=False
  ts: Dict[str,Any]|None=None

class CandidatePlan(BaseModel):
  model_family:str; why:List[str]; preprocessing:List[str]; tuning_notes:List[str]; eval_metrics:List[str]

class Recommendation(BaseModel):
  candidates: List[CandidatePlan]; primary_choice:str

class RunState(BaseModel):
  run_id:str; dataset_ref:str; target:Optional[str]=None; time_col:Optional[str]=None
  basic_profile: Optional[BasicProfile]=None
  scale_check: Optional[ScaleCheck]=None
  corr_vif: Optional[CorrVIFResult]=None
  outliers: Optional[OutliersResult]=None
  ts_diag: Optional[TSDecomp]=None
  summary: Optional[ProblemSummary]=None
  recommendation: Optional[Recommendation]=None
  artifacts_dir: Optional[str]=None
