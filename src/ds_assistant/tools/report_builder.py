from jinja2 import Template
import os
from ..common.schemas import RunState
T=Template("""# DS Assistant Report
**Run ID:** {{ s.run_id }}  
**Dataset:** {{ s.dataset_ref }}
{% if s.basic_profile %}
## Overview
Rows×Cols: {{ s.basic_profile.shape.rows }} × {{ s.basic_profile.shape.cols }}
{% endif %}
{% if s.corr_vif and s.corr_vif.corr_matrix_path %}
## Correlations
![corr]({{ s.corr_vif.corr_matrix_path }})
{% endif %}
{% if s.ts_diag and s.ts_diag.trend_seasonality_plots %}
## Time Series
![stl]({{ s.ts_diag.trend_seasonality_plots['stl'] }})
{% endif %}
## Recommendation
Primary: **{{ s.recommendation.primary_choice if s.recommendation else "n/a" }}**
""")
def build_report(state:RunState, out_dir:str)->str:
  os.makedirs(out_dir, exist_ok=True)
  p=os.path.join(out_dir, f"report_{state.run_id}.md")
  with open(p,"w") as f: f.write(T.render(s=state))
  return p
