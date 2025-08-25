-- Unity Catalog Functions for DS Assistant Agent Framework
-- These functions will be registered as MCP tools for the agent to use

-- Main comprehensive pre-checks function
CREATE OR REPLACE FUNCTION main.analytics.ds_prechecks(
    table_name STRING, 
    target STRING, 
    time_col STRING, 
    artifacts_dir STRING
)
RETURNS STRING
LANGUAGE PYTHON
AS $$
from ds_assistant.tools.data_connector import DataConnector
from ds_assistant.agents.orchestrator import run_prechecks

def main(table_name, target, time_col, artifacts_dir):
    # Load data from Unity Catalog table
    df, meta = DataConnector.load_uc_table(table_name)
    
    # Run comprehensive pre-checks
    state = run_prechecks(
        df, 
        table_name, 
        target if target != "null" else None, 
        time_col if time_col != "null" else None, 
        artifacts_dir
    )
    
    # Return report path
    return f"{artifacts_dir}/report_{state.run_id}.md"

return main(table_name, target, time_col, artifacts_dir)
$$;

-- Data profiling function
CREATE OR REPLACE FUNCTION main.analytics.data_profiling(
    table_name STRING,
    artifacts_dir STRING
)
RETURNS STRING
LANGUAGE PYTHON
AS $$
from ds_assistant.tools.data_connector import DataConnector
from ds_assistant.tools.data_profiling import basic_profile, scale_check
import json

def main(table_name, artifacts_dir):
    # Load data
    df, meta = DataConnector.load_uc_table(table_name)
    
    # Perform basic profiling
    basic_profile_result = basic_profile(df, artifacts_dir)
    scale_check_result = scale_check(df)
    
    # Combine results
    profile_summary = {
        "shape": {
            "rows": basic_profile_result.shape.rows,
            "cols": basic_profile_result.shape.cols
        },
        "missing_data": {
            "columns_with_missing": [
                {"col": m.col, "missing_rate": m.missing_rate} 
                for m in basic_profile_result.missing 
                if m.missing_rate > 0
            ],
            "overall_missing_rate": sum(m.missing_rate for m in basic_profile_result.missing) / len(basic_profile_result.missing)
        },
        "numerical_features": len(basic_profile_result.numerical_summary),
        "categorical_features": len(basic_profile_result.categorical_cardinality),
        "requires_scaling": scale_check_result.requires_scaling,
        "high_cardinality_features": [
            c.col for c in basic_profile_result.categorical_cardinality 
            if c.n_unique > 50
        ]
    }
    
    return json.dumps(profile_summary, indent=2)

return main(table_name, artifacts_dir)
$$;

-- Feature analysis function
CREATE OR REPLACE FUNCTION main.analytics.feature_analysis(
    table_name STRING,
    target STRING,
    artifacts_dir STRING
)
RETURNS STRING
LANGUAGE PYTHON
AS $$
from ds_assistant.tools.data_connector import DataConnector
from ds_assistant.tools.relationships import corr_and_vif, outliers
import json

def main(table_name, target, artifacts_dir):
    # Load data
    df, meta = DataConnector.load_uc_table(table_name)
    
    # Perform correlation and VIF analysis
    corr_vif_result = corr_and_vif(df, artifacts_dir, target=target if target != "null" else None)
    
    # Perform outlier analysis
    outliers_result = outliers(df, method="iqr")
    
    # Combine results
    analysis_summary = {
        "correlation_analysis": {
            "high_correlations": [
                {"col1": pair.col1, "col2": pair.col2, "correlation": pair.corr}
                for pair in corr_vif_result.top_pairwise_corrs[:10]  # Top 10
            ],
            "multicollinearity_detected": corr_vif_result.multicollinearity_flag,
            "high_vif_features": [
                {"col": item.col, "vif": item.vif}
                for item in corr_vif_result.vif
                if item.vif > 5  # VIF threshold
            ]
        },
        "outlier_analysis": {
            "global_outlier_percentage": outliers_result.global_outlier_pct,
            "columns_with_outliers": [
                {"col": col, "outlier_pct": pct}
                for col_data in outliers_result.per_col_outlier_pct
                for col, pct in col_data.items()
                if pct > 5  # Outlier threshold
            ],
            "suggest_capping": outliers_result.suggest_capping
        }
    }
    
    return json.dumps(analysis_summary, indent=2)

return main(table_name, target, artifacts_dir)
$$;

-- Time series analysis function
CREATE OR REPLACE FUNCTION main.analytics.time_series_analysis(
    table_name STRING,
    target STRING,
    time_col STRING,
    artifacts_dir STRING
)
RETURNS STRING
LANGUAGE PYTHON
AS $$
from ds_assistant.tools.data_connector import DataConnector
from ds_assistant.tools.timeseries import stl_adf_acf_pacf
import json

def main(table_name, target, time_col, artifacts_dir):
    # Load data
    df, meta = DataConnector.load_uc_table(table_name)
    
    # Perform time series analysis
    ts_result = stl_adf_acf_pacf(
        df, 
        target if target != "null" else "", 
        time_col, 
        artifacts_dir, 
        None
    )
    
    # Create summary
    ts_summary = {
        "is_time_series": True,
        "stationarity": {
            "is_stationary": ts_result.adf.get("stationary", False) if ts_result.adf else False,
            "adf_statistic": ts_result.adf.get("adf_stat", None) if ts_result.adf else None,
            "p_value": ts_result.adf.get("p_value", None) if ts_result.adf else None
        },
        "seasonality": {
            "detected": ts_result.seasonality_detected,
            "frequency_guess": ts_result.freq_guess
        },
        "missing_timestamps": {
            "percentage": ts_result.missing_timestamps_pct
        },
        "external_variables": {
            "recommended": ts_result.exog_recommended,
            "variables": ts_result.exog if ts_result.exog else []
        }
    }
    
    return json.dumps(ts_summary, indent=2)

return main(table_name, target, time_col, artifacts_dir)
$$;

-- Model recommendation function
CREATE OR REPLACE FUNCTION main.analytics.model_recommendation(
    problem_summary_json STRING
)
RETURNS STRING
LANGUAGE PYTHON
AS $$
from ds_assistant.tools.recommender import recommend
from ds_assistant.common.schemas import ProblemSummary
import json

def main(problem_summary_json):
    # Parse problem summary
    summary_dict = json.loads(problem_summary_json)
    summary = ProblemSummary(**summary_dict)
    
    # Get recommendations
    recommendation = recommend(summary)
    
    # Format recommendations
    rec_summary = {
        "problem_type": summary.problem_type,
        "primary_recommendation": recommendation.primary_choice,
        "all_candidates": [
            {
                "model_family": candidate.model_family,
                "reasoning": candidate.why,
                "preprocessing_steps": candidate.preprocessing,
                "tuning_notes": candidate.tuning_notes,
                "evaluation_metrics": candidate.eval_metrics
            }
            for candidate in recommendation.candidates
        ],
        "convergence_strategy": {
            "scaling_required": summary.requires_scaling,
            "outlier_handling": summary.outliers_heavy,
            "multicollinearity_handling": summary.multicollinearity_flag,
            "missing_data_strategy": "imputation" if summary.missing_overall_pct > 0 else "none"
        }
    }
    
    return json.dumps(rec_summary, indent=2)

return main(problem_summary_json)
$$;
