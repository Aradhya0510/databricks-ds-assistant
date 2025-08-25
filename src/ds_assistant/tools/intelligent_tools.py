"""
Enhanced LangChain Tools for DS Assistant
Provides intelligent, LLM-powered data science analysis tools
"""

from typing import Optional, Dict, Any, List
from langchain_core.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
import polars as pl
import json

from .data_connector import DataConnector
from .data_profiling import basic_profile, scale_check
from .relationships import corr_and_vif, outliers
from .timeseries import stl_adf_acf_pacf
from .recommender import recommend
from ..common.schemas import ProblemSummary

# Enhanced tool schemas
class DataProfilingInput(BaseModel):
    table_name: str = Field(description="Unity Catalog table name (catalog.schema.table)")
    artifacts_dir: str = Field(description="Directory to save artifacts")
    include_insights: bool = Field(default=True, description="Include LLM insights in results")

class FeatureAnalysisInput(BaseModel):
    table_name: str = Field(description="Unity Catalog table name")
    target: Optional[str] = Field(default=None, description="Target variable name")
    artifacts_dir: str = Field(description="Directory to save artifacts")
    correlation_threshold: float = Field(default=0.7, description="Threshold for high correlations")
    vif_threshold: float = Field(default=5.0, description="Threshold for high VIF")

class TimeSeriesAnalysisInput(BaseModel):
    table_name: str = Field(description="Unity Catalog table name")
    target: str = Field(description="Target variable name")
    time_col: str = Field(description="Time column name")
    artifacts_dir: str = Field(description="Directory to save artifacts")
    include_forecasting: bool = Field(default=True, description="Include forecasting recommendations")

class ModelRecommendationInput(BaseModel):
    problem_summary_json: str = Field(description="JSON string of problem summary")
    include_convergence: bool = Field(default=True, description="Include convergence optimization")
    include_risk_assessment: bool = Field(default=True, description="Include risk assessment")

@tool
def data_profiling(input_data: DataProfilingInput) -> str:
    """
    Perform comprehensive data profiling with intelligent insights.
    
    This tool analyzes dataset characteristics including:
    - Basic statistics (shape, data types, missing values)
    - Distribution analysis for numerical and categorical features
    - Data quality assessment
    - Scaling requirements detection
    - LLM-powered insights and recommendations
    """
    
    try:
        # Load data
        df, meta = DataConnector.load_uc_table(input_data.table_name)
        
        # Perform profiling
        profile_result = basic_profile(df, input_data.artifacts_dir)
        scale_result = scale_check(df)
        
        # Create comprehensive result
        result = {
            "table_name": input_data.table_name,
            "metadata": meta,
            "profile": profile_result.model_dump(),
            "scale_check": scale_result.model_dump(),
            "quality_metrics": {
                "missing_data_pct": sum(m.missing_rate for m in profile_result.missing) / len(profile_result.missing),
                "high_cardinality_features": len([c for c in profile_result.categorical_cardinality if c.n_unique > 50]),
                "requires_scaling": scale_result.requires_scaling,
                "data_quality_score": _calculate_quality_score(profile_result)
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Data profiling failed: {str(e)}",
            "table_name": input_data.table_name
        })

@tool
def feature_analysis(input_data: FeatureAnalysisInput) -> str:
    """
    Perform intelligent feature analysis with correlation and multicollinearity detection.
    
    This tool analyzes:
    - Feature correlations and relationships
    - Multicollinearity using VIF analysis
    - Outlier detection and analysis
    - Feature importance insights
    - Intelligent recommendations for feature engineering
    """
    
    try:
        # Load data
        df, meta = DataConnector.load_uc_table(input_data.table_name)
        
        # Perform analysis
        corr_result = corr_and_vif(df, input_data.artifacts_dir, target=input_data.target)
        outlier_result = outliers(df, method="iqr")
        
        # Create comprehensive result
        result = {
            "table_name": input_data.table_name,
            "target": input_data.target,
            "correlation_analysis": {
                "high_correlations": [
                    {"col1": pair.col1, "col2": pair.col2, "correlation": pair.corr}
                    for pair in corr_result.top_pairwise_corrs
                    if abs(pair.corr) >= input_data.correlation_threshold
                ],
                "multicollinearity_detected": corr_result.multicollinearity_flag,
                "high_vif_features": [
                    {"col": item.col, "vif": item.vif}
                    for item in corr_result.vif
                    if item.vif >= input_data.vif_threshold
                ]
            },
            "outlier_analysis": {
                "global_outlier_percentage": outlier_result.global_outlier_pct,
                "columns_with_outliers": [
                    {"col": col, "outlier_pct": pct}
                    for col_data in outlier_result.per_col_outlier_pct
                    for col, pct in col_data.items()
                    if pct > 5  # Outlier threshold
                ],
                "suggest_capping": outlier_result.suggest_capping
            },
            "feature_engineering_recommendations": _generate_feature_recommendations(
                corr_result, outlier_result, input_data.target
            )
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Feature analysis failed: {str(e)}",
            "table_name": input_data.table_name
        })

@tool
def time_series_analysis(input_data: TimeSeriesAnalysisInput) -> str:
    """
    Perform comprehensive time series analysis with forecasting insights.
    
    This tool analyzes:
    - Time series characteristics and patterns
    - Seasonality detection and analysis
    - Stationarity testing
    - Missing timestamp analysis
    - Forecasting approach recommendations
    """
    
    try:
        # Load data
        df, meta = DataConnector.load_uc_table(input_data.table_name)
        
        # Perform time series analysis
        ts_result = stl_adf_acf_pacf(
            df, input_data.target, input_data.time_col, 
            input_data.artifacts_dir, None
        )
        
        # Create comprehensive result
        result = {
            "table_name": input_data.table_name,
            "target": input_data.target,
            "time_column": input_data.time_col,
            "time_series_characteristics": {
                "is_stationary": ts_result.adf.get("stationary", False) if ts_result.adf else False,
                "adf_statistic": ts_result.adf.get("adf_stat", None) if ts_result.adf else None,
                "p_value": ts_result.adf.get("p_value", None) if ts_result.adf else None,
                "seasonality_detected": ts_result.seasonality_detected,
                "frequency_guess": ts_result.freq_guess,
                "missing_timestamps_pct": ts_result.missing_timestamps_pct
            },
            "forecasting_recommendations": _generate_forecasting_recommendations(ts_result) if input_data.include_forecasting else None,
            "preprocessing_recommendations": _generate_timeseries_preprocessing(ts_result)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Time series analysis failed: {str(e)}",
            "table_name": input_data.table_name
        })

@tool
def model_recommendation(input_data: ModelRecommendationInput) -> str:
    """
    Generate intelligent model recommendations with convergence optimization.
    
    This tool provides:
    - Model family recommendations with reasoning
    - Preprocessing strategies
    - Hyperparameter tuning approaches
    - Evaluation metrics selection
    - Convergence optimization techniques
    - Risk assessment and mitigation
    """
    
    try:
        # Parse problem summary
        summary_dict = json.loads(input_data.problem_summary_json)
        summary = ProblemSummary(**summary_dict)
        
        # Get base recommendations
        base_recommendation = recommend(summary)
        
        # Create enhanced result
        result = {
            "problem_summary": summary_dict,
            "base_recommendations": base_recommendation.model_dump(),
            "enhanced_recommendations": {
                "primary_model": base_recommendation.primary_choice,
                "all_candidates": [
                    {
                        "model_family": candidate.model_family,
                        "reasoning": candidate.why,
                        "preprocessing": candidate.preprocessing,
                        "tuning_notes": candidate.tuning_notes,
                        "eval_metrics": candidate.eval_metrics
                    }
                    for candidate in base_recommendation.candidates
                ]
            }
        }
        
        # Add convergence optimization if requested
        if input_data.include_convergence:
            result["convergence_optimization"] = _generate_convergence_strategy(summary)
        
        # Add risk assessment if requested
        if input_data.include_risk_assessment:
            result["risk_assessment"] = _generate_risk_assessment(summary)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Model recommendation failed: {str(e)}",
            "problem_summary": input_data.problem_summary_json
        })

@tool
def intelligent_data_assessment(table_name: str, target: Optional[str] = None, time_col: Optional[str] = None) -> str:
    """
    Perform intelligent data assessment to determine analysis strategy.
    
    This tool provides:
    - Problem type classification
    - Data quality assessment
    - Analysis strategy recommendations
    - Risk identification
    - Resource requirements estimation
    """
    
    try:
        # Load data
        df, meta = DataConnector.load_uc_table(table_name)
        
        # Perform quick assessment
        assessment = {
            "table_name": table_name,
            "data_characteristics": {
                "shape": {"rows": meta["rows"], "cols": meta["cols"]},
                "target_variable": target,
                "time_column": time_col,
                "problem_type": _infer_problem_type(df, target),
                "data_types": _analyze_data_types(df)
            },
            "quality_indicators": {
                "missing_data": _assess_missing_data(df),
                "outliers": _assess_outliers(df),
                "cardinality": _assess_cardinality(df)
            },
            "analysis_strategy": _recommend_analysis_strategy(df, target, time_col),
            "risk_assessment": _assess_risks(df, target),
            "resource_requirements": _estimate_resources(df)
        }
        
        return json.dumps(assessment, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Data assessment failed: {str(e)}",
            "table_name": table_name
        })

# Helper functions for enhanced insights
def _calculate_quality_score(profile_result) -> float:
    """Calculate data quality score based on profiling results"""
    score = 100.0
    
    # Penalize for missing data
    missing_rates = [m.missing_rate for m in profile_result.missing]
    avg_missing = sum(missing_rates) / len(missing_rates) if missing_rates else 0
    score -= avg_missing * 50
    
    # Penalize for high cardinality
    high_cardinality = [c for c in profile_result.categorical_cardinality if c.n_unique > 50]
    score -= len(high_cardinality) * 5
    
    return max(0, min(100, score))

def _generate_feature_recommendations(corr_result, outlier_result, target: Optional[str]) -> Dict[str, Any]:
    """Generate intelligent feature engineering recommendations"""
    recommendations = {
        "feature_selection": [],
        "feature_engineering": [],
        "outlier_handling": [],
        "encoding_strategies": []
    }
    
    # Feature selection based on correlations
    if corr_result.multicollinearity_flag:
        recommendations["feature_selection"].append({
            "action": "Remove high VIF features",
            "features": [item.col for item in corr_result.vif if item.vif > 5],
            "reasoning": "High multicollinearity detected"
        })
    
    # Outlier handling
    if outlier_result.global_outlier_pct > 10:
        recommendations["outlier_handling"].append({
            "action": "Apply outlier capping",
            "method": "IQR-based capping",
            "reasoning": f"High outlier percentage: {outlier_result.global_outlier_pct:.1f}%"
        })
    
    return recommendations

def _generate_forecasting_recommendations(ts_result) -> Dict[str, Any]:
    """Generate forecasting approach recommendations"""
    recommendations = {
        "model_families": [],
        "preprocessing": [],
        "evaluation": []
    }
    
    # Model recommendations based on characteristics
    if ts_result.seasonality_detected:
        recommendations["model_families"].append({
            "family": "SARIMAX/Prophet",
            "reasoning": "Seasonality detected in data"
        })
    else:
        recommendations["model_families"].append({
            "family": "ARIMA/Exponential Smoothing",
            "reasoning": "No clear seasonality detected"
        })
    
    # Preprocessing recommendations
    if not ts_result.adf.get("stationary", False):
        recommendations["preprocessing"].append({
            "action": "Apply differencing",
            "reasoning": "Non-stationary time series detected"
        })
    
    return recommendations

def _generate_timeseries_preprocessing(ts_result) -> List[Dict[str, str]]:
    """Generate time series preprocessing recommendations"""
    preprocessing = []
    
    if ts_result.missing_timestamps_pct > 5:
        preprocessing.append({
            "action": "Handle missing timestamps",
            "method": "Forward fill or interpolation",
            "reasoning": f"Missing timestamps: {ts_result.missing_timestamps_pct:.1f}%"
        })
    
    if not ts_result.adf.get("stationary", False):
        preprocessing.append({
            "action": "Make series stationary",
            "method": "Differencing or transformation",
            "reasoning": "Non-stationary series detected"
        })
    
    return preprocessing

def _generate_convergence_strategy(summary: ProblemSummary) -> Dict[str, Any]:
    """Generate convergence optimization strategies"""
    strategy = {
        "scaling": [],
        "regularization": [],
        "optimization": [],
        "monitoring": []
    }
    
    if summary.requires_scaling:
        strategy["scaling"].append({
            "method": "StandardScaler",
            "reasoning": "Features require scaling for convergence"
        })
    
    if summary.multicollinearity_flag:
        strategy["regularization"].append({
            "method": "L1/L2 regularization",
            "reasoning": "Multicollinearity detected"
        })
    
    return strategy

def _generate_risk_assessment(summary: ProblemSummary) -> Dict[str, Any]:
    """Generate risk assessment for model deployment"""
    risks = {
        "data_quality_risks": [],
        "model_risks": [],
        "deployment_risks": [],
        "mitigation_strategies": []
    }
    
    if summary.missing_overall_pct > 0.1:
        risks["data_quality_risks"].append({
            "risk": "High missing data",
            "impact": "Model performance degradation",
            "probability": "High"
        })
    
    if summary.outliers_heavy:
        risks["data_quality_risks"].append({
            "risk": "Heavy outliers",
            "impact": "Model instability",
            "probability": "Medium"
        })
    
    return risks

def _infer_problem_type(df: pl.DataFrame, target: Optional[str]) -> str:
    """Infer problem type from data"""
    if not target:
        return "unsupervised"
    
    t = df[target]
    if str(t.dtype).startswith(("i", "u", "f")):
        return "regression"
    
    unique_count = int(t.n_unique())
    return "binary" if unique_count == 2 else "multiclass"

def _analyze_data_types(df: pl.DataFrame) -> Dict[str, int]:
    """Analyze data types distribution"""
    numeric_count = len(df.select(pl.col('*').filter(pl.col('*').is_numeric())).columns)
    categorical_count = len(df.columns) - numeric_count
    
    return {
        "numeric": numeric_count,
        "categorical": categorical_count,
        "total": len(df.columns)
    }

def _assess_missing_data(df: pl.DataFrame) -> Dict[str, Any]:
    """Assess missing data patterns"""
    missing_counts = df.null_count()
    missing_pct = (missing_counts / len(df)) * 100
    
    return {
        "columns_with_missing": len([pct for pct in missing_pct if pct > 0]),
        "max_missing_pct": float(missing_pct.max()),
        "avg_missing_pct": float(missing_pct.mean())
    }

def _assess_outliers(df: pl.DataFrame) -> Dict[str, Any]:
    """Quick outlier assessment"""
    numeric_cols = df.select(pl.col('*').filter(pl.col('*').is_numeric())).columns
    
    if not numeric_cols:
        return {"outlier_risk": "low", "reasoning": "No numeric columns"}
    
    # Simple outlier detection using IQR
    outlier_columns = []
    for col in numeric_cols[:5]:  # Check first 5 numeric columns
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df.filter((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr))
        if len(outliers) > len(df) * 0.05:  # More than 5% outliers
            outlier_columns.append(col)
    
    return {
        "outlier_risk": "high" if outlier_columns else "low",
        "columns_with_outliers": len(outlier_columns),
        "reasoning": f"Found outliers in {len(outlier_columns)} columns"
    }

def _assess_cardinality(df: pl.DataFrame) -> Dict[str, Any]:
    """Assess categorical feature cardinality"""
    categorical_cols = df.select(pl.col('*').filter(~pl.col('*').is_numeric())).columns
    
    high_cardinality = []
    for col in categorical_cols:
        unique_count = df[col].n_unique()
        if unique_count > 50:
            high_cardinality.append(col)
    
    return {
        "high_cardinality_features": len(high_cardinality),
        "max_cardinality": max([df[col].n_unique() for col in categorical_cols]) if categorical_cols else 0,
        "encoding_challenge": "high" if high_cardinality else "low"
    }

def _recommend_analysis_strategy(df: pl.DataFrame, target: Optional[str], time_col: Optional[str]) -> Dict[str, Any]:
    """Recommend analysis strategy based on data characteristics"""
    strategy = {
        "required_analyses": [],
        "optional_analyses": [],
        "priority": "medium"
    }
    
    # Always include data profiling
    strategy["required_analyses"].append("data_profiling")
    
    # Feature analysis if multiple features
    if len(df.columns) > 3:
        strategy["required_analyses"].append("feature_analysis")
    
    # Time series analysis if time column present
    if time_col:
        strategy["required_analyses"].append("time_series_analysis")
    
    # Quality assessment
    strategy["required_analyses"].append("quality_assessment")
    
    # Model recommendation
    strategy["required_analyses"].append("model_recommendation")
    
    return strategy

def _assess_risks(df: pl.DataFrame, target: Optional[str]) -> Dict[str, Any]:
    """Assess risks for modeling"""
    risks = {
        "data_quality_risks": [],
        "modeling_risks": [],
        "overall_risk": "low"
    }
    
    # Data quality risks
    missing_pct = (df.null_count() / len(df)).mean() * 100
    if missing_pct > 10:
        risks["data_quality_risks"].append("High missing data")
    
    # Modeling risks
    if target and target in df.columns:
        if df[target].n_unique() < 10:
            risks["modeling_risks"].append("Low cardinality target")
    
    # Overall risk assessment
    total_risks = len(risks["data_quality_risks"]) + len(risks["modeling_risks"])
    if total_risks > 3:
        risks["overall_risk"] = "high"
    elif total_risks > 1:
        risks["overall_risk"] = "medium"
    
    return risks

def _estimate_resources(df: pl.DataFrame) -> Dict[str, Any]:
    """Estimate computational resources needed"""
    return {
        "memory_estimate_mb": len(df) * len(df.columns) * 8 / (1024 * 1024),  # Rough estimate
        "computation_time_minutes": max(5, len(df.columns) * 2),  # Rough estimate
        "storage_requirements_mb": len(df) * len(df.columns) * 8 / (1024 * 1024)
    }

# Export all intelligent tools
INTELLIGENT_TOOLS = [
    data_profiling,
    feature_analysis,
    time_series_analysis,
    model_recommendation,
    intelligent_data_assessment
]
