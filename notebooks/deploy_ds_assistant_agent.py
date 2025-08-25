# Databricks notebook source
# MAGIC %md
# MAGIC # DS Assistant Agent Framework - Complete Deployment Guide
# MAGIC 
# MAGIC This notebook demonstrates the complete deployment of the DS Assistant Agent Framework that integrates:
# MAGIC - **Tool-calling LLM endpoints** on Databricks
# MAGIC - **Unity Catalog tables** with three-level namespace
# MAGIC - **MCP (Model Context Protocol)** tools for data science analysis
# MAGIC - **Intelligent orchestration** with LLM-powered decision making
# MAGIC - **Adaptive analysis** and modeling strategy recommendations
# MAGIC - **Convergence optimization** techniques
# MAGIC 
# MAGIC ## Architecture Overview
# MAGIC 
# MAGIC 1. **Unity Catalog Functions** → Registered as MCP tools
# MAGIC 2. **LangGraph Agent** → Orchestrates tool calling and reasoning
# MAGIC 3. **MLflow Model** → Deployed for production inference
# MAGIC 4. **Databricks Agent Framework** → End-to-end deployment

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv ds-assistant
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure Your Deployment
# MAGIC 
# MAGIC Customize these settings for your environment:

# COMMAND ----------

# Configuration - Customize these for your environment
CONFIG = {
    "llm_endpoint_name": "databricks-claude-3-7-sonnet",  # Your LLM endpoint
    "catalog": "main",                                    # Your Unity Catalog catalog
    "schema": "analytics",                                # Your Unity Catalog schema
    "model_name": "ds_assistant_agent",                   # Name for your deployed model
    "auto_create_functions": True                         # Whether to auto-create UC functions
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create the DS Assistant Agent Framework
# MAGIC 
# MAGIC The framework will automatically generate the SQL needed to create Unity Catalog functions.

# COMMAND ----------

from ds_assistant.agents.ds_agent_framework import create_ds_assistant_agent

# Create the agent with your configuration
agent = create_ds_assistant_agent(
    llm_endpoint_name=CONFIG["llm_endpoint_name"],
    catalog=CONFIG["catalog"],
    schema=CONFIG["schema"],
    auto_create_functions=CONFIG["auto_create_functions"]
)

print(f"✅ DS Assistant Agent created successfully!")
print(f"   Catalog: {CONFIG['catalog']}")
print(f"   Schema: {CONFIG['schema']}")
print(f"   LLM Endpoint: {CONFIG['llm_endpoint_name']}")
print(f"   Intelligent Orchestration: Enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Execute UC Function Creation (if auto_create_functions=True)
# MAGIC 
# MAGIC If you set `auto_create_functions=True`, the framework will have printed the SQL commands needed to create the Unity Catalog functions. Execute them below:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Execute the SQL commands printed by the framework above
# MAGIC -- This creates the schema and all required UC functions
# MAGIC 
# MAGIC -- Create the schema
# MAGIC CREATE SCHEMA IF NOT EXISTS main.analytics;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Main comprehensive pre-checks function
# MAGIC CREATE OR REPLACE FUNCTION main.analytics.ds_prechecks(
# MAGIC     table_name STRING, 
# MAGIC     target STRING, 
# MAGIC     time_col STRING, 
# MAGIC     artifacts_dir STRING
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC AS $$
# MAGIC from ds_assistant.tools.data_connector import DataConnector
# MAGIC from ds_assistant.agents.orchestrator import run_prechecks
# MAGIC 
# MAGIC def main(table_name, target, time_col, artifacts_dir):
# MAGIC     # Load data from Unity Catalog table
# MAGIC     df, meta = DataConnector.load_uc_table(table_name)
# MAGIC     
# MAGIC     # Run comprehensive pre-checks
# MAGIC     state = run_prechecks(
# MAGIC         df, 
# MAGIC         table_name, 
# MAGIC         target if target != "null" else None, 
# MAGIC         time_col if time_col != "null" else None, 
# MAGIC         artifacts_dir
# MAGIC     )
# MAGIC     
# MAGIC     # Return report path
# MAGIC     return f"{artifacts_dir}/report_{state.run_id}.md"
# MAGIC 
# MAGIC return main(table_name, target, time_col, artifacts_dir)
# MAGIC $$;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Data profiling function
# MAGIC CREATE OR REPLACE FUNCTION main.analytics.data_profiling(
# MAGIC     table_name STRING,
# MAGIC     artifacts_dir STRING
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC AS $$
# MAGIC from ds_assistant.tools.data_connector import DataConnector
# MAGIC from ds_assistant.tools.data_profiling import basic_profile, scale_check
# MAGIC import json
# MAGIC 
# MAGIC def main(table_name, artifacts_dir):
# MAGIC     # Load data
# MAGIC     df, meta = DataConnector.load_uc_table(table_name)
# MAGIC     
# MAGIC     # Perform basic profiling
# MAGIC     basic_profile_result = basic_profile(df, artifacts_dir)
# MAGIC     scale_check_result = scale_check(df)
# MAGIC     
# MAGIC     # Combine results
# MAGIC     profile_summary = {
# MAGIC         "shape": {
# MAGIC             "rows": basic_profile_result.shape.rows,
# MAGIC             "cols": basic_profile_result.shape.cols
# MAGIC         },
# MAGIC         "missing_data": {
# MAGIC             "columns_with_missing": [
# MAGIC                 {"col": m.col, "missing_rate": m.missing_rate} 
# MAGIC                 for m in basic_profile_result.missing 
# MAGIC                 if m.missing_rate > 0
# MAGIC             ],
# MAGIC             "overall_missing_rate": sum(m.missing_rate for m in basic_profile_result.missing) / len(basic_profile_result.missing)
# MAGIC         },
# MAGIC         "numerical_features": len(basic_profile_result.numerical_summary),
# MAGIC         "categorical_features": len(basic_profile_result.categorical_cardinality),
# MAGIC         "requires_scaling": scale_check_result.requires_scaling,
# MAGIC         "high_cardinality_features": [
# MAGIC             c.col for c in basic_profile_result.categorical_cardinality 
# MAGIC             if c.n_unique > 50
# MAGIC         ]
# MAGIC     }
# MAGIC     
# MAGIC     return json.dumps(profile_summary, indent=2)
# MAGIC 
# MAGIC return main(table_name, artifacts_dir)
# MAGIC $$;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Feature analysis function
# MAGIC CREATE OR REPLACE FUNCTION main.analytics.feature_analysis(
# MAGIC     table_name STRING,
# MAGIC     target STRING,
# MAGIC     artifacts_dir STRING
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC AS $$
# MAGIC from ds_assistant.tools.data_connector import DataConnector
# MAGIC from ds_assistant.tools.relationships import corr_and_vif, outliers
# MAGIC import json
# MAGIC 
# MAGIC def main(table_name, target, artifacts_dir):
# MAGIC     # Load data
# MAGIC     df, meta = DataConnector.load_uc_table(table_name)
# MAGIC     
# MAGIC     # Perform correlation and VIF analysis
# MAGIC     corr_vif_result = corr_and_vif(df, artifacts_dir, target=target if target != "null" else None)
# MAGIC     
# MAGIC     # Perform outlier analysis
# MAGIC     outliers_result = outliers(df, method="iqr")
# MAGIC     
# MAGIC     # Combine results
# MAGIC     analysis_summary = {
# MAGIC         "correlation_analysis": {
# MAGIC             "high_correlations": [
# MAGIC                 {"col1": pair.col1, "col2": pair.col2, "correlation": pair.corr}
# MAGIC                 for pair in corr_vif_result.top_pairwise_corrs[:10]  # Top 10
# MAGIC             ],
# MAGIC             "multicollinearity_detected": corr_vif_result.multicollinearity_flag,
# MAGIC             "high_vif_features": [
# MAGIC                 {"col": item.col, "vif": item.vif}
# MAGIC                 for item in corr_vif_result.vif
# MAGIC                 if item.vif > 5  # VIF threshold
# MAGIC             ]
# MAGIC         },
# MAGIC         "outlier_analysis": {
# MAGIC             "global_outlier_percentage": outliers_result.global_outlier_pct,
# MAGIC             "columns_with_outliers": [
# MAGIC                 {"col": col, "outlier_pct": pct}
# MAGIC                 for col_data in outliers_result.per_col_outlier_pct
# MAGIC                 for col, pct in col_data.items()
# MAGIC                 if pct > 5  # Outlier threshold
# MAGIC             ],
# MAGIC             "suggest_capping": outliers_result.suggest_capping
# MAGIC         }
# MAGIC     }
# MAGIC     
# MAGIC     return json.dumps(analysis_summary, indent=2)
# MAGIC 
# MAGIC return main(table_name, target, artifacts_dir)
# MAGIC $$;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Time series analysis function
# MAGIC CREATE OR REPLACE FUNCTION main.analytics.time_series_analysis(
# MAGIC     table_name STRING,
# MAGIC     target STRING,
# MAGIC     time_col STRING,
# MAGIC     artifacts_dir STRING
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC AS $$
# MAGIC from ds_assistant.tools.data_connector import DataConnector
# MAGIC from ds_assistant.tools.timeseries import stl_adf_acf_pacf
# MAGIC import json
# MAGIC 
# MAGIC def main(table_name, target, time_col, artifacts_dir):
# MAGIC     # Load data
# MAGIC     df, meta = DataConnector.load_uc_table(table_name)
# MAGIC     
# MAGIC     # Perform time series analysis
# MAGIC     ts_result = stl_adf_acf_pacf(
# MAGIC         df, 
# MAGIC         target if target != "null" else "", 
# MAGIC         time_col, 
# MAGIC         artifacts_dir, 
# MAGIC         None
# MAGIC     )
# MAGIC     
# MAGIC     # Create summary
# MAGIC     ts_summary = {
# MAGIC         "is_time_series": True,
# MAGIC         "stationarity": {
# MAGIC             "is_stationary": ts_result.adf.get("stationary", False) if ts_result.adf else False,
# MAGIC             "adf_statistic": ts_result.adf.get("adf_stat", None) if ts_result.adf else None,
# MAGIC             "p_value": ts_result.adf.get("p_value", None) if ts_result.adf else None
# MAGIC         },
# MAGIC         "seasonality": {
# MAGIC             "detected": ts_result.seasonality_detected,
# MAGIC             "frequency_guess": ts_result.freq_guess
# MAGIC         },
# MAGIC         "missing_timestamps": {
# MAGIC             "percentage": ts_result.missing_timestamps_pct
# MAGIC         },
# MAGIC         "external_variables": {
# MAGIC             "recommended": ts_result.exog_recommended,
# MAGIC             "variables": ts_result.exog if ts_result.exog else []
# MAGIC         }
# MAGIC     }
# MAGIC     
# MAGIC     return json.dumps(ts_summary, indent=2)
# MAGIC 
# MAGIC return main(table_name, target, time_col, artifacts_dir)
# MAGIC $$;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test the Agent

# COMMAND ----------

# Test the agent with a sample query
test_response = agent.predict({
    "messages": [
        {
            "role": "user", 
            "content": f"Analyze the table '{CONFIG['catalog']}.{CONFIG['schema']}.sample_table' and recommend the best modeling strategy for predicting 'target_column'. Include intelligent insights, convergence optimization, and risk assessment."
        }
    ]
})

print("Agent Response:")
for message in test_response.messages:
    if message.role == "assistant":
        print(message.content[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test Intelligent Analysis Workflow
# MAGIC 
# MAGIC Test the intelligent orchestration capabilities:

# COMMAND ----------

# Test the intelligent analysis workflow
print("🧠 Testing Intelligent Analysis Workflow...")

# Run the intelligent analysis
analysis_state = agent.run_analysis(
    dataset_ref=f"{CONFIG['catalog']}.{CONFIG['schema']}.sample_table",
    target="target_column",
    time_col=None,
    artifacts_dir="/tmp/intelligent_analysis"
)

print(f"✅ Intelligent Analysis Completed!")
print(f"   Run ID: {analysis_state['run_id']}")
print(f"   Completed Steps: {analysis_state['completed_steps']}")
print(f"   Quality Score: {analysis_state['data_quality_score']:.1f}/100")
print(f"   Confidence Score: {analysis_state['confidence_score']:.2f}")
print(f"   Errors: {len(analysis_state['errors'])}")
print(f"   Warnings: {len(analysis_state['warnings'])}")

# Show reasoning trail
print("\n🤔 Decision Reasoning Trail:")
for i, reasoning in enumerate(analysis_state['reasoning'], 1):
    print(f"   {i}. {reasoning}")

# Show final recommendation
if analysis_state['final_recommendation']:
    print(f"\n🎯 Final Recommendation:")
    print(f"   {analysis_state['final_recommendation']['recommendation'][:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Deploy the Agent

# COMMAND ----------

from ds_assistant.agents.ds_agent_framework import deploy_ds_assistant_agent

# Deploy the agent using the framework
deployment_info = deploy_ds_assistant_agent(
    agent=agent,
    model_name=CONFIG["model_name"],
    catalog=CONFIG["catalog"],
    schema=CONFIG["schema"],
    llm_endpoint_name=CONFIG["llm_endpoint_name"]
)

print("🎉 Deployment successful!")
print(f"Model URI: {deployment_info['model_uri']}")
print(f"UC Model Name: {deployment_info['uc_model_name']}")
print(f"Version: {deployment_info['version']}")
print(f"Deployment Info: {deployment_info['deployment_info']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test the Deployed Agent

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test with different scenarios
# MAGIC 
# MAGIC You can now test the deployed agent with various use cases:

# COMMAND ----------

# Example 1: Regression problem
regression_query = {
    "messages": [
        {
            "role": "user",
            "content": f"Perform intelligent analysis of '{CONFIG['catalog']}.{CONFIG['schema']}.sales_data' for predicting 'revenue'. Include adaptive feature engineering, convergence optimization, and comprehensive risk assessment."
        }
    ]
}

# Example 2: Classification problem  
classification_query = {
    "messages": [
        {
            "role": "user", 
            "content": f"Analyze '{CONFIG['catalog']}.{CONFIG['schema']}.customer_data' for predicting 'churn' with intelligent class imbalance handling, feature selection, and model convergence strategies."
        }
    ]
}

# Example 3: Time series problem
timeseries_query = {
    "messages": [
        {
            "role": "user",
            "content": f"Perform intelligent time series analysis of '{CONFIG['catalog']}.{CONFIG['schema']}.time_series_data' with time column 'timestamp' for predicting 'demand'. Include adaptive seasonality detection, stationarity analysis, and forecasting strategy optimization."
        }
    ]
}

print("Intelligent Analysis Examples Ready:")
print("1. 🧠 Intelligent Regression: Revenue prediction with adaptive features")
print("2. 🎯 Smart Classification: Churn prediction with imbalance handling") 
print("3. 📈 Adaptive Time Series: Demand forecasting with intelligent seasonality")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC 🎉 **DS Assistant Agent Framework Successfully Deployed!**
# MAGIC 
# MAGIC ### What We Built:
# MAGIC 
# MAGIC 1. **🧠 Intelligent Orchestration** → LLM-powered decision making and routing
# MAGIC 2. **🤖 LangGraph Agent** → Tool-calling LLM with intelligent reasoning
# MAGIC 3. **📊 Adaptive Analysis** → Dynamic workflows based on data characteristics
# MAGIC 4. **🚀 Production Deployment** → MLflow + Databricks Agent Framework
# MAGIC 
# MAGIC ### Key Features:
# MAGIC 
# MAGIC - **Intelligent Routing**: AI decides next analysis step based on current state
# MAGIC - **Adaptive Workflows**: Dynamic analysis sequences based on data characteristics
# MAGIC - **Error Recovery**: Intelligent retry and recovery strategies
# MAGIC - **Confidence Scoring**: Quality and confidence metrics for recommendations
# MAGIC - **Risk Assessment**: Comprehensive risk analysis and mitigation strategies
# MAGIC - **Convergence Optimization**: Advanced strategies for successful learning
# MAGIC - **Parallel Execution**: Concurrent analysis using LangGraph ToolNode
# MAGIC - **Production Ready**: MLflow integration and Unity Catalog deployment
# MAGIC 
# MAGIC ### Usage:
# MAGIC 
# MAGIC The agent can now analyze any Unity Catalog table and provide:
# MAGIC - **Problem classification** (regression/classification/forecasting)
# MAGIC - **Model family recommendations** with reasoning
# MAGIC - **Preprocessing strategies** for optimal convergence
# MAGIC - **Evaluation metrics** and tuning guidance
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC 
# MAGIC 1. **Test with real data** in your Unity Catalog
# MAGIC 2. **Customize system prompts** for domain-specific expertise
# MAGIC 3. **Add more MCP tools** for specialized analysis
# MAGIC 4. **Monitor and evaluate** agent performance
# MAGIC 5. **Scale to multiple tables** and use cases
# MAGIC 
# MAGIC The framework is now ready for production use! 🚀
