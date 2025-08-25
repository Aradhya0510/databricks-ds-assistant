# DS Assistant Agent Framework

A comprehensive framework for deploying **intelligent, adaptive data science agents** on Databricks that automatically analyze Unity Catalog tables and recommend optimal modeling strategies for successful learning convergence. Built with **LangChain/LangGraph** for intelligent orchestration and LLM-powered decision making.

## 🎯 Overview

The DS Assistant Agent Framework integrates **tool-calling LLM endpoints** with **MCP (Model Context Protocol)** tools to provide automated data science analysis and intelligent modeling recommendations. It's designed to work seamlessly with Databricks Unity Catalog and the Databricks Agent Framework.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Unity Catalog │    │   LangGraph      │    │   MLflow Model  │
│   Functions     │───▶│   Agent          │───▶│   Deployment    │
│   (MCP Tools)   │    │   (Tool Calling) │    │   (Production)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Profiling│    │   Reasoning      │    │   Databricks    │
│   Feature Anal. │    │   & Planning     │    │   Agent         │
│   Time Series   │    │   (LLM)          │    │   Framework     │
│   Model Rec.    │    │                  │    │   (End-to-End)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Key Features

### 🧠 **Intelligent Orchestration**
- **LLM-powered Routing**: AI decides the next analysis step based on current state
- **Adaptive Workflows**: Dynamic analysis sequences based on data characteristics
- **Error Recovery**: Intelligent retry and recovery strategies
- **Confidence Scoring**: Quality and confidence metrics for recommendations

### 🔍 **Intelligent Analysis**
- **Intelligent Data Profiling**: LLM insights on data quality and characteristics
- **Advanced Feature Analysis**: Smart recommendations for feature engineering
- **Time Series Intelligence**: Adaptive forecasting approach recommendations
- **Risk Assessment**: Comprehensive risk analysis and mitigation strategies

### 🛠️ **Flexible Configuration**
- **Custom Catalog/Schema**: Specify your own Unity Catalog namespace
- **Auto UC Function Creation**: Dynamic SQL generation for MCP tools
- **Configurable LLM Endpoints**: Use any Databricks LLM endpoint
- **Intelligent Tools**: LangChain tools with LLM-powered insights

### 🏭 **Production Ready**
- **MLflow Integration**: Model versioning and deployment
- **Databricks Agent Framework**: End-to-end deployment
- **Unity Catalog**: Three-level namespace support
- **Parallel Execution**: Concurrent analysis using LangGraph ToolNode
- **Streaming Responses**: Real-time progress updates
- **Comprehensive Logging**: Full audit trail of decisions and reasoning

## 📁 Project Structure

```
ds-assistant-repo/
├── src/ds_assistant/
│   ├── agents/
│   │   ├── orchestrator.py          # Intelligent LangGraph orchestrator
│   │   └── ds_agent_framework.py    # Main framework with intelligent tools
│   ├── tools/
│   │   ├── __init__.py              # Exports intelligent tools as primary
│   │   ├── intelligent_tools.py     # Primary intelligent LangChain tools
│   │   ├── data_connector.py        # Data loading utilities
│   │   ├── data_profiling.py        # Legacy tools (internal use)
│   │   ├── relationships.py         # Legacy tools (internal use)
│   │   ├── timeseries.py            # Legacy tools (internal use)
│   │   ├── recommender.py           # Legacy tools (internal use)
│   │   └── report_builder.py        # Legacy tools (internal use)
│   └── common/
│       ├── schemas.py               # Pydantic models
│       ├── state.py                 # State management
│       └── util_io.py               # Utility functions
├── notebooks/
│   └── deploy_ds_assistant_agent.py # Complete deployment guide
├── examples/
│   ├── config.sample.yaml           # Configuration template
│   ├── uc_functions_setup.sql       # UC function definitions
│   └── usage_example.py             # Usage examples
└── scripts/
    └── build_wheel.sh               # Build utilities
```

## 🚀 Quick Start

### 1. Configure Your Deployment

```python
# Customize these settings for your environment
CONFIG = {
    "llm_endpoint_name": "databricks-claude-3-7-sonnet",  # Your LLM endpoint
    "catalog": "main",                                    # Your Unity Catalog catalog
    "schema": "analytics",                                # Your Unity Catalog schema
    "model_name": "ds_assistant_agent",                   # Name for your deployed model
    "auto_create_functions": True                         # Whether to auto-create UC functions
}
```

### 2. Create and Deploy the Agent

```python
from ds_assistant.agents.ds_agent_framework import create_ds_assistant_agent, deploy_ds_assistant_agent

# Create the agent with your configuration
agent = create_ds_assistant_agent(
    llm_endpoint_name=CONFIG["llm_endpoint_name"],
    catalog=CONFIG["catalog"],
    schema=CONFIG["schema"],
    auto_create_functions=CONFIG["auto_create_functions"]
)

# Deploy the agent
deployment_info = deploy_ds_assistant_agent(
    agent=agent,
    model_name=CONFIG["model_name"],
    catalog=CONFIG["catalog"],
    schema=CONFIG["schema"],
    llm_endpoint_name=CONFIG["llm_endpoint_name"]
)
```

### 3. Use the Agent

```python
# The agent can now analyze any Unity Catalog table
query = {
    "messages": [
        {
            "role": "user",
            "content": f"Analyze the table '{catalog}.{schema}.transactions' and recommend the best modeling strategy for predicting 'target_column'"
        }
    ]
}

# Agent will automatically:
# 1. Intelligently assess the data and decide analysis strategy
# 2. Perform adaptive analysis based on data characteristics
# 3. Provide LLM-powered insights and recommendations
# 4. Generate confidence scores and risk assessments
# 5. Optimize for convergence with advanced strategies
```

## 🔧 Intelligent Tools

The framework provides **LangChain tools** and **Unity Catalog functions** as MCP tools:

### **LangChain Tools (Intelligent)**
- **`data_profiling`**: Intelligent data profiling with LLM insights
- **`feature_analysis`**: Advanced feature analysis with smart recommendations
- **`time_series_analysis`**: Adaptive time series analysis with forecasting insights
- **`model_recommendation`**: Intelligent model recommendations with convergence optimization
- **`intelligent_data_assessment`**: LLM-powered data assessment and strategy planning

### **Unity Catalog Functions (Backward Compatible)**
- **`{catalog}.{schema}.ds_prechecks`**: Comprehensive analysis including all pre-checks
- **`{catalog}.{schema}.data_profiling`**: Basic data profiling with quality metrics
- **`{catalog}.{schema}.feature_analysis`**: Feature relationships and correlation analysis
- **`{catalog}.{schema}.time_series_analysis`**: Time series analysis with seasonality detection

## 📊 Intelligent Analysis Capabilities

### **Intelligent Data Assessment**
- **LLM-powered Analysis**: AI-driven data quality assessment and insights
- **Adaptive Strategy**: Dynamic analysis planning based on data characteristics
- **Risk Identification**: Comprehensive risk assessment and mitigation strategies
- **Resource Estimation**: Computational resource requirements and optimization

### **Advanced Data Quality Assessment**
- **Missing Data**: Intelligent detection and imputation strategies with reasoning
- **Outliers**: IQR-based detection with adaptive capping recommendations
- **Data Types**: Automatic detection with encoding strategy optimization
- **Cardinality**: High-cardinality feature handling with intelligent recommendations

### **Smart Feature Engineering**
- **Correlations**: Intelligent correlation detection with feature selection strategies
- **Multicollinearity**: VIF analysis with automated feature selection
- **Scaling**: Adaptive scaling detection with optimization strategies
- **Encoding**: Intelligent categorical encoding with performance optimization

### **Intelligent Model Selection**
- **Problem Type**: LLM-powered classification with confidence scoring
- **Algorithm Families**: Evidence-based recommendations with reasoning
- **Hyperparameter Tuning**: Advanced optimization strategies with convergence focus
- **Evaluation Metrics**: Adaptive metric selection based on problem characteristics

### **Advanced Convergence Optimization**
- **Scaling Strategies**: Intelligent scaling with performance optimization
- **Outlier Handling**: Adaptive outlier treatment with model stability focus
- **Feature Selection**: Intelligent feature selection with multicollinearity handling
- **Class Imbalance**: Advanced imbalance handling with performance optimization
- **Risk Mitigation**: Comprehensive risk assessment and mitigation strategies

## 🎯 Use Cases

### Regression Problems
```python
"Analyze '{catalog}.{schema}.sales_data' and recommend strategy for predicting 'revenue'"
```

### Classification Problems
```python
"Analyze '{catalog}.{schema}.customer_data' and recommend strategy for predicting 'churn'"
```

### Time Series Forecasting
```python
"Analyze '{catalog}.{schema}.time_series_data' with time column 'timestamp' and recommend forecasting strategy for 'demand'"
```

## 🔗 Integration with Databricks

### Unity Catalog Integration
- **Three-level namespace**: `catalog.schema.table`
- **Automatic authentication**: Passthrough from agent to tables
- **Scalable**: Handles large datasets with sampling
- **Flexible**: Custom catalog.schema configuration

### Databricks Agent Framework
- **Production deployment**: End-to-end agent deployment
- **MLflow integration**: Model versioning and tracking
- **Monitoring**: Built-in monitoring and evaluation

### MCP Integration
Based on [Databricks MCP documentation](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html):
- **Tool calling**: Seamless LLM-tool integration
- **LangGraph**: State management and workflow orchestration
- **Extensible**: Easy to add new tools and capabilities

## 📈 Performance

### Scalability
- **Large datasets**: Automatic sampling for analysis
- **Parallel processing**: Efficient tool execution
- **Memory optimization**: Streaming for large results

### Accuracy
- **Comprehensive analysis**: Multiple analysis dimensions
- **Evidence-based**: Data-driven recommendations
- **Domain expertise**: Specialized system prompts

## 🔮 Future Enhancements

### Planned Features
- **Custom model training**: Automated model training pipelines
- **Feature store integration**: Unity Catalog feature store support
- **Advanced analytics**: Deep learning and ensemble recommendations
- **Real-time analysis**: Streaming data analysis capabilities

### Extensibility
- **Custom tools**: Easy addition of domain-specific tools
- **Custom prompts**: Specialized system prompts for different domains
- **Custom metrics**: Domain-specific evaluation metrics

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Add your enhancements**
4. **Test with real data**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Databricks**: For the excellent Agent Framework and MCP integration
- **LangGraph**: For the powerful workflow orchestration
- **MLflow**: For the robust model management
- **MCP**: For the standardized tool calling protocol

---

**Ready to revolutionize your data science workflow? Deploy the DS Assistant Agent Framework today! 🚀**
