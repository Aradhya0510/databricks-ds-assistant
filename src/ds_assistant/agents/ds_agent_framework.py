"""
DS Assistant Agent Framework
Integrates tool-calling LLM endpoints with MCP tools for automated data science analysis
Now enhanced with LangChain/LangGraph for intelligent, adaptive orchestration
"""

from typing import Any, Generator, Optional, Sequence, Union, Dict, List
import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

# Import intelligent components
from .orchestrator import DSOrchestrator, DSState
from ..tools import INTELLIGENT_TOOLS

mlflow.langchain.autolog()

class DSAssistantAgentFramework:
    """
    Enhanced Framework for deploying intelligent data science agents on Databricks
    that can analyze Unity Catalog tables and recommend optimal modeling strategies
    Now powered by LangChain/LangGraph for intelligent orchestration
    """
    
    def __init__(self, 
                 llm_endpoint_name: str = "databricks-claude-3-7-sonnet",
                 catalog: str = "main",
                 schema: str = "analytics",
                 auto_create_functions: bool = True):
        """
        Initialize the DS Assistant Agent Framework
        
        Args:
            llm_endpoint_name: Name of the Databricks LLM endpoint
            catalog: Unity Catalog catalog name
            schema: Unity Catalog schema name
            auto_create_functions: Whether to automatically create UC functions
        """
        self.llm_endpoint_name = llm_endpoint_name
        self.catalog = catalog
        self.schema = schema
        self.auto_create_functions = auto_create_functions
        
        # Initialize Databricks clients
        self.client = DatabricksFunctionClient()
        set_uc_function_client(self.client)
        
        # Initialize LLM
        self.llm = ChatDatabricks(endpoint=llm_endpoint_name)
        
        # System prompt for data science expertise
        self.system_prompt = self._get_system_prompt()
        
        # Create UC functions if requested
        if self.auto_create_functions:
            self._create_uc_functions()
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Initialize intelligent orchestrator
        self.orchestrator = DSOrchestrator(self.llm, self.tools)
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_uc_functions(self):
        """Create Unity Catalog functions for the DS Assistant tools"""
        try:
            # This would typically be done via SQL execution in Databricks
            # For now, we'll provide the SQL that needs to be executed
            sql_commands = self._get_uc_function_sql()
            print(f"UC Functions SQL generated for {self.catalog}.{self.schema}")
            print("Please execute the following SQL commands in your Databricks workspace:")
            print("=" * 80)
            for cmd in sql_commands:
                print(cmd)
                print()
            print("=" * 80)
        except Exception as e:
            print(f"Warning: Could not create UC functions automatically: {e}")
            print("Please create them manually using the provided SQL commands.")
    
    def _get_uc_function_sql(self) -> List[str]:
        """Generate SQL commands for creating UC functions"""
        sql_commands = []
        
        # Create schema if it doesn't exist
        sql_commands.append(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema};")
        
        # Main comprehensive pre-checks function
        sql_commands.append(f"""
CREATE OR REPLACE FUNCTION {self.catalog}.{self.schema}.ds_prechecks(
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
    return f"{{artifacts_dir}}/report_{{state.run_id}}.md"

return main(table_name, target, time_col, artifacts_dir)
$$;
""")
        
        # Data profiling function
        sql_commands.append(f"""
CREATE OR REPLACE FUNCTION {self.catalog}.{self.schema}.data_profiling(
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
    profile_summary = {{
        "shape": {{
            "rows": basic_profile_result.shape.rows,
            "cols": basic_profile_result.shape.cols
        }},
        "missing_data": {{
            "columns_with_missing": [
                {{"col": m.col, "missing_rate": m.missing_rate}} 
                for m in basic_profile_result.missing 
                if m.missing_rate > 0
            ],
            "overall_missing_rate": sum(m.missing_rate for m in basic_profile_result.missing) / len(basic_profile_result.missing)
        }},
        "numerical_features": len(basic_profile_result.numerical_summary),
        "categorical_features": len(basic_profile_result.categorical_cardinality),
        "requires_scaling": scale_check_result.requires_scaling,
        "high_cardinality_features": [
            c.col for c in basic_profile_result.categorical_cardinality 
            if c.n_unique > 50
        ]
    }}
    
    return json.dumps(profile_summary, indent=2)

return main(table_name, artifacts_dir)
$$;
""")
        
        # Feature analysis function
        sql_commands.append(f"""
CREATE OR REPLACE FUNCTION {self.catalog}.{self.schema}.feature_analysis(
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
    analysis_summary = {{
        "correlation_analysis": {{
            "high_correlations": [
                {{"col1": pair.col1, "col2": pair.col2, "correlation": pair.corr}}
                for pair in corr_vif_result.top_pairwise_corrs[:10]  # Top 10
            ],
            "multicollinearity_detected": corr_vif_result.multicollinearity_flag,
            "high_vif_features": [
                {{"col": item.col, "vif": item.vif}}
                for item in corr_vif_result.vif
                if item.vif > 5  # VIF threshold
            ]
        }},
        "outlier_analysis": {{
            "global_outlier_percentage": outliers_result.global_outlier_pct,
            "columns_with_outliers": [
                {{"col": col, "outlier_pct": pct}}
                for col_data in outliers_result.per_col_outlier_pct
                for col, pct in col_data.items()
                if pct > 5  # Outlier threshold
            ],
            "suggest_capping": outliers_result.suggest_capping
        }}
    }}
    
    return json.dumps(analysis_summary, indent=2)

return main(table_name, target, artifacts_dir)
$$;
""")
        
        # Time series analysis function
        sql_commands.append(f"""
CREATE OR REPLACE FUNCTION {self.catalog}.{self.schema}.time_series_analysis(
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
    ts_summary = {{
        "is_time_series": True,
        "stationarity": {{
            "is_stationary": ts_result.adf.get("stationary", False) if ts_result.adf else False,
            "adf_statistic": ts_result.adf.get("adf_stat", None) if ts_result.adf else None,
            "p_value": ts_result.adf.get("p_value", None) if ts_result.adf else None
        }},
        "seasonality": {{
            "detected": ts_result.seasonality_detected,
            "frequency_guess": ts_result.freq_guess
        }},
        "missing_timestamps": {{
            "percentage": ts_result.missing_timestamps_pct
        }},
        "external_variables": {{
            "recommended": ts_result.exog_recommended,
            "variables": ts_result.exog if ts_result.exog else []
        }}
    }}
    
    return json.dumps(ts_summary, indent=2)

return main(table_name, target, time_col, artifacts_dir)
$$;
""")
        
        return sql_commands
        
    def _get_system_prompt(self) -> str:
        """Get the intelligent system prompt for data science expertise"""
        return """You are an expert Data Science Assistant specialized in automated machine learning analysis and recommendations.

Your capabilities include:
1. **Intelligent Data Profiling**: Analyze dataset characteristics, missing values, distributions, and data quality with LLM-powered insights
2. **Advanced Feature Analysis**: Detect correlations, multicollinearity, outliers, and feature relationships with intelligent recommendations
3. **Time Series Intelligence**: Identify seasonality, trends, and forecasting requirements with adaptive analysis
4. **Problem Classification**: Determine if the problem is regression, classification, or forecasting using data-driven reasoning
5. **Model Recommendations**: Suggest optimal algorithms, preprocessing steps, and evaluation metrics with convergence optimization
6. **Adaptive Orchestration**: Intelligently decide analysis steps based on data characteristics and current state

When analyzing a dataset:
1. First perform intelligent data assessment to understand the problem
2. Use LLM reasoning to decide the optimal analysis sequence
3. Perform comprehensive pre-checks using available tools with intelligent insights
4. Synthesize findings into a problem summary with confidence scoring
5. Provide specific, actionable recommendations for:
   - Model family selection with detailed reasoning
   - Required data transformations and preprocessing
   - Hyperparameter tuning strategies
   - Evaluation metrics and validation approaches
   - Convergence optimization techniques
   - Risk assessment and mitigation strategies

Always explain your reasoning and provide evidence from the data analysis to support your recommendations. Use the enhanced tools for intelligent insights and adaptive analysis."""

    def _setup_tools(self) -> List[BaseTool]:
        """Setup intelligent MCP tools for data science analysis"""
        tools = []
        
        # Add intelligent LangChain tools
        tools.extend(INTELLIGENT_TOOLS)
        
        # Register Unity Catalog functions as tools (for backward compatibility)
        uc_tool_names = [
            f"{self.catalog}.{self.schema}.ds_prechecks",
            f"{self.catalog}.{self.schema}.data_profiling",
            f"{self.catalog}.{self.schema}.feature_analysis",
            f"{self.catalog}.{self.schema}.time_series_analysis"
        ]
        
        # Create UC Function Toolkit
        uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
        tools.extend(uc_toolkit.tools)
        
        return tools
    
    def _create_agent(self) -> CompiledGraph:
        """Create the intelligent tool-calling agent with LangGraph"""
        
        def should_continue(state: ChatAgentState):
            messages = state["messages"]
            last_message = messages[-1]
            # If there are function calls, continue. else, end
            if last_message.get("tool_calls"):
                return "continue"
            else:
                return "end"

        # Add system prompt preprocessing
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": self.system_prompt}]
            + state["messages"]
        )
        
        model_runnable = preprocessor | self.llm.bind_tools(self.tools)

        def call_model(
            state: ChatAgentState,
            config: RunnableConfig,
        ):
            response = model_runnable.invoke(state, config)
            return {"messages": [response]}

        # Create workflow
        workflow = StateGraph(ChatAgentState)
        workflow.add_node("agent", RunnableLambda(call_model))
        workflow.add_node("tools", ChatAgentToolNode(self.tools))
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

class LangGraphDSChatAgent(ChatAgent):
    """LangGraph-based Chat Agent for Data Science"""
    
    def __init__(self, agent: CompiledStateGraph, orchestrator: Optional[DSOrchestrator] = None):
        self.agent = agent
        self.orchestrator = orchestrator

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )
    
    def run_analysis(self, dataset_ref: str, target: Optional[str] = None, 
                    time_col: Optional[str] = None, artifacts_dir: str = "./artifacts") -> DSState:
        """Run the intelligent analysis workflow"""
        if self.orchestrator:
            return self.orchestrator.run_analysis(dataset_ref, target, time_col, artifacts_dir)
        else:
            raise ValueError("Orchestrator not available")

def create_ds_assistant_agent(
    llm_endpoint_name: str = "databricks-claude-3-7-sonnet",
    catalog: str = "main", 
    schema: str = "analytics",
    auto_create_functions: bool = True
) -> LangGraphDSChatAgent:
    """
    Factory function to create a DS Assistant Agent
    
    Args:
        llm_endpoint_name: Databricks LLM endpoint name
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        auto_create_functions: Whether to automatically create UC functions
        
    Returns:
        Configured DS Assistant Chat Agent
    """
    framework = DSAssistantAgentFramework(
        llm_endpoint_name=llm_endpoint_name,
        catalog=catalog,
        schema=schema,
        auto_create_functions=auto_create_functions
    )
    
    return LangGraphDSChatAgent(framework.agent, framework.orchestrator)

def deploy_ds_assistant_agent(
    agent: LangGraphDSChatAgent,
    model_name: str,
    catalog: str = "main",
    schema: str = "analytics",
    llm_endpoint_name: str = "databricks-claude-3-7-sonnet"
) -> Dict[str, Any]:
    """
    Deploy the DS Assistant Agent to Unity Catalog
    
    Args:
        agent: The DS Assistant Chat Agent
        model_name: Name for the model in Unity Catalog
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        llm_endpoint_name: LLM endpoint name for resources
        
    Returns:
        Deployment information
    """
    from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
    from pkg_resources import get_distribution
    
    # Set up resources for automatic auth passthrough
    resources = [DatabricksServingEndpoint(endpoint_name=llm_endpoint_name)]
    
    # Add UC function resources
    uc_functions = [
        f"{catalog}.{schema}.ds_prechecks",
        f"{catalog}.{schema}.data_profiling", 
        f"{catalog}.{schema}.feature_analysis",
        f"{catalog}.{schema}.time_series_analysis"
    ]
    
    for func_name in uc_functions:
        resources.append(DatabricksFunction(function_name=func_name))
    
    # Input example for the agent
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the table '{catalog}.{schema}.transactions' and recommend the best modeling strategy for predicting 'target_column'"
            }
        ]
    }
    
    # Log the model
    with mlflow.start_run():
        logged_agent_info = mlflow.pyfunc.log_model(
            name=model_name,
            python_model=agent,
            input_example=input_example,
            resources=resources,
            pip_requirements=[
                f"databricks-connect=={get_distribution('databricks-connect').version}",
                f"mlflow=={get_distribution('mlflow').version}",
                f"databricks-langchain=={get_distribution('databricks-langchain').version}",
                f"langgraph=={get_distribution('langgraph').version}",
                f"ds-assistant=={get_distribution('ds-assistant').version}",
            ],
        )
    
    # Register to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    uc_model_name = f"{catalog}.{schema}.{model_name}"
    
    uc_registered_model_info = mlflow.register_model(
        model_uri=logged_agent_info.model_uri, 
        name=uc_model_name
    )
    
    # Deploy using Databricks Agent Framework
    from databricks import agents
    deployment_info = agents.deploy(
        uc_model_name, 
        uc_registered_model_info.version, 
        tags={"endpointSource": "ds-assistant", "type": "data-science-agent", "version": "intelligent"}
    )
    
    return {
        "model_uri": logged_agent_info.model_uri,
        "uc_model_name": uc_model_name,
        "version": uc_registered_model_info.version,
        "deployment_info": deployment_info
    }
