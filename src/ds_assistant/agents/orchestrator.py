"""
DS Assistant Orchestrator using LangGraph
Intelligent, adaptive data science workflow orchestration
"""

from typing import TypedDict, Annotated, Dict, List, Any, Optional
import polars as pl
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from ..common.schemas import ProblemSummary, Recommendation
from ..tools import data_profiling, relationships, timeseries, recommender, report_builder

# Enhanced state management with LangGraph TypedDict
class DSState(TypedDict):
    """Enhanced state for intelligent data science workflow"""
    # Core data
    dataset_ref: str
    target: Optional[str]
    time_col: Optional[str]
    artifacts_dir: str
    run_id: str
    
    # Analysis progress
    current_step: str
    completed_steps: List[str]
    analysis_results: Dict[str, Any]
    
    # LLM context
    messages: List[Any]  # LangChain messages
    reasoning: List[str]
    
    # Quality and errors
    data_quality_score: float
    confidence_score: float
    errors: List[str]
    warnings: List[str]
    
    # Recommendations
    recommendations: List[Dict[str, Any]]
    final_recommendation: Optional[Dict[str, Any]]
    
    # Metadata
    metadata: Dict[str, Any]

# LLM-powered decision making
class AnalysisDecision(BaseModel):
    """LLM decision about next analysis step"""
    next_step: str = Field(description="Next step to execute")
    reasoning: str = Field(description="Reasoning for this decision")
    confidence: float = Field(description="Confidence in this decision (0-1)")
    skip_steps: List[str] = Field(description="Steps to skip based on current state")
    priority: str = Field(description="Priority: high, medium, low")

class DSOrchestrator:
    """
    Intelligent orchestrator using LangGraph and LLM reasoning
    """
    
    def __init__(self, llm, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for intelligent data science analysis"""
        
        # Create the state graph
        workflow = StateGraph(DSState)
        
        # Add nodes
        workflow.add_node("intelligent_router", self._intelligent_router_node)
        workflow.add_node("data_profiling", self._data_profiling_node)
        workflow.add_node("feature_analysis", self._feature_analysis_node)
        workflow.add_node("time_series_analysis", self._time_series_node)
        workflow.add_node("quality_assessment", self._quality_assessment_node)
        workflow.add_node("model_recommendation", self._model_recommendation_node)
        workflow.add_node("error_recovery", self._error_recovery_node)
        workflow.add_node("final_synthesis", self._final_synthesis_node)
        
        # Add tool node for parallel execution
        tool_node = ToolNode(self.tools)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("intelligent_router")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "intelligent_router",
            self._route_to_next_step,
            {
                "data_profiling": "data_profiling",
                "feature_analysis": "feature_analysis", 
                "time_series_analysis": "time_series_analysis",
                "quality_assessment": "quality_assessment",
                "model_recommendation": "model_recommendation",
                "error_recovery": "error_recovery",
                "final_synthesis": "final_synthesis",
                "tools": "tools",
                "end": END
            }
        )
        
        # Add edges from analysis nodes back to router
        workflow.add_edge("data_profiling", "intelligent_router")
        workflow.add_edge("feature_analysis", "intelligent_router")
        workflow.add_edge("time_series_analysis", "intelligent_router")
        workflow.add_edge("quality_assessment", "intelligent_router")
        workflow.add_edge("error_recovery", "intelligent_router")
        workflow.add_edge("tools", "intelligent_router")
        
        # Final synthesis leads to end
        workflow.add_edge("model_recommendation", "final_synthesis")
        workflow.add_edge("final_synthesis", END)
        
        return workflow.compile()
    
    def _intelligent_router_node(self, state: DSState) -> DSState:
        """LLM-powered router that decides the next analysis step"""
        
        # Create decision prompt
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Science Orchestrator. Your job is to intelligently decide the next analysis step based on the current state.

Available analysis steps:
1. data_profiling - Basic data profiling (shape, missing values, distributions)
2. feature_analysis - Feature relationships, correlations, multicollinearity
3. time_series_analysis - Time series specific analysis (seasonality, stationarity)
4. quality_assessment - Overall data quality and issues assessment
5. model_recommendation - Final model recommendations and strategies
6. error_recovery - Handle errors and retry strategies
7. final_synthesis - Synthesize all results into final report

Decision criteria:
- Always start with data_profiling if not completed
- Run feature_analysis if data has multiple features and correlations matter
- Run time_series_analysis if time column is present
- Run quality_assessment after basic profiling
- Run model_recommendation only after all analysis is complete
- Run error_recovery if there are errors
- Run final_synthesis to complete the workflow

Current state: {current_state}
Completed steps: {completed_steps}
Errors: {errors}
Data characteristics: {data_characteristics}

Decide the next step and provide reasoning."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "What should be the next analysis step?")
        ])
        
        # Get current state summary
        current_state = self._summarize_current_state(state)
        
        # Create decision chain
        decision_chain = (
            decision_prompt 
            | self.llm 
            | JsonOutputParser(pydantic_object=AnalysisDecision)
        )
        
        # Get decision
        decision = decision_chain.invoke({
            "current_state": current_state,
            "completed_steps": state.get("completed_steps", []),
            "errors": state.get("errors", []),
            "data_characteristics": state.get("analysis_results", {}),
            "messages": state.get("messages", [])
        })
        
        # Update state with decision
        state["current_step"] = decision.next_step
        state["reasoning"].append(f"Decision: {decision.reasoning} (confidence: {decision.confidence})")
        
        # Add decision to messages
        state["messages"].append(
            AIMessage(content=f"Next step: {decision.next_step}. Reasoning: {decision.reasoning}")
        )
        
        return state
    
    def _route_to_next_step(self, state: DSState) -> str:
        """Route to the next step based on intelligent decision"""
        return state["current_step"]
    
    def _data_profiling_node(self, state: DSState) -> DSState:
        """Enhanced data profiling with LLM reasoning"""
        
        try:
            # Load data
            df, meta = self._load_data(state["dataset_ref"])
            
            # Perform profiling
            profile_result = data_profiling.basic_profile(df, state["artifacts_dir"])
            scale_result = data_profiling.scale_check(df)
            
            # LLM analysis of profiling results
            analysis_prompt = f"""
            Analyze the data profiling results:
            
            Shape: {profile_result.shape.rows} rows, {profile_result.shape.cols} columns
            Missing data: {len([m for m in profile_result.missing if m.missing_rate > 0])} columns with missing values
            Numerical features: {len(profile_result.numerical_summary)}
            Categorical features: {len(profile_result.categorical_cardinality)}
            Requires scaling: {scale_result.requires_scaling}
            
            Provide insights about:
            1. Data quality issues
            2. Potential preprocessing needs
            3. Feature engineering opportunities
            4. Model selection implications
            """
            
            analysis = self.llm.invoke(analysis_prompt)
            
            # Update state
            state["analysis_results"]["data_profiling"] = {
                "profile": profile_result.model_dump(),
                "scale_check": scale_result.model_dump(),
                "llm_insights": analysis.content
            }
            state["completed_steps"].append("data_profiling")
            state["data_quality_score"] = self._calculate_quality_score(profile_result)
            
            # Add to messages
            state["messages"].append(
                ToolMessage(
                    content=f"Data profiling completed. Quality score: {state['data_quality_score']:.2f}",
                    tool_call_id="data_profiling"
                )
            )
            
        except Exception as e:
            state["errors"].append(f"Data profiling failed: {str(e)}")
            state["current_step"] = "error_recovery"
        
        return state
    
    def _feature_analysis_node(self, state: DSState) -> DSState:
        """Enhanced feature analysis with intelligent decision making"""
        
        try:
            # Load data
            df, meta = self._load_data(state["dataset_ref"])
            
            # Perform feature analysis
            corr_result = relationships.corr_and_vif(df, state["artifacts_dir"], target=state["target"])
            outlier_result = relationships.outliers(df, method="iqr")
            
            # LLM analysis of feature relationships
            analysis_prompt = f"""
            Analyze the feature analysis results:
            
            High correlations: {len(corr_result.top_pairwise_corrs)} pairs
            Multicollinearity detected: {corr_result.multicollinearity_flag}
            High VIF features: {len([v for v in corr_result.vif if v.vif > 5])}
            Global outlier percentage: {outlier_result.global_outlier_pct:.2f}%
            
            Provide insights about:
            1. Feature selection strategy
            2. Multicollinearity handling
            3. Outlier treatment approach
            4. Impact on model performance
            """
            
            analysis = self.llm.invoke(analysis_prompt)
            
            # Update state
            state["analysis_results"]["feature_analysis"] = {
                "correlations": corr_result.model_dump(),
                "outliers": outlier_result.model_dump(),
                "llm_insights": analysis.content
            }
            state["completed_steps"].append("feature_analysis")
            
            # Add to messages
            state["messages"].append(
                ToolMessage(
                    content=f"Feature analysis completed. {len(corr_result.top_pairwise_corrs)} high correlations found.",
                    tool_call_id="feature_analysis"
                )
            )
            
        except Exception as e:
            state["errors"].append(f"Feature analysis failed: {str(e)}")
            state["current_step"] = "error_recovery"
        
        return state
    
    def _time_series_node(self, state: DSState) -> DSState:
        """Enhanced time series analysis with LLM reasoning"""
        
        if not state["time_col"]:
            state["warnings"].append("Time series analysis skipped - no time column specified")
            return state
        
        try:
            # Load data
            df, meta = self._load_data(state["dataset_ref"])
            
            # Perform time series analysis
            ts_result = timeseries.stl_adf_acf_pacf(
                df, state["target"] or "", state["time_col"], 
                state["artifacts_dir"], None
            )
            
            # LLM analysis of time series characteristics
            analysis_prompt = f"""
            Analyze the time series results:
            
            Stationary: {ts_result.adf.get('stationary', False) if ts_result.adf else False}
            Seasonality detected: {ts_result.seasonality_detected}
            Frequency guess: {ts_result.freq_guess}
            Missing timestamps: {ts_result.missing_timestamps_pct:.2f}%
            
            Provide insights about:
            1. Forecasting approach recommendations
            2. Seasonality handling
            3. Stationarity treatment
            4. Model family suggestions
            """
            
            analysis = self.llm.invoke(analysis_prompt)
            
            # Update state
            state["analysis_results"]["time_series"] = {
                "analysis": ts_result.model_dump(),
                "llm_insights": analysis.content
            }
            state["completed_steps"].append("time_series_analysis")
            
            # Add to messages
            state["messages"].append(
                ToolMessage(
                    content=f"Time series analysis completed. Seasonality: {ts_result.seasonality_detected}",
                    tool_call_id="time_series_analysis"
                )
            )
            
        except Exception as e:
            state["errors"].append(f"Time series analysis failed: {str(e)}")
            state["current_step"] = "error_recovery"
        
        return state
    
    def _quality_assessment_node(self, state: DSState) -> DSState:
        """LLM-powered overall quality assessment"""
        
        try:
            # Synthesize all analysis results
            all_results = state["analysis_results"]
            
            quality_prompt = f"""
            Assess the overall data quality and analysis results:
            
            Data Profiling: {all_results.get('data_profiling', {})}
            Feature Analysis: {all_results.get('feature_analysis', {})}
            Time Series: {all_results.get('time_series', {})}
            Errors: {state['errors']}
            Warnings: {state['warnings']}
            
            Provide a comprehensive quality assessment including:
            1. Overall data quality score (0-100)
            2. Major issues and concerns
            3. Recommendations for data preparation
            4. Impact on modeling success
            5. Risk assessment for model deployment
            """
            
            assessment = self.llm.invoke(quality_prompt)
            
            # Update state
            state["analysis_results"]["quality_assessment"] = {
                "assessment": assessment.content,
                "quality_score": state.get("data_quality_score", 0)
            }
            state["completed_steps"].append("quality_assessment")
            
            # Add to messages
            state["messages"].append(
                ToolMessage(
                    content=f"Quality assessment completed. Score: {state.get('data_quality_score', 0):.1f}/100",
                    tool_call_id="quality_assessment"
                )
            )
            
        except Exception as e:
            state["errors"].append(f"Quality assessment failed: {str(e)}")
            state["current_step"] = "error_recovery"
        
        return state
    
    def _model_recommendation_node(self, state: DSState) -> DSState:
        """LLM-powered intelligent model recommendations"""
        
        try:
            # Create comprehensive problem summary
            problem_summary = self._create_problem_summary(state)
            
            # LLM-powered recommendation
            recommendation_prompt = f"""
            Based on the comprehensive analysis, provide detailed model recommendations:
            
            Problem Summary: {problem_summary}
            Analysis Results: {state['analysis_results']}
            Quality Score: {state.get('data_quality_score', 0)}
            
            Provide detailed recommendations for:
            1. Primary model family with reasoning
            2. Alternative model families
            3. Preprocessing strategies
            4. Feature engineering approaches
            5. Hyperparameter tuning strategy
            6. Evaluation metrics
            7. Convergence optimization techniques
            8. Risk mitigation strategies
            
            Structure your response as a comprehensive recommendation plan.
            """
            
            recommendation = self.llm.invoke(recommendation_prompt)
            
            # Update state
            state["final_recommendation"] = {
                "recommendation": recommendation.content,
                "problem_summary": problem_summary,
                "confidence_score": self._calculate_confidence_score(state)
            }
            state["completed_steps"].append("model_recommendation")
            
            # Add to messages
            state["messages"].append(
                ToolMessage(
                    content="Model recommendations generated with intelligent reasoning",
                    tool_call_id="model_recommendation"
                )
            )
            
        except Exception as e:
            state["errors"].append(f"Model recommendation failed: {str(e)}")
            state["current_step"] = "error_recovery"
        
        return state
    
    def _error_recovery_node(self, state: DSState) -> DSState:
        """Intelligent error recovery with LLM reasoning"""
        
        if not state["errors"]:
            return state
        
        try:
            # LLM-powered error recovery
            recovery_prompt = f"""
            The analysis encountered errors: {state['errors']}
            Current state: {state['current_step']}
            Completed steps: {state['completed_steps']}
            
            Provide recovery strategies:
            1. Can we retry with different parameters?
            2. Should we skip this step and continue?
            3. Are there alternative approaches?
            4. Should we stop the analysis?
            
            Provide specific recovery actions and reasoning.
            """
            
            recovery_plan = self.llm.invoke(recovery_prompt)
            
            # Update state
            state["analysis_results"]["error_recovery"] = {
                "recovery_plan": recovery_plan.content,
                "errors": state["errors"]
            }
            
            # Decide next action based on recovery plan
            if "retry" in recovery_plan.content.lower():
                # Retry the failed step
                pass
            elif "skip" in recovery_plan.content.lower():
                # Skip to next step
                state["current_step"] = "intelligent_router"
            elif "stop" in recovery_plan.content.lower():
                # End analysis
                state["current_step"] = "final_synthesis"
            
            # Add to messages
            state["messages"].append(
                ToolMessage(
                    content=f"Error recovery plan: {recovery_plan.content[:100]}...",
                    tool_call_id="error_recovery"
                )
            )
            
        except Exception as e:
            # If error recovery itself fails, go to final synthesis
            state["current_step"] = "final_synthesis"
        
        return state
    
    def _final_synthesis_node(self, state: DSState) -> DSState:
        """Final synthesis of all results with LLM reasoning"""
        
        try:
            # Generate final report
            synthesis_prompt = f"""
            Synthesize all analysis results into a comprehensive final report:
            
            Analysis Results: {state['analysis_results']}
            Recommendations: {state['final_recommendation']}
            Quality Score: {state.get('data_quality_score', 0)}
            Errors: {state['errors']}
            Warnings: {state['warnings']}
            
            Create a comprehensive final report including:
            1. Executive summary
            2. Data quality assessment
            3. Key findings and insights
            4. Model recommendations
            5. Implementation roadmap
            6. Risk assessment
            7. Next steps
            """
            
            final_report = self.llm.invoke(synthesis_prompt)
            
            # Build final report
            report_builder.build_report(state, state["artifacts_dir"])
            
            # Update state
            state["analysis_results"]["final_synthesis"] = {
                "final_report": final_report.content,
                "completion_status": "success" if not state["errors"] else "partial"
            }
            state["completed_steps"].append("final_synthesis")
            
            # Add to messages
            state["messages"].append(
                ToolMessage(
                    content="Analysis completed successfully. Final report generated.",
                    tool_call_id="final_synthesis"
                )
            )
            
        except Exception as e:
            state["errors"].append(f"Final synthesis failed: {str(e)}")
        
        return state
    
    # Helper methods
    def _load_data(self, dataset_ref: str) -> tuple[pl.DataFrame, dict]:
        """Load data from Unity Catalog"""
        from ..tools.data_connector import DataConnector
        return DataConnector.load_uc_table(dataset_ref)
    
    def _summarize_current_state(self, state: DSState) -> str:
        """Summarize current state for LLM decision making"""
        return f"""
        Current step: {state.get('current_step', 'initial')}
        Completed: {len(state.get('completed_steps', []))} steps
        Quality score: {state.get('data_quality_score', 0):.1f}
        Errors: {len(state.get('errors', []))}
        Target: {state.get('target', 'None')}
        Time column: {state.get('time_col', 'None')}
        """
    
    def _calculate_quality_score(self, profile_result) -> float:
        """Calculate data quality score based on profiling results"""
        score = 100.0
        
        # Penalize for missing data
        missing_rates = [m.missing_rate for m in profile_result.missing]
        avg_missing = sum(missing_rates) / len(missing_rates) if missing_rates else 0
        score -= avg_missing * 50  # 50% penalty for missing data
        
        # Penalize for high cardinality
        high_cardinality = [c for c in profile_result.categorical_cardinality if c.n_unique > 50]
        score -= len(high_cardinality) * 5  # 5 points per high cardinality feature
        
        return max(0, min(100, score))
    
    def _calculate_confidence_score(self, state: DSState) -> float:
        """Calculate confidence in recommendations"""
        confidence = 1.0
        
        # Reduce confidence for errors
        confidence -= len(state.get("errors", [])) * 0.1
        
        # Reduce confidence for low quality data
        quality_score = state.get("data_quality_score", 0)
        confidence *= quality_score / 100
        
        # Increase confidence for comprehensive analysis
        completed_steps = state.get("completed_steps", [])
        if len(completed_steps) >= 4:
            confidence *= 1.2
        
        return max(0, min(1, confidence))
    
    def _create_problem_summary(self, state: DSState) -> ProblemSummary:
        """Create problem summary from analysis results"""
        # Implementation would synthesize all analysis results
        # into a comprehensive problem summary
        pass
    
    def run_analysis(self, dataset_ref: str, target: Optional[str] = None, 
                    time_col: Optional[str] = None, artifacts_dir: str = "./artifacts") -> DSState:
        """Run the intelligent data science analysis workflow"""
        
        # Initialize state
        import uuid
        initial_state = DSState(
            dataset_ref=dataset_ref,
            target=target,
            time_col=time_col,
            artifacts_dir=artifacts_dir,
            run_id=uuid.uuid4().hex[:12],
            current_step="intelligent_router",
            completed_steps=[],
            analysis_results={},
            messages=[],
            reasoning=[],
            data_quality_score=0.0,
            confidence_score=0.0,
            errors=[],
            warnings=[],
            recommendations=[],
            final_recommendation=None,
            metadata={}
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state


