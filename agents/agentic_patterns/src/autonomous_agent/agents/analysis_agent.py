from typing import Dict, Any
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage

from src.autonomous_agent.utils.prompts import ANALYSIS_PROMPT
from src.autonomous_agent.memory.memory import AgentState
from src.autonomous_agent.utils.retry_utils import (
    retry_with_exponential_backoff,
    safe_json_loads
)
from src.autonomous_agent.utils.json_utils import json_serialize
from src.autonomous_agent.utils.format_helpers import (
    add_timestamp,
    format_analysis_output
)
from src.utils.llm import llm_call, extract_xml


def analysis_agent(state: AgentState) -> Dict[str, Any]:
    """Analyzes patient data with memory integration."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Analysis Agent")
    
    data = state["data"]
    memory = state["memory"]
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def run_analysis():
        try:
            # Format prompt
            prompt = ANALYSIS_PROMPT.format(
                case=data["patient_data"]["medical_case"]
            )
            
            # Get LLM response
            result = llm_call(prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            # Extract and clean up analysis
            analysis_text = extract_xml(result, "analysis")
            analysis_text = analysis_text.strip()
            
            if state["metadata"].get("show_reasoning"):
                st.write("Raw LLM response:", result)
                st.write("Extracted analysis:", analysis_text)
            
            # Parse the JSON
            analysis = safe_json_loads(analysis_text)
            
            # Validate patient state
            valid_states = ["STABLE", "REQUIRES_MONITORING", "URGENT", "CRITICAL"]
            if analysis.get("patient_state") not in valid_states:
                analysis["patient_state"] = "STABLE"
            
            # Ensure required fields
            default_fields = {
                "immediate_concerns": [],
                "risk_factors": [],
                "monitoring_needs": [],
                "preliminary_diagnosis": [],
                "reasoning": "Analysis completed"
            }
            
            for field, default in default_fields.items():
                if field not in analysis:
                    analysis[field] = default
            
            # Update memory
            memory.update_working_memory(
                state=analysis["patient_state"],
                reasoning={
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis
                },
                decision={
                    "type": "initial_analysis",
                    "state": analysis["patient_state"],
                    "concerns": analysis["immediate_concerns"]
                }
            )
            
            if state["metadata"].get("show_reasoning"):
                format_analysis_output(analysis)
                    
            return analysis
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Analysis attempt failed: {str(e)}")
            raise
    
    try:
        analysis = run_analysis()
        
    except Exception as e:
        if state["metadata"].get("show_reasoning"):
            st.error(f"All analysis attempts failed: {str(e)}")
        
        analysis = {
            "patient_state": "STABLE",
            "immediate_concerns": ["Analysis failed - using default values"],
            "risk_factors": [],
            "monitoring_needs": ["Manual review required"],
            "preliminary_diagnosis": [],
            "reasoning": f"Analysis error: {str(e)}"
        }
        
        memory.update_working_memory(
            state="STABLE",
            reasoning={
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "fallback": "Using default analysis"
            },
            decision={
                "type": "analysis_failure",
                "action": "use_defaults",
                "reason": str(e)
            }
        )
    
    # Update state
    data["analysis"] = analysis
    data["patient_state"] = analysis["patient_state"]
    
    # Create message with proper serialization
    message = HumanMessage(
        content=json_serialize(analysis),
        name="analysis_agent"
    )
    
    memory.add_to_conversation(json_serialize(analysis), "ai")
    
    return {
        "messages": [message],
        "data": data,
        "memory": memory
    }