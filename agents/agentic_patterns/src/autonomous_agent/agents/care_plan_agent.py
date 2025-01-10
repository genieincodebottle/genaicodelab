from typing import Dict, Any
import json
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage

from src.autonomous_agent.utils.prompts import CARE_PLAN_PROMPT
from src.autonomous_agent.memory.memory import AgentState
from src.autonomous_agent.utils.retry_utils import (
    retry_with_exponential_backoff,
    safe_extract_and_parse_json
)
from src.utils.llm import llm_call
from src.autonomous_agent.utils.format_helpers import (
    add_timestamp,
    format_care_plan_output
)

def enhance_care_plan_prompt(prompt: str, memory: Any, completed_tasks: Dict[str, Any]) -> str:
    """Enhance care plan prompt with historical context"""
    if memory.patient_history:
        context = {
            "previous_diagnoses": memory.patient_history.previous_diagnoses[-3:],
            "treatment_history": memory.patient_history.treatment_history[-3:],
            "chronic_conditions": memory.patient_history.chronic_conditions,
            "current_episode": {
                "observations": memory.working_memory.key_observations,
                "interventions": memory.working_memory.interventions,
                "decisions": memory.working_memory.decision_points
            }
        }
        
        enhanced_prompt = f"""
        Patient History and Current Episode:
        {json.dumps(context, indent=2)}
        
        Completed Tasks and Results:
        {json.dumps(completed_tasks, indent=2)}
        
        Generate Care Plan:
        {prompt}
        """
        return enhanced_prompt
    return prompt

def care_plan_agent(state: AgentState) -> Dict[str, Any]:
    """Creates comprehensive care plans with memory integration."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Care Plan Agent")
        
    data = state["data"]
    memory = state["memory"]
    completed_tasks = data.get("completed_tasks", [])
    
    if not completed_tasks:
        return state
    
    task_results = {
        task.id: task.result 
        for task in data["tasks"]
        if task.id in completed_tasks
    }
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def generate_care_plan():
        try:
            # Format base prompt
            base_prompt = CARE_PLAN_PROMPT.format(
                analysis=json.dumps(data["analysis"]),
                completed_tasks=json.dumps(completed_tasks),
                task_results=json.dumps(task_results)
            )
            
            # Enhance prompt with historical context
            enhanced_prompt = enhance_care_plan_prompt(base_prompt, memory, task_results)
            
            # Get LLM response
            result = llm_call(enhanced_prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            # Extract and parse with enhanced safety
            care_plan = safe_extract_and_parse_json(
                response=result,
                xml_tag="care_plan",
                default_value={
                    "diagnosis": {
                        "primary": "Requires manual review",
                        "secondary": []
                    },
                    "treatment_plan": [],
                    "monitoring_plan": [],
                    "follow_up": [],
                    "emergency_plan": "Contact healthcare provider immediately if condition worsens"
                }
            )
            
            # Update memory with care plan
            memory.update_working_memory(
                state=data["patient_state"],
                reasoning={
                    "timestamp": datetime.now().isoformat(),
                    "care_plan": care_plan
                },
                decision={
                    "type": "care_plan_generation",
                    "primary_diagnosis": care_plan["diagnosis"]["primary"],
                    "treatment_count": len(care_plan["treatment_plan"])
                }
            )
            
            # Add to patient history
            memory.update_patient_history(
                diagnosis=care_plan["diagnosis"],
                treatment={
                    "timestamp": datetime.now().isoformat(),
                    "plan": care_plan["treatment_plan"],
                    "monitoring": care_plan["monitoring_plan"]
                }
            )
            
            # Format output if debugging
            if state["metadata"].get("show_reasoning"):
                format_care_plan_output(care_plan)
            
            return care_plan
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Care plan generation failed: {str(e)}")
            raise
    
    try:
        # Generate care plan with retries
        care_plan = generate_care_plan()
        
        # Update state with care plan
        data["care_plan"] = care_plan
        
    except Exception as e:
        # Create fallback care plan on complete failure
        fallback_care_plan = {
            "diagnosis": {
                "primary": "System Error - Manual Review Required",
                "secondary": []
            },
            "treatment_plan": [
                f"Care plan generation failed: {str(e)}",
                "Please review all completed tasks manually"
            ],
            "monitoring_plan": [
                "Regular vital signs monitoring",
                "Report any concerning symptoms"
            ],
            "follow_up": [
                "Schedule follow-up appointment with primary care provider"
            ],
            "emergency_plan": "Seek immediate medical attention if condition worsens"
        }
        
        # Update memory with failure
        memory.update_working_memory(
            state=data["patient_state"],
            reasoning={
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "fallback": "Using default care plan"
            },
            decision={
                "type": "care_plan_failure",
                "action": "use_defaults",
                "reason": str(e)
            }
        )
        
        data["care_plan"] = fallback_care_plan
    
    # Save completed episode to memory
    memory.save_to_episodic_memory()
    
    # Create message and update conversation memory
    message_content = json.dumps(data["care_plan"])
    message = HumanMessage(
        content=message_content,
        name="care_plan_agent"
    )
    
    memory.add_to_conversation(message_content, "ai")
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
        "memory": memory
    }