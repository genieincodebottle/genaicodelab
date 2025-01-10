from typing import Dict, Any
import json
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage

from src.autonomous_agent.utils.prompts import PRESCRIPTION_PROMPT
from src.autonomous_agent.memory.memory import AgentState
from src.autonomous_agent.utils.retry_utils import (
    retry_with_exponential_backoff,
    safe_extract_and_parse_json
)
from src.utils.llm import llm_call
from src.autonomous_agent.utils.format_helpers import (
    add_timestamp,
    format_prescription_output
)

def enhance_prescription_prompt(prompt: str, memory: Any, data: Dict[str, Any]) -> str:
    """Enhance prescription prompt with historical context"""
    if memory.patient_history:
        medication_history = memory.patient_history.medication_history[-5:]  # Last 5 medication records
        recent_lab_results = memory.patient_history.lab_results[-3:]  # Last 3 lab results
        
        # Extract allergies and adverse reactions from history
        allergies = memory.patient_history.allergies
        adverse_reactions = [
            med for med in medication_history
            if med.get("adverse_reaction")
        ]
        
        context = {
            "medication_history": medication_history,
            "current_medications": data.get("current_medications", []),
            "allergies": allergies,
            "adverse_reactions": adverse_reactions,
            "recent_lab_results": recent_lab_results,
            "chronic_conditions": memory.patient_history.chronic_conditions
        }
        
        enhanced_prompt = f"""
        Patient Medication History:
        {json.dumps(context, indent=2)}
        
        Current Prescription Request:
        {prompt}
        
        Consider all medication interactions, allergies, and previous adverse reactions.
        """
        return enhanced_prompt
    return prompt

def prescription_agent(state: AgentState) -> Dict[str, Any]:
    """Manages medication prescriptions with memory integration."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Prescription Agent")
        
    data = state["data"]
    memory = state["memory"]
    
    if not data["task_queue"]:
        return state
        
    current_task = next(
        task for task in data["tasks"]
        if task.id == data["task_queue"][0]
    )
    
    if "prescribe_medication" not in current_task.required_tools:
        return state
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def process_prescriptions():
        try:
            # Format base prompt
            base_prompt = PRESCRIPTION_PROMPT.format(
                task_description=current_task.description,
                patient_state=data["patient_state"],
                medications=json.dumps(data.get("current_medications", [])),
                context=json.dumps(data.get("context", {}))
            )
            
            # Enhance prompt with historical context
            enhanced_prompt = enhance_prescription_prompt(base_prompt, memory, data)
            
            # Get LLM response
            result = llm_call(enhanced_prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            # Extract and parse with enhanced safety
            prescriptions = safe_extract_and_parse_json(
                response=result,
                xml_tag="prescriptions",
                default_value={
                    "medication_orders": [],
                    "rationale": "No prescriptions generated"
                }
            )
            
            # Update memory with prescription decisions
            memory.update_working_memory(
                state=data["patient_state"],
                reasoning={
                    "timestamp": datetime.now().isoformat(),
                    "prescriptions": prescriptions
                },
                decision={
                    "type": "medication_prescription",
                    "medications": [med["medication"] for med in prescriptions["medication_orders"]],
                    "rationale": prescriptions["rationale"]
                }
            )
            
            # Add intervention record
            intervention = {
                "type": "prescription",
                "timestamp": datetime.now().isoformat(),
                "details": prescriptions
            }
            memory.working_memory.interventions.append(intervention)
            
            # Add observation for potential drug interactions
            if len(prescriptions["medication_orders"]) > 1:
                observation = {
                    "type": "drug_interaction_check",
                    "timestamp": datetime.now().isoformat(),
                    "medications": [med["medication"] for med in prescriptions["medication_orders"]],
                    "current_medications": data.get("current_medications", [])
                }
                memory.working_memory.key_observations.append(observation)
            
            # Format output if debugging
            if state["metadata"].get("show_reasoning"):
                format_prescription_output(prescriptions)
            
            return prescriptions
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Prescription processing failed: {str(e)}")
            raise
    
    try:
        # Process prescriptions with retries
        prescriptions = process_prescriptions()
        
        # Update task state
        current_task.state = "completed"
        current_task.result = prescriptions
        current_task.completed_at = datetime.now()
        
        # Update task queue
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
        
        # Update patient medication history
        memory.update_patient_history(
            medications={
                "timestamp": datetime.now().isoformat(),
                "prescriptions": prescriptions["medication_orders"],
                "rationale": prescriptions["rationale"]
            }
        )
        
    except Exception as e:
        # Create fallback prescriptions on complete failure
        fallback_prescriptions = {
            "medication_orders": [{
                "medication": "manual_review_required",
                "dosage": "N/A",
                "frequency": "N/A",
                "duration": "N/A",
                "special_instructions": f"Prescription processing failed: {str(e)}"
            }],
            "rationale": "Fallback due to processing error"
        }
        
        # Update memory with failure
        memory.update_working_memory(
            state=data["patient_state"],
            reasoning={
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "fallback": "Using default prescriptions"
            },
            decision={
                "type": "prescription_failure",
                "action": "use_defaults",
                "reason": str(e)
            }
        )
        
        current_task.state = "completed"
        current_task.result = fallback_prescriptions
        current_task.completed_at = datetime.now()
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
    
    # Create message and update conversation memory
    message_content = json.dumps(current_task.result)
    message = HumanMessage(
        content=message_content,
        name="prescription_agent"
    )
    
    memory.add_to_conversation(message_content, "ai")
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
        "memory": memory
    }