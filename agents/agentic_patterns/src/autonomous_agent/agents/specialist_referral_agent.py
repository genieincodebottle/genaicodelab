from typing import Dict, Any
import json
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage

from src.autonomous_agent.utils.prompts import SPECIALIST_REFERRAL_PROMPT
from src.autonomous_agent.memory.memory import AgentState
from src.autonomous_agent.utils.retry_utils import (
    retry_with_exponential_backoff,
    safe_extract_and_parse_json
)
from src.utils.llm import llm_call
from src.autonomous_agent.utils.format_helpers import (
    add_timestamp,
    format_referral_output
)

def enhance_referral_prompt(prompt: str, memory: Any) -> str:
    """Enhance referral prompt with historical context"""
    if memory.patient_history:
        previous_referrals = [
            episode.get("referrals", [])
            for episode in memory.episodic_memory
            if "referrals" in episode
        ]
        flattened_referrals = [
            ref for episode_refs in previous_referrals
            for ref in episode_refs
        ]
        
        context = {
            "previous_referrals": flattened_referrals[-5:],  # Last 5 referrals
            "chronic_conditions": memory.patient_history.chronic_conditions,
            "current_specialties": list(set(
                ref.get("specialty", "") 
                for ref in flattened_referrals[-5:]
            ))
        }
        
        enhanced_prompt = f"""
        Patient Referral History:
        {json.dumps(context, indent=2)}
        
        Current Case:
        {prompt}
        """
        return enhanced_prompt
    return prompt

def specialist_referral_agent(state: AgentState) -> Dict[str, Any]:
    """Manages specialist referrals with memory integration."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Specialist Referral Agent")
        
    data = state["data"]
    memory = state["memory"]
    
    if not data["task_queue"]:
        return state
        
    current_task = next(
        task for task in data["tasks"]
        if task.id == data["task_queue"][0]
    )
    
    if "refer_specialist" not in current_task.required_tools:
        return state
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def process_referrals():
        try:
            # Format base prompt
            base_prompt = SPECIALIST_REFERRAL_PROMPT.format(
                task_description=current_task.description,
                patient_state=data["patient_state"],
                context=json.dumps(data.get("context", {}))
            )
            
            # Enhance prompt with historical context
            enhanced_prompt = enhance_referral_prompt(base_prompt, memory)
            
            # Get LLM response
            result = llm_call(enhanced_prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            # Extract and parse with enhanced safety
            referrals = safe_extract_and_parse_json(
                response=result,
                xml_tag="referrals",
                default_value={
                    "specialist_referrals": [],
                    "rationale": "No referrals generated"
                }
            )
            
            # Update memory with referral decisions
            memory.update_working_memory(
                state=data["patient_state"],
                reasoning={
                    "timestamp": datetime.now().isoformat(),
                    "referrals": referrals
                },
                decision={
                    "type": "specialist_referral",
                    "specialties": [ref["specialty"] for ref in referrals["specialist_referrals"]],
                    "rationale": referrals["rationale"]
                }
            )
            
            # Add intervention record
            intervention = {
                "type": "specialist_referral",
                "timestamp": datetime.now().isoformat(),
                "details": referrals
            }
            memory.working_memory.interventions.append(intervention)
            
            # Format output if debugging
            if state["metadata"].get("show_reasoning"):
                format_referral_output(referrals)
            
            return referrals
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Referral processing failed: {str(e)}")
            raise
    
    try:
        # Process referrals with retries
        referrals = process_referrals()
        
        # Update task state
        current_task.state = "completed"
        current_task.result = referrals
        current_task.completed_at = datetime.now()
        
        # Update task queue
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
        
    except Exception as e:
        # Create fallback referrals on complete failure
        fallback_referrals = {
            "specialist_referrals": [{
                "specialty": "manual_review_required",
                "priority": "routine",
                "reason": f"Referral processing failed: {str(e)}",
                "notes": "System error occurred"
            }],
            "rationale": "Fallback due to processing error"
        }
        
        # Update memory with failure
        memory.update_working_memory(
            state=data["patient_state"],
            reasoning={
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "fallback": "Using default referrals"
            },
            decision={
                "type": "referral_failure",
                "action": "use_defaults",
                "reason": str(e)
            }
        )
        
        current_task.state = "completed"
        current_task.result = fallback_referrals
        current_task.completed_at = datetime.now()
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
    
    # Create message and update conversation memory
    message_content = json.dumps(current_task.result)
    message = HumanMessage(
        content=message_content,
        name="specialist_referral_agent"
    )
    
    memory.add_to_conversation(message_content, "ai")
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
        "memory": memory
    }