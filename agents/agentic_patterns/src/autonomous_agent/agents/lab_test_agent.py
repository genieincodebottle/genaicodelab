from typing import Dict, Any
import json
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage

from src.autonomous_agent.utils.prompts import LAB_TEST_PROMPT
from src.autonomous_agent.memory.memory import AgentState
from src.autonomous_agent.utils.retry_utils import (
    retry_with_exponential_backoff,
    safe_extract_and_parse_json
)
from src.utils.llm import llm_call
from src.autonomous_agent.utils.format_helpers import (
    add_timestamp,
    format_lab_test_output
)

def enhance_lab_test_prompt(prompt: str, memory: Any, data: Dict[str, Any]) -> str:
    """Enhance lab test prompt with historical context"""
    if memory and hasattr(memory, "patient_history") and memory.patient_history:
        # Safely extract data from patient history
        historical_context = {
            "previous_lab_results": getattr(memory.patient_history, "lab_results", []),
            "chronic_conditions": getattr(data.get("patient_data", {}), "chronic_conditions", []),
            "current_medications": getattr(data.get("patient_data", {}), "current_medications", [])
        }
        
        # Add relevant observations from working memory
        if hasattr(memory, "working_memory") and memory.working_memory:
            observations = getattr(memory.working_memory, "key_observations", [])
            if observations:
                historical_context["current_observations"] = observations
        
        enhanced_prompt = f"""
        Patient History:
        {json.dumps(historical_context, indent=2)}
        
        Current Case:
        {prompt}
        """
        return enhanced_prompt
    return prompt

def lab_test_agent(state: AgentState) -> Dict[str, Any]:
    """Orders and processes laboratory tests with memory integration."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Lab Test Agent")
        
    data = state["data"]
    memory = state.get("memory")
    
    # Initialize task queue and tasks list if they don't exist
    if "task_queue" not in data:
        data["task_queue"] = []
    if "tasks" not in data:
        data["tasks"] = []
    if "completed_tasks" not in data:
        data["completed_tasks"] = []
    
    # Check if there are any tasks to process
    if not data["task_queue"]:
        if state["metadata"].get("show_reasoning"):
            st.info("No tasks in queue to process")
        return state
        
    try:
        # Get current task
        current_task = next(
            task for task in data["tasks"]
            if task.id == data["task_queue"][0]
        )
    except (StopIteration, IndexError):
        if state["metadata"].get("show_reasoning"):
            st.warning("No valid task found in queue")
        return state
    
    # Check if this task needs lab tests
    if "order_lab_test" not in current_task.required_tools:
        return state
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def process_lab_tests():
        try:
            # Format base prompt
            base_prompt = LAB_TEST_PROMPT.format(
                task_description=current_task.description,
                patient_state=data["patient_state"],
                analysis=json.dumps(data["analysis"])
            )
            
            # Enhance prompt with historical context if memory is available
            if memory:
                enhanced_prompt = enhance_lab_test_prompt(base_prompt, memory, data)
            else:
                enhanced_prompt = base_prompt
            
            # Get LLM response
            result = llm_call(enhanced_prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            # Debug output
            if state["metadata"].get("show_reasoning"):
                st.write("Raw lab test response:", result)
            
            # Extract and parse with enhanced safety
            orders = safe_extract_and_parse_json(
                response=result,
                xml_tag="orders",
                default_value={
                    "test_orders": [],
                    "rationale": "No test orders generated"
                }
            )
            
            # Update memory if available
            if memory and hasattr(memory, "update_working_memory"):
                memory.update_working_memory(
                    state=data["patient_state"],
                    reasoning={
                        "timestamp": datetime.now().isoformat(),
                        "lab_orders": orders
                    },
                    decision={
                        "type": "lab_test_order",
                        "tests_ordered": len(orders["test_orders"]),
                        "rationale": orders["rationale"]
                    }
                )
            
            # Format output if debugging
            if state["metadata"].get("show_reasoning"):
                format_lab_test_output(orders)
            
            return orders
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Lab test processing failed: {str(e)}")
            raise
    
    try:
        # Process lab tests with retries
        orders = process_lab_tests()
        
        # Update task state
        current_task.state = "completed"
        current_task.result = orders
        current_task.completed_at = datetime.now()
        
        # Safely update task queue
        if data["task_queue"]:
            data["task_queue"] = data["task_queue"][1:]  # Remove first item safely
        data["completed_tasks"].append(current_task.id)
        
        # Update memory if available
        if memory and hasattr(memory, "update_patient_history"):
            memory.update_patient_history(
                lab_results={
                    "timestamp": datetime.now().isoformat(),
                    "orders": orders,
                    "task_id": current_task.id
                }
            )
        
    except Exception as e:
        # Create fallback orders on complete failure
        fallback_orders = {
            "test_orders": [{
                "test_name": "manual_review_required",
                "priority": "routine",
                "instructions": f"Lab test processing failed: {str(e)}"
            }],
            "rationale": "Fallback due to processing error"
        }
        
        # Update memory with failure if available
        if memory and hasattr(memory, "update_working_memory"):
            memory.update_working_memory(
                state=data["patient_state"],
                reasoning={
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "fallback": "Using default lab orders"
                },
                decision={
                    "type": "lab_test_failure",
                    "action": "use_defaults",
                    "reason": str(e)
                }
            )
        
        current_task.state = "completed"
        current_task.result = fallback_orders
        current_task.completed_at = datetime.now()
        
        # Safely update task queue
        if data["task_queue"]:
            data["task_queue"] = data["task_queue"][1:]  # Remove first item safely
        data["completed_tasks"].append(current_task.id)
    
    # Create message and update conversation memory
    message_content = json.dumps(current_task.result)
    message = HumanMessage(
        content=message_content,
        name="lab_test_agent"
    )
    
    if memory and hasattr(memory, "add_to_conversation"):
        memory.add_to_conversation(message_content, "ai")
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
        "memory": memory
    }