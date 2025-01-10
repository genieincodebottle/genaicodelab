from typing import Dict, Any
import json
import streamlit as st
from langchain_core.messages import HumanMessage

from src.autonomous_agent.utils.constants import TOOL_DEFINITIONS
from src.autonomous_agent.utils.prompts import (
    ANALYSIS_PROMPT,
    TASK_PLANNING_PROMPT,
    LAB_TEST_PROMPT,
    SPECIALIST_REFERRAL_PROMPT,
    PRESCRIPTION_PROMPT,
    CARE_PLAN_PROMPT
)
from src.autonomous_agent.utils.models import MedicalTask
from src.autonomous_agent.memory.memory import AgentState
from src.autonomous_agent.utils.retry_utils import (
    retry_with_exponential_backoff, 
    safe_json_loads, 
    safe_extract_and_parse_json
)
from src.utils.llm import llm_call, extract_xml
from src.autonomous_agent.utils.format_helpers import (
    add_timestamp,
    format_analysis_output,
    format_task_output,
    format_lab_test_output,
    format_referral_output,
    format_prescription_output,
    format_care_plan_output
)

def analysis_agent(state: AgentState) -> Dict[str, Any]:
    """Analyzes patient data and determines initial state."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Analysis Agent")
    
    data = state["data"]
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def run_analysis():
        try:
            # Format the prompt with the medical case
            prompt = ANALYSIS_PROMPT.format(
                case=data["patient_data"]["medical_case"]
            )
            
            # Get the LLM response
            result = llm_call(prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            # Extract and clean up the analysis text
            analysis_text = extract_xml(result, "analysis")
            analysis_text = analysis_text.strip()
            
            # Debug output if requested
            if state["metadata"].get("show_reasoning"):
                st.write("Raw LLM response:", result)
                st.write("Extracted analysis:", analysis_text)
            
            # Parse the JSON with safety measures
            analysis = safe_json_loads(analysis_text)
            
            # Validate the patient state
            valid_states = ["STABLE", "REQUIRES_MONITORING", "URGENT", "CRITICAL"]
            if analysis.get("patient_state") not in valid_states:
                analysis["patient_state"] = "STABLE"
            
            # Ensure all required fields exist
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
            
            # Display formatted analysis if debugging
            if state["metadata"].get("show_reasoning"):
                format_analysis_output(analysis)
                    
            return analysis
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Analysis attempt failed: {str(e)}")
            raise
    
    try:
        # Run analysis with retries
        analysis = run_analysis()
        
    except Exception as e:
        if state["metadata"].get("show_reasoning"):
            st.error(f"All analysis attempts failed: {str(e)}")
        
        # Use default analysis on complete failure
        analysis = {
            "patient_state": "STABLE",
            "immediate_concerns": ["Analysis failed - using default values"],
            "risk_factors": [],
            "monitoring_needs": ["Manual review required"],
            "preliminary_diagnosis": [],
            "reasoning": f"Analysis error: {str(e)}"
        }
    
    # Update state
    data["analysis"] = analysis
    data["patient_state"] = analysis["patient_state"]
    
    # Create message
    message = HumanMessage(
        content=json.dumps(analysis),
        name="analysis_agent"
    )
    
    return {
        "messages": [message],
        "data": data
    }

def task_planning_agent(state: AgentState) -> Dict[str, Any]:
    """Plans medical tasks based on analysis."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Task Planning Agent")
    
    data = state["data"]
    
    # Debug: Show what tools are available
    available_tools = list(TOOL_DEFINITIONS.keys())
    if state["metadata"].get("show_reasoning"):
        st.write("Available tools:", available_tools)
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def run_task_planning():
        try:
            # Format prompt with proper JSON escaping
            prompt = TASK_PLANNING_PROMPT.format(
                patient_state=json.dumps(data["patient_state"]),
                analysis=json.dumps(data["analysis"]),
                tools=json.dumps(available_tools)
            )
            
            # Get LLM response
            result = llm_call(prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            if state["metadata"].get("show_reasoning"):
                st.write("Raw LLM response:", result)
            
            # Extract and parse tasks with enhanced safety
            tasks_data = safe_extract_and_parse_json(
                response=result,
                xml_tag="tasks",
                default_value=[{
                    "id": "task_1",
                    "description": "Initial medical evaluation",
                    "priority": 1,
                    "required_tools": ["order_lab_test"],
                    "dependencies": []
                }]
            )
            
            if not isinstance(tasks_data, list):
                if state["metadata"].get("show_reasoning"):
                    st.error("Tasks data is not a list, using default tasks")
                tasks_data = [{
                    "id": "task_1",
                    "description": "Initial medical evaluation",
                    "priority": 1,
                    "required_tools": ["order_lab_test"],
                    "dependencies": []
                }]
            
            # Validate and create tasks
            tasks = []
            for i, task_data in enumerate(tasks_data, 1):
                # Ensure required fields exist with proper types
                validated_task = {
                    "id": str(task_data.get("id", f"task_{i}")),
                    "description": str(task_data.get("description", f"Task {i}")),
                    "priority": int(task_data.get("priority", 1)),
                    "required_tools": list(task_data.get("required_tools", [])),
                    "dependencies": list(task_data.get("dependencies", [])),
                    "state": "pending"
                }
                
                # Validate tools exist
                validated_task["required_tools"] = [
                    tool for tool in validated_task["required_tools"]
                    if tool in available_tools
                ]
                
                # Create MedicalTask instance
                task = MedicalTask(**validated_task)
                tasks.append(task)
            
            if state["metadata"].get("show_reasoning"):
                st.write("Validated tasks:", [vars(task) for task in tasks])
            
            return tasks
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Task planning attempt failed: {str(e)}")
            raise
    
    try:
        # Run task planning with retries
        tasks = run_task_planning()
        
        # Update state
        data["tasks"] = tasks
        data["task_queue"] = [
            task.id for task in tasks 
            if not task.dependencies
        ]
        data["completed_tasks"] = []
        
        if state["metadata"].get("show_reasoning"):
            st.write("Initial task queue:", data["task_queue"])
            # Show tool usage statistics
            tools_used = set()
            for task in tasks:
                tools_used.update(task.required_tools)
            st.write("Tools used in tasks:", sorted(list(tools_used)))
        
        message = HumanMessage(
            content=json.dumps([vars(task) for task in tasks]),
            name="task_planning_agent"
        )
        
        return {
            "messages": state["messages"] + [message],
            "data": data
        }
        
    except Exception as e:
        if state["metadata"].get("show_reasoning"):
            st.error(f"Task planning failed: {str(e)}")
            
        # Create default fallback tasks
        default_tasks = [
            MedicalTask(
                id="task_1",
                description="Complete medical evaluation",
                priority=1,
                required_tools=["order_lab_test"],
                dependencies=[],
                state="pending"
            )
        ]
        
        data["tasks"] = default_tasks
        data["task_queue"] = ["task_1"]
        data["completed_tasks"] = []
        
        message = HumanMessage(
            content=json.dumps([vars(task) for task in default_tasks]),
            name="task_planning_agent"
        )
        
        return {
            "messages": state["messages"] + [message],
            "data": data
        }

def lab_test_agent(state: AgentState) -> Dict[str, Any]:
    """Orders and processes laboratory tests."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Lab Test Agent")
        
    data = state["data"]
    if not data["task_queue"]:
        return state
        
    current_task = next(
        task for task in data["tasks"]
        if task.id == data["task_queue"][0]
    )
    
    if "order_lab_test" not in current_task.required_tools:
        return state
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def process_lab_tests():
        try:
            prompt = LAB_TEST_PROMPT.format(
                task_description=current_task.description,
                patient_state=data["patient_state"],
                analysis=json.dumps(data["analysis"])
            )
            
            # Get LLM response
            result = llm_call(prompt, 
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
        
        # Update task queue
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
        
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
        
        current_task.state = "completed"
        current_task.result = fallback_orders
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
    
    message = HumanMessage(
        content=json.dumps(current_task.result),
        name="lab_test_agent"
    )
    
    return {
        "messages": state["messages"] + [message],
        "data": data
    }

def specialist_referral_agent(state: AgentState) -> Dict[str, Any]:
    """Manages specialist referrals."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Specialist Referral Agent")
        
    data = state["data"]
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
            prompt = SPECIALIST_REFERRAL_PROMPT.format(
                task_description=current_task.description,
                patient_state=data["patient_state"],
                context=json.dumps(data.get("context", {}))
            )
            
            result = llm_call(prompt, 
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
        
        current_task.state = "completed"
        current_task.result = fallback_referrals
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
    
    message = HumanMessage(
        content=json.dumps(current_task.result),
        name="specialist_referral_agent"
    )
    
    return {
        "messages": state["messages"] + [message],
        "data": data
    }

def prescription_agent(state: AgentState) -> Dict[str, Any]:
    """Manages medication prescriptions."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Prescription Agent")
        
    data = state["data"]
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
            prompt = PRESCRIPTION_PROMPT.format(
                task_description=current_task.description,
                patient_state=data["patient_state"],
                medications=json.dumps(data.get("current_medications", [])),
                context=json.dumps(data.get("context", {}))
            )
            
            result = llm_call(prompt, 
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
        
        # Update task queue
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
        
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
        
        current_task.state = "completed"
        current_task.result = fallback_prescriptions
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
    
    message = HumanMessage(
        content=json.dumps(current_task.result),
        name="prescription_agent"
    )
    
    return {
        "messages": state["messages"] + [message],
        "data": data
    }

def care_plan_agent(state: AgentState) -> Dict[str, Any]:
    """Creates comprehensive care plans."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Care Plan Agent")
        
    data = state["data"]
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
            prompt = CARE_PLAN_PROMPT.format(
                analysis=json.dumps(data["analysis"]),
                completed_tasks=json.dumps(completed_tasks),
                task_results=json.dumps(task_results)
            )
            
            result = llm_call(prompt, 
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
        
        data["care_plan"] = fallback_care_plan
    
    message = HumanMessage(
        content=json.dumps(data["care_plan"]),
        name="care_plan_agent"
    )
    
    return {
        "messages": state["messages"] + [message],
        "data": data
    }