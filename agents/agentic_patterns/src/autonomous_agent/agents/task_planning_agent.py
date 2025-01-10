from typing import Dict, Any, List
from datetime import datetime
import streamlit as st
from langchain_core.messages import HumanMessage

from src.autonomous_agent.utils.constants import TOOL_DEFINITIONS
from src.autonomous_agent.utils.prompts import TASK_PLANNING_PROMPT
from src.autonomous_agent.utils.models import MedicalTask
from src.autonomous_agent.memory.memory import AgentState
from src.autonomous_agent.utils.retry_utils import (
    retry_with_exponential_backoff,
    safe_extract_and_parse_json
)
from src.autonomous_agent.utils.json_utils import json_serialize
from src.utils.llm import llm_call
from src.autonomous_agent.utils.format_helpers import add_timestamp

def task_dict_to_json(task: MedicalTask) -> Dict[str, Any]:
    """Convert a task to a JSON-serializable dictionary."""
    return {
        "id": task.id,
        "description": task.description,
        "priority": task.priority,
        "required_tools": task.required_tools,
        "dependencies": task.dependencies,
        "state": task.state,
        "result": task.result,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "reasoning": task.reasoning
    }

def task_planning_agent(state: AgentState) -> Dict[str, Any]:
    """Plans medical tasks with memory integration."""
    if state["metadata"].get("show_reasoning"):
        add_timestamp("Task Planning Agent")
    
    data = state["data"]
    memory = state["memory"]
    
    available_tools = list(TOOL_DEFINITIONS.keys())
    if state["metadata"].get("show_reasoning"):
        st.write("Available tools:", available_tools)
    
    @retry_with_exponential_backoff(max_retries=3, show_progress=True)
    def run_task_planning():
        try:
            # Format base prompt
            base_prompt = TASK_PLANNING_PROMPT.format(
                patient_state=json_serialize(data["patient_state"]),
                analysis=json_serialize(data["analysis"]),
                tools=json_serialize(available_tools)
            )
            
            # Get LLM response
            result = llm_call(base_prompt, 
                              temperature=st.session_state.temperature,
                              provider=st.session_state.selected_llm_provider,
                              model=st.session_state.selected_llm_model)
            
            if state["metadata"].get("show_reasoning"):
                st.write("Raw LLM response:", result)
            
            # Extract and parse tasks
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
            
            # Validate and create tasks
            tasks = []
            for i, task_data in enumerate(tasks_data, 1):
                task = MedicalTask(
                    id=str(task_data.get("id", f"task_{i}")),
                    description=str(task_data.get("description", f"Task {i}")),
                    priority=int(task_data.get("priority", 1)),
                    required_tools=[
                        tool for tool in task_data.get("required_tools", [])
                        if tool in available_tools
                    ],
                    dependencies=list(task_data.get("dependencies", [])),
                    state="pending",
                    created_at=datetime.now()
                )
                tasks.append(task)
            
            # Update working memory
            memory.update_working_memory(
                state=data["patient_state"],
                reasoning={
                    "timestamp": datetime.now().isoformat(),
                    "planned_tasks": [task_dict_to_json(task) for task in tasks]
                },
                decision={
                    "type": "task_planning",
                    "tasks": len(tasks),
                    "tools_required": list(set(
                        tool for task in tasks
                        for tool in task.required_tools
                    ))
                }
            )
            
            return tasks
            
        except Exception as e:
            if state["metadata"].get("show_reasoning"):
                st.error(f"Task planning attempt failed: {str(e)}")
            raise
    
    try:
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
            tools_used = set()
            for task in tasks:
                tools_used.update(task.required_tools)
            st.write("Tools used in tasks:", sorted(list(tools_used)))
        
        # Create message
        message_content = json_serialize([task_dict_to_json(task) for task in tasks])
        message = HumanMessage(
            content=message_content,
            name="task_planning_agent"
        )
        
        memory.add_to_conversation(message_content, "ai")
        
        return {
            "messages": state["messages"] + [message],
            "data": data,
            "memory": memory
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
                state="pending",
                created_at=datetime.now()
            )
        ]
        
        # Update memory with failure information
        memory.update_working_memory(
            state=data["patient_state"],
            reasoning={
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "fallback": "Using default tasks"
            },
            decision={
                "type": "task_planning_failure",
                "action": "use_defaults",
                "reason": str(e)
            }
        )
        
        data["tasks"] = default_tasks
        data["task_queue"] = ["task_1"]
        data["completed_tasks"] = []
        
        # Create message with proper serialization
        message_content = json_serialize([task_dict_to_json(task) for task in default_tasks])
        message = HumanMessage(
            content=message_content,
            name="task_planning_agent"
        )
        
        memory.add_to_conversation(message_content, "ai")
        
        return {
            "messages": state["messages"] + [message],
            "data": data,
            "memory": memory
        }