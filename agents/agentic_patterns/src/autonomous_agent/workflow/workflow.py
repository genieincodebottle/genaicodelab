
from uuid import uuid4
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

import streamlit as st
from langgraph.graph import END, StateGraph

# Agent State
from src.autonomous_agent.memory.memory import AgentState

# Agents
from src.autonomous_agent.agents.analysis_agent import analysis_agent
from src.autonomous_agent.agents.task_planning_agent import task_planning_agent
from src.autonomous_agent.agents.lab_test_agent import lab_test_agent
from src.autonomous_agent.agents.specialist_referral_agent import specialist_referral_agent
from src.autonomous_agent.agents.prescription_agent import prescription_agent
from src.autonomous_agent.agents.care_plan_agent import care_plan_agent

def create_medical_workflow() -> StateGraph:
    """Create and return the medical analysis workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("task_planning", task_planning_agent)
    workflow.add_node("lab_tests", lab_test_agent)
    workflow.add_node("specialist_referral", specialist_referral_agent)
    workflow.add_node("prescriptions", prescription_agent)
    workflow.add_node("care_plan", care_plan_agent)
    
    # Define edges with conditional routing
    workflow.add_conditional_edges(
        "analysis",
        should_plan,
        {
            True: "task_planning",
            False: END
        }
    )
    
    # Add edges from task_planning
    workflow.add_conditional_edges(
        "task_planning",
        get_next_step,
        {
            "lab_tests": "lab_tests",
            "specialist_referral": "specialist_referral",
            "prescriptions": "prescriptions",
            "care_plan": "care_plan",
            END: END
        }
    )
    
    # Add edges from other nodes
    for node in ["lab_tests", "specialist_referral", "prescriptions"]:
        workflow.add_conditional_edges(
            node,
            get_next_step,
            {
                "lab_tests": "lab_tests",
                "specialist_referral": "specialist_referral",
                "prescriptions": "prescriptions",
                "care_plan": "care_plan",
                END: END
            }
        )
    
    # Add edge from care_plan to END
    workflow.add_edge("care_plan", END)
    
    # Set the entry point
    workflow.set_entry_point("analysis")
    
    return workflow.compile()

def should_plan(state: AgentState) -> bool:
    """Determine if task planning is needed."""
    return "analysis" in state["data"] and "tasks" not in state["data"]

def get_next_step(state: AgentState) -> str:
    """Determine the next step in the workflow with memory context."""
    data = state["data"]
    memory = state["memory"]
    show_reasoning = state["metadata"].get("show_reasoning", False)
    
    if show_reasoning:
        st.write("Current task queue:", data.get("task_queue"))
        st.write("Completed tasks:", data.get("completed_tasks", []))
        st.write("Current episode state:", memory.working_memory.current_state)
    
    # Check if there are tasks to process
    if not data.get("task_queue"):
        if show_reasoning:
            st.write("No tasks in queue")
        if data.get("completed_tasks"):
            if show_reasoning:
                st.write("Moving to care plan")
            return "care_plan"
        return END
    
    # Get the current task
    current_task = next(
        task for task in data["tasks"]
        if task.id == data["task_queue"][0]
    )
    
    if show_reasoning:
        st.write(f"Current task: {current_task.id}")
        st.write(f"Required tools: {current_task.required_tools}")
    
    # Record decision point in memory
    memory.add_decision_point(
        context={"current_task": current_task.id, "patient_state": data["patient_state"]},
        options=[{"tool": tool} for tool in current_task.required_tools],
        selected={"selected_tool": current_task.required_tools[0] if current_task.required_tools else None},
        reasoning=f"Selected based on task requirements: {current_task.description}",
        confidence=0.8
    )
    
    # Check dependencies with memory context
    if current_task.dependencies:
        pending_deps = [
            dep for dep in current_task.dependencies
            if dep not in data.get("completed_tasks", [])
        ]
        if pending_deps:
            if show_reasoning:
                st.write(f"Task {current_task.id} has pending dependencies: {pending_deps}")
            # Move task to end of queue
            task_id = data["task_queue"].pop(0)
            data["task_queue"].append(task_id)
            return get_next_step(state)
    
    # Route based on required tools
    tools = current_task.required_tools
    if not tools:
        if show_reasoning:
            st.write(f"No tools required for task {current_task.id}")
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
        return get_next_step(state)
    
    for tool in tools:
        if tool == "order_lab_test":
            return "lab_tests"
        elif tool == "refer_specialist":
            return "specialist_referral"
        elif tool == "prescribe_medication":
            return "prescriptions"
    
    # If no specific routing, complete the task
    data["task_queue"].pop(0)
    data["completed_tasks"].append(current_task.id)
    
    return get_next_step(state)