import os
from uuid import uuid4
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
import logging
from functools import wraps
from datetime import datetime

# Agent State & Model
from src.autonomous_agent.memory.memory import AgentState, AgentMemory
from src.autonomous_agent.memory.memory_persistence import MemoryPersistence
from src.autonomous_agent.utils.models import PatientHistory
from langgraph.graph import END, StateGraph

# Agents
from src.autonomous_agent.agents.analysis_agent import analysis_agent
from src.autonomous_agent.agents.task_planning_agent import task_planning_agent
from src.autonomous_agent.agents.lab_test_agent import lab_test_agent
from src.autonomous_agent.agents.specialist_referral_agent import specialist_referral_agent
from src.autonomous_agent.agents.prescription_agent import prescription_agent
from src.autonomous_agent.agents.care_plan_agent import care_plan_agent

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOGGING_LEVEL"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Starting execution of {func.__name__}")
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class WorkflowError(Exception):
    """Custom exception for workflow-related errors."""
    pass

@log_execution_time
def create_medical_workflow() -> StateGraph:
    """Create and return the medical analysis workflow."""
    logger.info("Initializing medical workflow")
    workflow = StateGraph(AgentState)
    
    # Add nodes
    nodes = {
        "analysis": analysis_agent,
        "task_planning": task_planning_agent,
        "lab_tests": lab_test_agent,
        "specialist_referral": specialist_referral_agent,
        "prescriptions": prescription_agent,
        "care_plan": care_plan_agent
    }
    
    for node_name, agent in nodes.items():
        workflow.add_node(node_name, agent)
        logger.debug(f"Added node: {node_name}")
    
    # Define edges with conditional routing
    workflow.add_conditional_edges(
        "analysis",
        should_plan,
        {True: "task_planning", False: END}
    )
    
    # Add edges from task_planning and other nodes
    edge_config = {
        "lab_tests": "lab_tests",
        "specialist_referral": "specialist_referral",
        "prescriptions": "prescriptions",
        "care_plan": "care_plan",
        END: END
    }
    
    nodes_with_conditional_edges = ["task_planning", "lab_tests", "specialist_referral", "prescriptions"]
    for node in nodes_with_conditional_edges:
        workflow.add_conditional_edges(node, get_next_step, edge_config)
    
    workflow.add_edge("care_plan", END)
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
    
    logger.debug(f"Current task queue: {data.get('task_queue')}")
    logger.debug(f"Completed tasks: {data.get('completed_tasks', [])}")
    
    if not data.get("task_queue"):
        logger.info("No tasks in queue")
        return "care_plan" if data.get("completed_tasks") else END
    
    try:
        current_task = next(
            task for task in data["tasks"]
            if task.id == data["task_queue"][0]
        )
    except StopIteration:
        logger.error("Could not find current task in task list")
        raise WorkflowError("Task queue and task list mismatch")
    
    logger.info(f"Processing task: {current_task.id}")
    
    # Record decision point in memory
    memory.add_decision_point(
        context={"current_task": current_task.id, "patient_state": data["patient_state"]},
        options=[{"tool": tool} for tool in current_task.required_tools],
        selected={"selected_tool": current_task.required_tools[0] if current_task.required_tools else None},
        reasoning=f"Selected based on task requirements: {current_task.description}",
        confidence=0.8
    )
    
    # Check dependencies
    if current_task.dependencies:
        pending_deps = [
            dep for dep in current_task.dependencies
            if dep not in data.get("completed_tasks", [])
        ]
        if pending_deps:
            logger.info(f"Task {current_task.id} has pending dependencies: {pending_deps}")
            task_id = data["task_queue"].pop(0)
            data["task_queue"].append(task_id)
            return get_next_step(state)
    
    # Route based on required tools
    tools = current_task.required_tools
    if not tools:
        logger.info(f"No tools required for task {current_task.id}")
        data["task_queue"].pop(0)
        data["completed_tasks"].append(current_task.id)
        return get_next_step(state)
    
    tool_routing = {
        "order_lab_test": "lab_tests",
        "refer_specialist": "specialist_referral",
        "prescribe_medication": "prescriptions"
    }
    
    for tool in tools:
        if tool in tool_routing:
            return tool_routing[tool]
    
    # If no specific routing, complete the task
    data["task_queue"].pop(0)
    data["completed_tasks"].append(current_task.id)
    
    return get_next_step(state)

@log_execution_time
def run_autonomous_medical_analysis(
    patient_data: Dict[str, Any],
    show_reasoning: bool = False,
    use_memory: bool = True
) -> Dict[str, Any]:
    """Run the medical analysis workflow."""
    
    workflow = create_medical_workflow()
    episode_id = str(uuid4())
    memory = None
    persistence = None
    
    logger.info(f"Starting medical analysis for patient {patient_data['patient_id']}")
    
    try:
        # Initialize memory
        memory = AgentMemory()
        memory.working_memory.episode_id = episode_id
        memory.initialize_patient_history(patient_data["patient_id"])
        
        if use_memory:
            try:
                persistence = MemoryPersistence()
                existing_history = persistence.load_patient_history(patient_data["patient_id"])
                if existing_history:
                    memory._patient_history = PatientHistory(**existing_history)
                    logger.info("Loaded existing patient history")
                
                previous_episodes = persistence.load_episodes(patient_data["patient_id"])
                if previous_episodes:
                    memory.episodic_memory.extend(previous_episodes)
                    logger.info(f"Loaded {len(previous_episodes)} previous episodes")
                    
            except Exception as e:
                logger.warning(f"Memory persistence error: {str(e)}. Continuing with temporary memory.")

        initial_state = {
            "messages": [
                HumanMessage(
                    content="Analyze patient data and create care plan by taking autonomous decision to call different agents.",
                )
            ],
            "data": {
                "patient_data": patient_data,
                "completed_tasks": [],
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            },
            "memory": memory
        }
        
        final_state = workflow.invoke(initial_state)

        # Save episode to memory if enabled
        if use_memory and persistence is not None:
            try:
                memory.save_to_episodic_memory()
                persistence.save_patient_history(
                    patient_data["patient_id"],
                    memory.patient_history.model_dump() if memory.patient_history else {}
                )
                persistence.save_episode(
                    episode_id,
                    patient_data["patient_id"],
                    memory.working_memory.model_dump()
                )
                logger.info("Successfully saved episode to persistent memory")
            except Exception as e:
                logger.error(f"Failed to save to persistent memory: {str(e)}")

        return final_state
        
    except Exception as e:
        logger.error(f"Critical error in medical analysis: {str(e)}")
        raise WorkflowError(f"Medical analysis failed: {str(e)}")