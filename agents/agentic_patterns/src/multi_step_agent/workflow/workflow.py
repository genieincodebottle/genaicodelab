import os
from typing import Dict, Tuple, Optional
import json
import logging

from src.multi_step_agent.memory.memory import MedicalMemory
from src.multi_step_agent.utils.prompts import MULTI_STEP_AGENT_PROMPT
from src.utils.llm import llm_call, extract_xml
from src.utils.tools import execute_medical_tool

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOGGING_LEVEL"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MultiStepAgent:
    """Agent that can execute multi-step medical workflows"""
    
    def __init__(self, temperature: float, provider: str, model: str):
        self.temperature = temperature
        self.provider = provider
        self.model = model
        
    def validate_action(self, action: Dict, memory: MedicalMemory) -> bool:
        """Validate if action should be executed"""
        tool_name = action["function"]
        args = action["args"]
        
        if tool_name == "schedule_appointment":
            if memory.has_appointment(args["department"], args["urgency"]):
                return False
        elif tool_name == "order_lab_test":
            if memory.has_test(args["test_type"]):
                return False
        elif tool_name == "refer_specialist":
            if memory.has_referral(args["specialty"]):
                return False
        elif tool_name == "update_patient_record":
            update_types = set(args["data"].keys())
            if all(memory.has_update(update_type) for update_type in update_types):
                return False
                
        return True
        
    def should_continue(self, memory: MedicalMemory, show_reasoning: bool = True) -> Tuple[bool, str, Optional[Dict]]:
        """Determine if workflow should continue and get next action"""
        try:
            prompt = MULTI_STEP_AGENT_PROMPT.format(
                context=memory.get_context(),
                patient_id=memory.steps[0].args["patient_id"] if memory.steps else "PATIENT_ID"
            )
            
            response = llm_call(prompt, self.temperature, self.provider, self.model)
            analysis = extract_xml(response, "analysis")
            should_continue = extract_xml(response, "continue").strip().upper() == "YES"
            
            if show_reasoning:
                logger.info("Step Analysis")
                logger.info(analysis)
            
            next_action = None
            if should_continue:
                action_json = extract_xml(response, "next_action")
                action_data = json.loads(action_json)
                
                if not self.validate_action(action_data, memory):
                    return False, "All necessary actions have been completed.", None
                    
                next_action = action_data
                
            return should_continue, analysis, next_action
            
        except Exception as e:
            logger.error(f"Error in continuation logic: {str(e)}")
            return False, str(e), None
            
    def execute_workflow(self, task: str, patient_id: str, show_reasoning: bool = True) -> MedicalMemory:
        """Execute multi-step agent workflow"""
        memory = MedicalMemory(task)
        max_steps = 10
        step_count = 0
        
        logger.info(f"Starting step {step_count + 1}/{max_steps}")
        
        while step_count < max_steps:
            if show_reasoning:
                logger.info(f"Processing step {step_count + 1}/{max_steps}")
            
            should_continue, analysis, next_action = self.should_continue(memory, show_reasoning)
            
            if not should_continue or next_action is None:
                logger.info("Workflow complete!")
                break
                
            try:
                if show_reasoning:
                    logger.info("Executing Next Action")
                    logger.info(json.dumps(next_action))
                
                next_action["args"]["patient_id"] = patient_id
                result = execute_medical_tool(next_action["function"], next_action["args"])
                
                memory.add_step(
                    next_action["function"],
                    next_action["args"],
                    result
                )
                
                if show_reasoning:
                    logger.info(f"Action completed: {result}")
                    
                step_count += 1
                
            except Exception as e:
                logger.error(f"Error executing action: {str(e)}")
                break
                
        if step_count >= max_steps:
            logger.warning("Workflow reached maximum number of steps")
            
        return memory