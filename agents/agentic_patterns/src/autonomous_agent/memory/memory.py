from typing import Dict, List, Any, Optional, TypedDict, Sequence, Annotated
from datetime import datetime
import operator
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

from src.autonomous_agent.utils.models import DecisionPoint, PatientHistory

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Utility function for merging dictionaries with nested structures"""
    result = a.copy()
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

class EpisodeMemory(BaseModel):
    """Memory for current medical episode"""
    episode_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    current_state: str = ""
    reasoning_chain: List[Dict[str, Any]] = []
    decision_points: List[Dict[str, Any]] = []
    key_observations: List[Dict[str, Any]] = []
    interventions: List[Dict[str, Any]] = []

class AgentMemory:
    """Comprehensive memory system for medical agents"""
    def __init__(self):
        self.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        self.working_memory = EpisodeMemory(
            episode_id="",
            start_time=datetime.now(),
            end_time=None,
            current_state="",
            reasoning_chain=[],
            decision_points=[],
            key_observations=[],
            interventions=[]
        )
        
        self.patient_history: Optional[PatientHistory] = None
        self.episodic_memory: List[Dict[str, Any]] = []
        self.decision_history: List[DecisionPoint] = []
        
    def add_to_conversation(self, message: str, role: str):
        """Add message to conversation history"""
        if role == "human":
            self.conversation_memory.chat_memory.add_message(
                HumanMessage(content=message)
            )
        elif role == "ai":
            self.conversation_memory.chat_memory.add_message(
                AIMessage(content=message)
            )
            
    def update_working_memory(self, 
                            state: str,
                            reasoning: Dict[str, Any],
                            decision: Dict[str, Any],
                            observation: Optional[Dict[str, Any]] = None):
        """Update working memory with current state and decisions"""
        self.working_memory.current_state = state
        self.working_memory.reasoning_chain.append(reasoning)
        self.working_memory.decision_points.append(decision)
        if observation:
            self.working_memory.key_observations.append(observation)
    
    def add_decision_point(self, 
                         context: Dict[str, Any],
                         options: List[Dict[str, Any]],
                         selected: Dict[str, Any],
                         reasoning: str,
                         confidence: float):
        """Record a clinical decision point"""
        decision = DecisionPoint(
            timestamp=datetime.now(),
            context=context,
            options_considered=options,
            selected_option=selected,
            reasoning=reasoning,
            confidence=confidence,
            outcome=None
        )
        self.decision_history.append(decision)
    
    def save_to_episodic_memory(self):
        """Save current episode to episodic memory for learning"""
        self.working_memory.end_time = datetime.now()
        episode = self.working_memory.model_dump()
        episode["patient_id"] = self.patient_history.patient_id if self.patient_history else None
        self.episodic_memory.append(episode)
    
    def initialize_patient_history(self, patient_id: str):
        """Initialize or load patient history"""
        self.patient_history = PatientHistory(
            patient_id=patient_id,
            visit_dates=[datetime.now()],
            previous_diagnoses=[],
            treatment_history=[],
            medication_history=[],
            lab_results=[],
            vital_signs_history=[],
            allergies=[],
            chronic_conditions=[]
        )
    
    def update_patient_history(self, 
                             diagnosis: Optional[Dict[str, Any]] = None,
                             treatment: Optional[Dict[str, Any]] = None,
                             medications: Optional[Dict[str, Any]] = None,
                             lab_results: Optional[Dict[str, Any]] = None,
                             vitals: Optional[Dict[str, Any]] = None):
        """Update patient history with new information"""
        if not self.patient_history:
            return
            
        if diagnosis:
            self.patient_history.previous_diagnoses.append(diagnosis)
        if treatment:
            self.patient_history.treatment_history.append(treatment)
        if medications:
            self.patient_history.medication_history.append(medications)
        if lab_results:
            self.patient_history.lab_results.append(lab_results)
        if vitals:
            self.patient_history.vital_signs_history.append(vitals)

class AgentState(TypedDict, total=False):
    """State model with memory integration"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]
    memory: AgentMemory