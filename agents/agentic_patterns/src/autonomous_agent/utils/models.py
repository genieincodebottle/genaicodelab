from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class PatientState(str, Enum):
    STABLE = "stable"
    REQUIRES_MONITORING = "requires_monitoring"
    URGENT = "urgent"
    CRITICAL = "critical"

@dataclass
class MedicalTask:
    """Represents a medical task in the workflow"""
    id: str
    description: str
    priority: int
    required_tools: List[str]
    dependencies: List[str]
    state: str = "pending"
    result: Optional[Dict] = None
    created_at: datetime = datetime.now()
    completed_at: Optional[datetime] = None
    reasoning: Optional[str] = None

class PatientHistory(BaseModel):
    """Long-term patient history tracking"""
    patient_id: str
    visit_dates: List[datetime]
    previous_diagnoses: List[Dict[str, Any]]
    treatment_history: List[Dict[str, Any]]
    medication_history: List[Dict[str, Any]]
    lab_results: List[Dict[str, Any]]
    vital_signs_history: List[Dict[str, Any]]
    allergies: List[str]
    chronic_conditions: List[str]

class DecisionPoint(BaseModel):
    """Represents a clinical decision point"""
    timestamp: datetime
    context: Dict[str, Any]
    options_considered: List[Dict[str, Any]]
    selected_option: Dict[str, Any]
    reasoning: str
    confidence: float
    outcome: Optional[Dict[str, Any]]

