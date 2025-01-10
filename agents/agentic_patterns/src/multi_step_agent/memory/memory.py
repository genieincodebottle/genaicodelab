from dataclasses import dataclass
from typing import List, Optional, Set
import json

@dataclass
class ActionStep:
    """Represents a single step in the medical workflow"""
    tool_name: str
    args: dict
    observation: Optional[str] = None

class MedicalMemory:
    """Manages the history of actions and observations with state tracking"""
    def __init__(self, initial_task: str):
        self.task = initial_task
        self.steps: List[ActionStep] = []
        self.scheduled_appointments: Set[tuple] = set()  # (department, urgency)
        self.ordered_tests: Set[str] = set()  # test types
        self.specialist_referrals: Set[str] = set()  # specialties
        self.completed_updates: Set[str] = set()  # Track types of updates
        
    def has_appointment(self, department: str, urgency: str) -> bool:
        """Check if appointment already scheduled"""
        return (department, urgency) in self.scheduled_appointments
        
    def has_test(self, test_type: str) -> bool:
        """Check if test already ordered"""
        return test_type in self.ordered_tests
        
    def has_referral(self, specialty: str) -> bool:
        """Check if referral already made"""
        return specialty in self.specialist_referrals
        
    def has_update(self, update_type: str) -> bool:
        """Check if specific update already done"""
        return update_type in self.completed_updates
        
    def add_step(self, tool_name: str, args: dict, observation: str):
        """Add step with duplicate prevention"""
        if tool_name == "schedule_appointment":
            self.scheduled_appointments.add((args["department"], args["urgency"]))
        elif tool_name == "order_lab_test":
            self.ordered_tests.add(args["test_type"])
        elif tool_name == "refer_specialist":
            self.specialist_referrals.add(args["specialty"])
        elif tool_name == "update_patient_record":
            update_types = set(args["data"].keys())
            self.completed_updates.update(update_types)
            
        self.steps.append(ActionStep(tool_name, args, observation))
        
    def get_context(self) -> str:
        """Get formatted context of all previous steps"""
        context = [f"Initial Task: {self.task}\n"]
        context.append("\nCompleted Actions Summary:")
        if self.scheduled_appointments:
            context.append("Appointments Scheduled:")
            for dept, urgency in self.scheduled_appointments:
                context.append(f"- {urgency} appointment with {dept}")
        if self.ordered_tests:
            context.append("\nTests Ordered:")
            for test in self.ordered_tests:
                context.append(f"- {test}")
        if self.specialist_referrals:
            context.append("\nReferrals Made:")
            for specialty in self.specialist_referrals:
                context.append(f"- {specialty}")
        if self.completed_updates:
            context.append("\nRecord Updates:")
            for update in self.completed_updates:
                context.append(f"- Updated {update}")
        
        context.append("\nDetailed Step History:")
        for i, step in enumerate(self.steps, 1):
            context.append(f"\nStep {i}:")
            context.append(f"Action: {step.tool_name}")
            context.append(f"Parameters: {json.dumps(step.args)}")
            context.append(f"Result: {step.observation}")
        
        return "\n".join(context)