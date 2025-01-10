import streamlit as st
from typing import Dict, Any, List
from datetime import datetime

def add_timestamp(title: str) -> None:
    """Add a timestamp for agent execution."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"#### {title}")
    st.markdown(f"*Executed at: {timestamp}*")
    st.markdown("---")

def format_analysis_output(analysis: Dict[str, Any]) -> None:
    """Format and display the analysis agent output."""
    # Display patient state with appropriate color
    state_colors = {
        "STABLE": "green",
        "REQUIRES_MONITORING": "yellow",
        "URGENT": "orange",
        "CRITICAL": "red"
    }
    state = analysis.get("patient_state", "STABLE")
    st.markdown(f"### Patient State")
    st.markdown(
        f'<div style="padding:10px;border-radius:5px;background-color:{state_colors[state]};'
        f'color:black;font-weight:bold;text-align:center">{state}</div>',
        unsafe_allow_html=True
    )
    
    # Display immediate concerns
    if analysis.get("immediate_concerns"):
        st.markdown("### Immediate Concerns")
        for concern in analysis["immediate_concerns"]:
            st.error(concern)
    
    # Display risk factors
    if analysis.get("risk_factors"):
        st.markdown("### Risk Factors")
        for risk in analysis["risk_factors"]:
            st.warning(risk)
    
    # Display monitoring needs
    if analysis.get("monitoring_needs"):
        st.markdown("### Monitoring Needs")
        for need in analysis["monitoring_needs"]:
            st.info(need)
    
    # Display preliminary diagnosis
    if analysis.get("preliminary_diagnosis"):
        st.markdown("### Preliminary Diagnosis")
        for diagnosis in analysis["preliminary_diagnosis"]:
            st.markdown(f"- {diagnosis}")
    
    # Display reasoning
    if analysis.get("reasoning"):
        st.markdown("### Clinical Reasoning")
        st.markdown(f"_{analysis['reasoning']}_")

def format_task_output(tasks: List[Dict[str, Any]]) -> None:
    """Format and display the task planning output."""
    st.markdown("### Planned Medical Tasks")
    
    # Group tasks by priority
    priority_tasks = {}
    for task in tasks:
        priority = task.get("priority", 3)
        if priority not in priority_tasks:
            priority_tasks[priority] = []
        priority_tasks[priority].append(task)
    
    # Display tasks by priority
    for priority in sorted(priority_tasks.keys()):
        with st.expander(f"Priority {priority} Tasks", expanded=priority == 1):
            for task in priority_tasks[priority]:
                st.markdown("---")
                st.markdown(f"**Task ID:** {task['id']}")
                st.markdown(f"**Description:** {task['description']}")
                st.markdown("**Required Tools:**")
                for tool in task.get("required_tools", []):
                    st.markdown(f"- {tool}")
                if task.get("dependencies"):
                    st.markdown("**Dependencies:**")
                    for dep in task["dependencies"]:
                        st.markdown(f"- {dep}")

def format_lab_test_output(orders: Dict[str, Any]) -> None:
    """Format and display the lab test orders."""
    st.markdown("### Laboratory Test Orders")
    
    # Display rationale first
    if orders.get("rationale"):
        st.info(f"**Rationale:** {orders['rationale']}")
    
    # Display test orders
    for test in orders.get("test_orders", []):
        with st.expander(f"Test: {test['test_name']}", expanded=True):
            st.markdown(f"**Priority:** {test['priority'].upper()}")
            st.markdown(f"**Instructions:** {test['instructions']}")

def format_referral_output(referrals: Dict[str, Any]) -> None:
    """Format and display the specialist referrals."""
    st.markdown("### Specialist Referrals")
    
    # Display rationale first
    if referrals.get("rationale"):
        st.info(f"**Rationale:** {referrals['rationale']}")
    
    # Display referrals
    for ref in referrals.get("specialist_referrals", []):
        with st.expander(f"Referral: {ref['specialty']}", expanded=True):
            st.markdown(f"**Priority:** {ref['priority'].upper()}")
            st.markdown(f"**Reason:** {ref['reason']}")
            if ref.get("notes"):
                st.markdown(f"**Additional Notes:** {ref['notes']}")

def format_prescription_output(prescriptions: Dict[str, Any]) -> None:
    """Format and display the medication prescriptions."""
    st.markdown("### Medication Orders")
    
    # Display rationale first
    if prescriptions.get("rationale"):
        st.info(f"**Rationale:** {prescriptions['rationale']}")
    
    # Display prescriptions
    for med in prescriptions.get("medication_orders", []):
        with st.expander(f"Medication: {med['medication']}", expanded=True):
            st.markdown(f"**Dosage:** {med['dosage']}")
            st.markdown(f"**Frequency:** {med['frequency']}")
            st.markdown(f"**Duration:** {med['duration']}")
            if med.get("special_instructions"):
                st.markdown(f"**Special Instructions:** {med['special_instructions']}")

def format_care_plan_output(care_plan: Dict[str, Any]) -> None:
    """Format and display the care plan."""
    st.markdown("### Comprehensive Care Plan")
    
    # Display diagnosis
    with st.expander("Diagnosis", expanded=True):
        st.markdown(f"**Primary:** {care_plan['diagnosis']['primary']}")
        if care_plan['diagnosis'].get('secondary'):
            st.markdown("**Secondary Diagnoses:**")
            for diag in care_plan['diagnosis']['secondary']:
                st.markdown(f"- {diag}")
    
    # Display treatment plan
    if care_plan.get("treatment_plan"):
        with st.expander("Treatment Plan", expanded=True):
            for step in care_plan["treatment_plan"]:
                st.markdown(f"- {step}")
    
    # Display monitoring plan
    if care_plan.get("monitoring_plan"):
        with st.expander("Monitoring Plan", expanded=True):
            for item in care_plan["monitoring_plan"]:
                st.markdown(f"- {item}")
    
    # Display follow up
    if care_plan.get("follow_up"):
        with st.expander("Follow-up Instructions", expanded=True):
            for instruction in care_plan["follow_up"]:
                st.markdown(f"- {instruction}")
    
    # Display emergency plan
    if care_plan.get("emergency_plan"):
        with st.expander("Emergency Plan", expanded=False):
            st.error(care_plan["emergency_plan"])