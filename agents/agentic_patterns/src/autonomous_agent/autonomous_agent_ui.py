import streamlit as st
from typing import Dict, Any

from src.autonomous_agent.utils.constants import SAMPLE_CASE
from src.autonomous_agent.workflow.workflow import run_autonomous_medical_analysis
from src.autonomous_agent.memory.memory_persistence import MemoryPersistence

from PIL import Image
from pathlib import Path

def render_workflow_diagram():
    """Render the system workflow diagram."""
    with st.expander("📖 System Workflow", expanded=False):
        st.markdown("Using LangGraph based Agentic Flow")
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'autonomous_agent.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'autonomous_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')


def render_usage_instruction():
    # Add instructions section
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Patient Information Input**
            - Enter unique patient ID
            - Provide detailed medical case information
            - Enable analysis steps display if needed
        
        2. **Analysis Components**
            - 🔍 Initial Analysis: Comprehensive case evaluation
            - 📋 Task Planning: Determine required medical actions
            - 🏥 Task Execution: Process medical tasks
            - 📊 Care Plan Generation: Create final recommendations
        
        3. **Review Results**
            - Check patient status
            - Review analysis details
            - Monitor completed tasks
            - Examine care plan
        
        ### System Features
        
        **🔄 Autonomous Agents**
        - Analysis Agent: Initial case evaluation
        - Task Planning Agent: Action determination
        - Lab Test Agent: Laboratory orders
        - Specialist Referral Agent: Consultations
        - Prescription Agent: Medication management
        - Care Plan Agent: Final recommendations
        
        **📊 Status Levels**
        - 🟢 STABLE: Regular monitoring
        - 🟡 REQUIRES_MONITORING: Enhanced observation
        - 🟠 URGENT: Immediate attention needed
        - 🔴 CRITICAL: Emergency intervention
        
        ### Best Practices
        
        - Provide comprehensive patient information
        - Include relevant medical history
        - Note current symptoms and vital signs
        - Review all agent outputs
        - Check care plan thoroughly
        - Save important analyses
        
        ### Analysis Outputs
        
        1. **📊 Patient Status**
            - Current condition
            - Stability assessment
            - Risk level
        
        2. **📋 Analysis Details**
            - Immediate concerns
            - Risk factors
            - Monitoring needs
            - Preliminary diagnosis
        
        3. **✅ Completed Tasks**
            - Lab test orders
            - Specialist referrals
            - Prescriptions
            - Updates
        
        4. **🏥 Care Plan**
            - Primary diagnosis
            - Treatment recommendations
            - Follow-up instructions
            - Emergency protocols
        """)

def render_patient_history_section(patient_id: str, persistence):
    """Render the patient history section with data from the database."""
    st.header("📋 Patient History")
    
    try:
        # Load patient history
        history = persistence.load_patient_history(patient_id)
        if not history:
            st.info("No previous history found for this patient.")
            return
            
        # Create tabs for different aspects of patient history
        history_tab1, history_tab2, history_tab3, history_tab4 = st.tabs([
            "Previous Diagnoses",
            "Treatment History",
            "Lab Results",
            "Medications"
        ])
        
        with history_tab1:
            st.subheader("Previous Diagnoses")
            if history.get("previous_diagnoses"):
                for diagnosis in history["previous_diagnoses"]:
                    with st.expander(f"Diagnosis from {diagnosis.get('timestamp', 'Unknown Date')}", expanded=False):
                        st.write("**Primary:**", diagnosis.get("diagnosis", {}).get("primary"))
                        if diagnosis.get("diagnosis", {}).get("secondary"):
                            st.write("**Secondary:**")
                            for sec in diagnosis["diagnosis"]["secondary"]:
                                st.write(f"- {sec}")
            else:
                st.info("No previous diagnoses recorded.")
        
        with history_tab2:
            st.subheader("Treatment History")
            if history.get("treatment_history"):
                for treatment in history["treatment_history"]:
                    with st.expander(f"Treatment from {treatment.get('timestamp', 'Unknown Date')}", expanded=False):
                        if treatment.get("plan"):
                            st.write("**Treatment Plan:**")
                            for step in treatment["plan"]:
                                st.write(f"- {step}")
                        if treatment.get("monitoring"):
                            st.write("**Monitoring Plan:**")
                            for item in treatment["monitoring"]:
                                st.write(f"- {item}")
            else:
                st.info("No treatment history recorded.")
        
        with history_tab3:
            st.subheader("Lab Results History")
            if history.get("lab_results"):
                for result in history["lab_results"]:
                    with st.expander(f"Labs from {result.get('timestamp', 'Unknown Date')}", expanded=False):
                        if result.get("orders", {}).get("test_orders"):
                            for test in result["orders"]["test_orders"]:
                                st.markdown(f"""
                                    **Test:** {test.get('test_name')}  
                                    **Priority:** {test.get('priority')}  
                                    **Instructions:** {test.get('instructions')}
                                    ---
                                """)
            else:
                st.info("No lab results recorded.")
        
        with history_tab4:
            st.subheader("Medication History")
            if history.get("medication_history"):
                for med_record in history["medication_history"]:
                    with st.expander(f"Medications from {med_record.get('timestamp', 'Unknown Date')}", expanded=False):
                        if med_record.get("prescriptions"):
                            for med in med_record["prescriptions"]:
                                st.markdown(f"""
                                    **Medication:** {med.get('medication')}  
                                    **Dosage:** {med.get('dosage')}  
                                    **Frequency:** {med.get('frequency')}  
                                    **Duration:** {med.get('duration')}  
                                    **Special Instructions:** {med.get('special_instructions', 'None')}
                                    ---
                                """)
            else:
                st.info("No medication history recorded.")
    except Exception as e:
        st.error(f"Error loading patient history: {str(e)}")

def render_episodes_section(patient_id: str, persistence):
    """Render the medical episodes section with data from the database."""
    st.header("🏥 Medical Episodes")
    
    try:
        # Load episodes
        episodes = persistence.load_episodes(patient_id)
        if not episodes:
            st.info("No previous episodes found for this patient.")
            return
            
        # Create timeline of episodes
        for episode in episodes:
            with st.expander(f"Episode: {episode['episode_id']} ({episode.get('start_time', 'Unknown')})", expanded=False):
                # Display episode state
                state = episode.get('current_state', 'Unknown')
                state_colors = {
                    "STABLE": "🟢",
                    "REQUIRES_MONITORING": "🟡",
                    "URGENT": "🟠",
                    "CRITICAL": "🔴",
                    "Unknown": "⚪"
                }
                st.write(f"**State:** {state_colors.get(state, '⚪')} {state}")
                
                # Create columns for different aspects
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display key observations
                    st.write("**Key Observations:**")
                    observations = episode.get('key_observations', [])
                    if observations:
                        for obs in observations:
                            st.info(f"- {obs.get('type')}: {obs.get('details', '')}")
                    else:
                        st.write("No key observations recorded.")
                
                with col2:
                    # Display interventions
                    st.write("**Interventions:**")
                    interventions = episode.get('interventions', [])
                    if interventions:
                        for intervention in interventions:
                            st.success(f"- {intervention.get('type')}: {intervention.get('details', '')}")
                    else:
                        st.write("No interventions recorded.")
                
                # Display decision points
                st.write("**Decision Points:**")
                decisions = episode.get('decision_points', [])
                if decisions:
                    for decision in decisions:
                        st.warning(
                            f"- Type: {decision.get('type')}  \n"
                            f"  Action: {decision.get('action', 'Not specified')}  \n"
                            f"  Reason: {decision.get('reason', 'Not specified')}"
                        )
                else:
                    st.write("No decision points recorded.")
                
                # Display episode duration
                if episode.get('start_time') and episode.get('end_time'):
                    st.write(f"**Duration:** {episode['start_time']} to {episode['end_time']}")
    except Exception as e:
        st.error(f"Error loading patient history: {str(e)}")
                
def display_medical_history(patient_id: str):
    """Main function to display all medical history data."""
    try:
        persistence = MemoryPersistence()
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Patient History", "Medical Episodes"])
        
        with tab1:
            render_patient_history_section(patient_id, persistence)
            
        with tab2:
            render_episodes_section(patient_id, persistence)
            
    except Exception as e:
        st.error(f"Error loading medical history: {str(e)}")

def render_patient_info_form():
    """Render the patient information input form."""
    with st.form("patient_info"):
        st.markdown("### Patient Information")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            patient_id = st.text_input("Patient ID", "P12345")
            
        medical_case = st.text_area(
            "Medical Case",
            SAMPLE_CASE,
            height=150,
            help="Enter the patient's medical case details here"
        )
        
        col1, col2= st.columns([2, 1])
        with col1:
            show_reasoning = st.checkbox(
                "Show Analysis Steps",
                value=True,
                help="Display detailed analysis steps from each agent"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Start Analysis",
                use_container_width=True
            )
            
    return submitted, patient_id, medical_case, show_reasoning

def display_analysis_results(final_state: Dict[str, Any]):
    """Display the medical analysis results in a well-organized format."""
    st.success("Analysis completed successfully!")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Patient Status",
        "📋 Analysis Details",
        "✅ Completed Tasks",
        "🏥 Care Plan"
    ])
    
    with tab1:
        st.subheader("Current Patient Status")
        state = final_state['data']['patient_state']
        state_colors = {
            "STABLE": "green",
            "REQUIRES_MONITORING": "yellow",
            "URGENT": "orange",
            "CRITICAL": "red"
        }
        st.markdown(
            f'<div style="padding:20px;border-radius:10px;background-color:'
            f'{state_colors[state]};color:black;font-weight:bold;text-align:center">'
            f'{state}</div>',
            unsafe_allow_html=True
        )
    
    with tab2:
        # Check if analysis exists in the data
        if 'analysis' not in final_state['data']:
            st.warning("No analysis data available.")
            return
            
        analysis = final_state['data']['analysis']
        
        # Add a summary section at the top
        st.subheader("Analysis Summary")
        st.markdown(f"**Clinical Reasoning:** {analysis.get('reasoning', 'No reasoning provided')}")
        st.markdown("---")
        
        # Create columns for better organization
        col1, col2 = st.columns(2)
        
        # Display immediate concerns in first column
        with col1:
            if analysis.get('immediate_concerns'):
                with st.expander("🚨 Immediate Concerns", expanded=True):
                    for concern in analysis['immediate_concerns']:
                        st.error(concern, icon="🚨")
            
            # Display risk factors
            if analysis.get('risk_factors'):
                with st.expander("⚠️ Risk Factors", expanded=True):
                    for risk in analysis['risk_factors']:
                        st.warning(risk, icon="⚠️")
        
        # Display monitoring needs and diagnosis in second column
        with col2:
            # Display monitoring needs
            if analysis.get('monitoring_needs'):
                with st.expander("👁️ Monitoring Needs", expanded=True):
                    for need in analysis['monitoring_needs']:
                        st.info(need, icon="👁️")
            
            # Display preliminary diagnosis
            if analysis.get('preliminary_diagnosis'):
                with st.expander("🔍 Preliminary Diagnosis", expanded=True):
                    for diagnosis in analysis['preliminary_diagnosis']:
                        st.write(f"• {diagnosis}")
        
        # Display additional information if available
        if any(key for key in analysis.keys() if key not in ['immediate_concerns', 'risk_factors', 'monitoring_needs', 'preliminary_diagnosis', 'reasoning', 'patient_state']):
            st.markdown("---")
            with st.expander("📌 Additional Information", expanded=False):
                st.json({k: v for k, v in analysis.items() 
                        if k not in ['immediate_concerns', 'risk_factors', 'monitoring_needs', 'preliminary_diagnosis', 'reasoning', 'patient_state']})
    
    with tab3:
        if final_state['data'].get('completed_tasks'):
            for task_id in final_state['data']['completed_tasks']:
                task = next(t for t in final_state['data']['tasks'] if t.id == task_id)
                with st.expander(f"Task: {task.description}", expanded=False):
                    st.json(task.result)
        else:
            st.info("No tasks have been completed yet.")
    
    with tab4:
        if 'care_plan' in final_state['data']:
            care_plan = final_state['data']['care_plan']
            
            # Display diagnosis
            with st.expander("🏥 Diagnosis", expanded=True):
                st.markdown(f"**Primary:** {care_plan['diagnosis']['primary']}")
                if care_plan['diagnosis']['secondary']:
                    st.markdown("**Secondary:**")
                    for diag in care_plan['diagnosis']['secondary']:
                        st.markdown(f"- {diag}")
            
            # Display treatment plan
            if care_plan.get('treatment_plan'):
                with st.expander("💊 Treatment Plan", expanded=True):
                    for step in care_plan['treatment_plan']:
                        st.markdown(f"- {step}")
            
            # Display follow-up
            if care_plan.get('follow_up'):
                with st.expander("📅 Follow-up Instructions", expanded=True):
                    for instruction in care_plan['follow_up']:
                        st.markdown(f"- {instruction}")
            
            # Display emergency plan
            if care_plan.get('emergency_plan'):
                with st.expander("🚑 Emergency Plan", expanded=False):
                    st.error(care_plan['emergency_plan'])
        else:
            st.warning("No care plan has been generated yet.")

def render_autonomous_multi_agent_medical_analysis():
    """Render the Autonomous Agent Medical Analysis interface."""
    
     # First, inject CSS to style the tabs properly
    st.markdown("""
        <style>
            /* Style for the tab container to take full width */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0px;
                width: 100%;
            }
            
            /* Style for the individual tabs to expand fully */
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                flex: 1 1 auto;
                background-color: transparent;
                padding: 10px 20px;
                font-size: 16px;
            }
            
            /* Remove any margin/padding that might affect width */
            .stTabs [data-baseweb="tab-panel"] {
                padding: 15px 0px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Autonomous Agent - Medical Analysis")

    # Create main tabs for different sections
    main_tab1, main_tab2 = st.tabs(["New Analysis", "Patient History"])
    
    with main_tab1:
        # Render workflow diagram
        render_workflow_diagram()
        # Render usage instructions
        render_usage_instruction()
        
        # Get input from form
        submitted, patient_id, medical_case, show_reasoning = render_patient_info_form()
        
        if submitted:
            if medical_case and patient_id:
                try:
                    # Initialize progress tracking
                    progress_text = "Analysis in progress. Please wait..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    # Run the analysis
                    with st.spinner():
                        final_state = run_autonomous_medical_analysis(
                            patient_data={
                                "patient_id": patient_id,
                                "medical_case": medical_case
                            },
                            show_reasoning=show_reasoning
                        )
                        
                        my_bar.progress(100, text="Analysis complete!")
                        
                        if final_state:
                            display_analysis_results(final_state)
                        else:
                            st.error("Analysis could not be completed.")
                            
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.exception(e)
            else:
                st.warning("Please enter both patient ID and medical case details.")
    
    with main_tab2:
        # Add patient ID input for history view
        history_patient_id = st.text_input(
            "Enter Patient ID to view history",
            key="history_patient_id"
        )
        
        if history_patient_id:
            display_medical_history(history_patient_id)
        else:
            st.info("Please enter a patient ID to view their medical history.")