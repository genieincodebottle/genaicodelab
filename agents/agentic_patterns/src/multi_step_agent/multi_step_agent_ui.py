import streamlit as st

from src.multi_step_agent.utils.constants import SAMPLE_MEDICAL_REPORT
from src.multi_step_agent.memory.memory import MedicalMemory
from src.multi_step_agent.workflow.workflow import MultiStepAgent

from PIL import Image
from pathlib import Path

def render_workflow_diagram():
    """Render the multi-step agent workflow diagram"""
    with st.expander("📖 System Workflow", expanded=False):
        st.markdown("Using LangGraph based Agentic Flow")
        # Get the relative path to the image
        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir.parent.parent / 'images'
                
        routing_diagram = Image.open(image_path/ 'multi_step_agent.png')
        st.image(routing_diagram, caption='High Level Architecture')
                
        sequence_diagram = Image.open(image_path/ 'multi_step_sequence_diagram.png')
        st.image(sequence_diagram, caption='Sequence Diagram')

def render_usage_instruction():
    with st.expander("📖 Usage Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        
        1. **Patient Information**
            - Enter a Patient ID or use the default
            - Use the sample case or enter your own medical case details
        
        2. **Configuration**
            - Check "Show Analysis Steps" to see detailed reasoning
            - Adjust temperature in sidebar if needed
        
        3. **Results**
            - View the complete workflow summary
            - Check detailed actions taken
            - Review statistics and timeline
        
        4. **Understanding Outputs**
            - 📋 Workflow Summary: Complete overview of all actions
            - 🏥 Actions Taken: Detailed list of all medical decisions
            - 📊 Statistics: Numerical overview of actions taken
        
        ### Available Medical Tools
        
        - 📅 Schedule Appointments
        - 🔬 Order Lab Tests
        - 👨‍⚕️ Make Specialist Referrals
        - 📝 Update Patient Records
        
        ### Best Practices
        
        - Provide detailed patient information
        - Review all actions in the summary
        - Check for any missed steps
        - Verify appointments and referrals
        """)

def render_patient_info_form():
    """Render the patient information input form"""
    with st.form("patient_info"):
        st.markdown("### Patient Information")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            patient_id = st.text_input("Patient ID", "P12345")
            
        use_sample = st.checkbox("Use sample medical case", value=True)
        if use_sample:
            medical_case = SAMPLE_MEDICAL_REPORT
            st.text_area(
                "Sample Case (read-only)", 
                medical_case, 
                height=150, 
                disabled=True,
                help="This is a sample medical case for demonstration"
            )
        else:
            medical_case = st.text_area(
                "Medical Case",
                """Patient Case:
                [Detailed patient information]
                [Clinical findings]
                [Relevant history]
                [Current status]""",
                height=150,
                help="Enter the patient's medical case details here"
            )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            show_reasoning = st.checkbox(
                "Show Analysis Steps",
                value=True,
                help="Display detailed analysis steps from the agent"
            )
            
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Start Analysis",
                use_container_width=True
            )
            
    return submitted, patient_id, medical_case, show_reasoning

def display_workflow_results(memory: MedicalMemory):
    """Display the multi-step workflow results in a well-organized format"""
    st.success("Workflow completed successfully!")
    
    tab1, tab2, tab3 = st.tabs([
        "📋 Workflow Summary",
        "🏥 Actions Taken",
        "📊 Statistics"
    ])
    
    with tab1:
        st.subheader("Workflow Overview")
        st.markdown(memory.get_context())
    
    with tab2:
        st.subheader("Detailed Actions")
        
        # Group actions by type
        appointments = []
        tests = []
        referrals = []
        updates = []
        
        for step in memory.steps:
            if step.tool_name == "schedule_appointment":
                appointments.append(step)
            elif step.tool_name == "order_lab_test":
                tests.append(step)
            elif step.tool_name == "refer_specialist":
                referrals.append(step)
            elif step.tool_name == "update_patient_record":
                updates.append(step)
        
        # Display appointments
        if appointments:
            with st.expander("📅 Scheduled Appointments", expanded=True):
                for step in appointments:
                    st.info(
                        f"**{step.args['urgency'].title()} appointment** with "
                        f"**{step.args['department']}**\n\n"
                        f"_{step.observation}_"
                    )
        
        if tests:
            with st.expander("🔬 Ordered Tests", expanded=True):
                for step in tests:
                    st.warning(
                        f"**{step.args['test_type']}**\n\n"
                        f"_{step.observation}_"
                    )
        
        if referrals:
            with st.expander("👨‍⚕️ Specialist Referrals", expanded=True):
                for step in referrals:
                    st.success(
                        f"**Referred to {step.args['specialty']}**\n\n"
                        f"_{step.observation}_"
                    )
        
        if updates:
            with st.expander("📝 Record Updates", expanded=True):
                for step in updates:
                    st.info(
                        f"**Updated:** {', '.join(step.args['data'].keys())}\n\n"
                        f"_{step.observation}_"
                    )
    
    with tab3:
        st.subheader("Workflow Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Steps", len(memory.steps))
        with col2:
            st.metric("Appointments", len(appointments))
        with col3:
            st.metric("Lab Tests", len(tests))
        with col4:
            st.metric("Referrals", len(referrals))
        
        st.markdown("### Action Timeline")
        for i, step in enumerate(memory.steps, 1):
            st.markdown(
                f"**Step {i}:** {step.tool_name} → "
                f"_{step.observation[:50]}{'...' if len(step.observation) > 50 else ''}_"
            )

def render_multi_step_agent_medical_analysis():
    """Render the Multi-step Agent orchestration interface"""
    st.subheader("Multi-Step Agent - Medical Analysis")
    
    # Render workflow diagram
    render_workflow_diagram()
    # Render Usage Instructions
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
                with st.spinner("Executing multi-step workflow..."):
                    agent = MultiStepAgent(temperature=st.session_state.temperature, 
                                           provider=st.session_state.selected_llm_provider,
                                           model=st.session_state.selected_llm_model)
                    memory = agent.execute_workflow(medical_case, patient_id, show_reasoning)
                    
                    my_bar.progress(100, text="Analysis complete!")
                    
                    if memory:
                        display_workflow_results(memory)
                    else:
                        st.error("Workflow could not be completed.")
                        
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)
        else:
            st.warning("Please enter both patient ID and medical case details.")