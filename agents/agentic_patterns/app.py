import streamlit as st
from src.utils.llm import LLM_CONFIGS

# Importing all level of Agentic Flow
from src.workflow.prompt_chaining import render_prompt_chain_medical_analysis
from src.workflow.parallelization import render_parallelization_medical_analysis
from src.workflow.query_routing import render_query_routing_medical_analysis
from src.workflow.evaluator_and_optimizer import render_eval_and_optimize_medical_analysis
from src.workflow.orchestrator import render_orchestrator_medical_analysis
from src.workflow.tool_calling import render_tool_calling_medical_analysis
from src.multi_step_agent.multi_step_agent_ui import render_multi_step_agent_medical_analysis
from src.autonomous_agent.autonomous_agent_ui import render_autonomous_multi_agent_medical_analysis
from src.crew_ai_autonomous_agent.crew_ai_ui import render_crew_ai_ui

# Set page config as the first command
st.set_page_config(
    page_title="Sample Agentic Helthcare App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def reset_session_state():
    """
    Ensure all required session state variables exist and 
    initialize them with default values if not present
    """
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.3
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
def render_llm_configuration():
    """Render LLM configuration section in sidebar"""
    st.header("LLM Configuration")
    
    # LLM Provider Selection
    provider = st.selectbox(
        "Select LLM Provider",
        options=list(LLM_CONFIGS.keys()),
        key='selected_llm_provider',
        help="Choose the AI model provider"
    )
    
    # Model Selection based on provider
    available_models = LLM_CONFIGS[provider]["models"]
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        key='selected_llm_model',
        help=f"Choose the specific {provider} model"
    )
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3

    # Temperature configuration
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        key='temperature',
        help="Lower values for more focused responses, higher for more creative ones"
    )
       
    # Model information display
    st.markdown(f"""
    **Current Configuration**:
    - Provider: {provider}
    - Model: {selected_model}
    - Temperature: {temperature}
    
    **Capabilities**:
    - Medical terminology processing
    - Clinical reasoning
    - Evidence-based analysis
    """)

def main():
    # Initialize session state
    reset_session_state()

    st.title("🏥 Sample Healthcare App")
    
    # Custom CSS
    st.markdown("""
        <style>
            body {
                zoom: 0.8;
                -moz-transform: scale(0.8);
                -moz-transform-origin: 0 0;
            }
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 500px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 500px;
                margin-left: -500px;
            }
            .main .block-container {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Noob Workflow to Autonomous Agentic Patterns")
        
        # Workflow Selection
        selected_workflow = st.selectbox(
            "Select Workflow/Agent for Medical Analysis",
            ["Prompt Chaining", 
             "Parallelization",
             "Routing",
             "Evaluator and Optimizers",
             "Orchestrator",
             "Tool Calling",
             "Multi-Step Agent",
             "Autonomous Multi Agent",
             "CrewAI Multi Agent"],
            key='workflow_selector'
        )
        
        # Render LLM Configuration
        st.markdown("---")
        render_llm_configuration()
        
        # Documentation Section
        st.markdown("---")
        st.header("Documentation")
        st.markdown("""
        **Workflow Levels**:
        1. **Prompt Chaining**: Sequential prompts for basic analysis
        2. **Parallelization**: Concurrent analysis tasks
        3. **Query Routing**: Dynamic task distribution
        4. **Evaluator/Optimizer**: Quality control and improvement
        5. **Orchestrator**: Complex workflow management
        6. **Tool Calling**: External tool integration
        7. **Multi-Step Agent**: Advanced decision making
        8. **Autonomous Agent**: Full automation with multiple agents
        
        **Best Practices**:
        - Start with simpler workflows
        - Monitor performance metrics
        - Review agent decisions
        - Adjust temperature as needed
        """)
    
    try:
        # Route to appropriate Agentic Level Workflow
        if selected_workflow == "Prompt Chaining":
            render_prompt_chain_medical_analysis()
        elif selected_workflow == "Parallelization":
            render_parallelization_medical_analysis()
        elif selected_workflow == "Routing":
            render_query_routing_medical_analysis()
        elif selected_workflow == "Evaluator and Optimizers":
            render_eval_and_optimize_medical_analysis()
        elif selected_workflow == "Orchestrator":
            render_orchestrator_medical_analysis()
        elif selected_workflow == "Tool Calling":
            render_tool_calling_medical_analysis()
        elif selected_workflow == "Multi-Step Agent":
            render_multi_step_agent_medical_analysis()
        elif selected_workflow == "Autonomous Multi Agent":
            render_autonomous_multi_agent_medical_analysis()
        elif selected_workflow == "CrewAI Multi Agent":
            render_crew_ai_ui()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.session_state.processing:
            st.session_state.processing = False

if __name__ == "__main__":
    main()