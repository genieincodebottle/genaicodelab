import streamlit as st
import subprocess
import os

# Page configuration with custom theme
st.set_page_config(
    page_title="GenAI Code Lab",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
        .main {
            padding: 0.5rem;
        }
        .stButton>button {
            width: 200px !important;
            height: 2rem;
            font-size: 0.9rem;
            margin: 1rem auto !important;
            display: block;
        }
        .highlight {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        /* Custom heading sizes */
        h1 {
            font-size: 1.8rem !important;
        }
        h2 {
            font-size: 1.4rem !important;
            margin-top: 0 !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        h4 {
            font-size: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        p {
            font-size: 0.9rem !important;
        }
        .stMarkdown {
            font-size: 0.9rem;
        }
        /* Feature card styling */
        .element-container {
            margin-bottom: 0.5rem;
        }
        /* Remove default padding from the header */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        /* Reduce space between header elements */
        div[data-testid="stMarkdownContainer"] > * {
            margin-bottom: 0.3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🧪 GenAI Code Lab", unsafe_allow_html=True)
    st.markdown("""
        You are here because you love to learn. Welcome to GenAI Code Lab – let's explore and create together!
    """)

# Main Features Section
st.markdown("---")
st.markdown("### 🚀 Available Tools")

# Create three columns for features
col1, spacer1, col2, spacer2, col3, spacer3, col4= st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1])

def create_feature_card(title, description, button_label, button_key, is_disabled=True, on_click=None):
    st.markdown(f"#### {title}")
    container = st.container()
    with container:
        # Fixed height container for description with left alignment
        st.markdown(f"""
            <div style='height: 100px; overflow: auto; text-align: left;'>
                {description}
            </div>
        """, unsafe_allow_html=True)
        
        # Button with consistent positioning
        if on_click:
            if st.button(button_label, key=button_key, disabled=is_disabled):
                on_click()
        else:
            st.button(button_label, key=button_key, disabled=is_disabled)

def launch_agentic_rag():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, "agentic_app.py")
        subprocess.Popen(["streamlit", "run", app_path])
        st.success("✅ Agentic-RAG launched successfully! Check your browser for the new tab.")
    except Exception as e:
        st.error(f"❌ Error launching app: {str(e)}")

def launch_non_agentic_rag():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, "non_agentic_app.py")
        subprocess.Popen(["streamlit", "run", app_path])
        st.success("✅ Non Agentic RAG launched successfully! Check your browser for the new tab.")
    except Exception as e:
        st.error(f"❌ Error launching app: {str(e)}")

with col1:
    create_feature_card(
        "🤖 Agentic-RAG",
        "Advanced Retrieval-Augmented Generation with autonomous agent capabilities. Perfect for complex document analysis and intelligent responses.",
        "Launch Agentic-RAG",
        "agentic_rag_button",
        False,
        launch_agentic_rag
    )

with col2:
    create_feature_card(
        "📊 Non Agentic-RAG",
        "Powerful data analysis tools with AI-driven insights. Transform your raw data into actionable intelligence. *(Coming Soon)*",
        "Non Agentic RAG",
        "data_analysis_button",
        False,
        launch_non_agentic_rag
    )

with col3:
    create_feature_card(
        "🔍 Multimodal RAG",
        "Interactive environment to experiment with different AI models and parameters. Fine-tune your models for optimal performance. *(Coming Soon)*",
        "Coming Soon",
        "model_explorer_button"
    )
with col4:
    create_feature_card(
        "🔍 Prompt Engineering",
        "Interactive environment to experiment with different AI models and parameters. Fine-tune your models for optimal performance. *(Coming Soon)*",
        "Coming Soon",
        "model_explorer_button"
    )

# Quick Start Guide
st.markdown("---")
st.markdown("### 🏃‍♂️ Quick Start Guide")
with st.expander("How to Get Started"):
    st.markdown("""
    1. **Launch Agentic-RAG**: Click the 'Launch Agentic-RAG' button above to start the application
    2. **Upload Documents**: In the new window, upload your documents for analysis
    3. **Ask Questions**: Interact with the AI to analyze your documents
    4. **Export Results**: Download your analysis results and insights
    
    Need help? Check out our [documentation](https://docs.example.com) *(coming soon)*
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: grey;'>
        Made with ❤️ by GenAI Code Lab Team | Version 1.0.0
    </div>
""", unsafe_allow_html=True)