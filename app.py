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
        body {
            zoom: 0.9;
            -moz-transform: scale(0.9);
            -moz-transform-origin: 0 0;
        }
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

# Header Section with Logo
st.image("static/images/logo.png", width=200)  # Add logo image at the top
col1, col2 = st.columns([2, 1])

with col1:
    #st.markdown("## 🧪 AilluminatiLab - GenAI Code Lab", unsafe_allow_html=True)
    st.markdown("""
        Welcome to GenAI Code Lab – let's explore and create together :)
    """)

# Main Features Section
st.markdown("---")
st.markdown("### ✨ Resources")

# Create three columns for features
col1, spacer1, col2, spacer2, col3, spacer3, col4= st.columns([2, 0.1, 2, 0.1, 2, 0.1, 2])

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

def launch_agents():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, "agents/agentic_patterns/app.py")
        subprocess.Popen(["streamlit", "run", app_path])
        st.success("✅ Agent App launched successfully! Check your browser for the new tab.")
    except Exception as e:
        st.error(f"❌ Error launching app: {str(e)}")

with col1:
    create_feature_card(
        "🤖 Noob to Autonomous Agents",
        "Guide to learning and building different Agentic Patterns",
        "Launch",
        "agentic_button",
        False,
        launch_agents
    )
    
with col2:
    create_feature_card(
        "🛠️ Prompt Engineering",
        "Guide to learning and building different Prompt Engineering Techniques",
        "Coming Soon",
        "prompt_engineering_button"
    )
    
with col3:
    create_feature_card(
        "🌟 Advance Agentic-RAG",
        "Guide to learning and building different Advance Agentic-RAG",
        "Coming Soon",
        "agentic_rag_button",
    )

with col4:
    create_feature_card(
        "🌐 Multimodal RAG",
        "Guide to learning and building Multimodal RAG",
        "Coming Soon",
        "ultimodal_rag_button"
    )

# Footer
st.markdown("---")
st.markdown("""
            
    <div align="center">
        <a target="_blank" href="https://www.youtube.com/@genieincodebottle"><img src="https://img.shields.io/badge/YouTube-@genieincodebottle-blue"></a>&nbsp;
        <a target="_blank" href="https://www.linkedin.com/in/rajesh-srivastava"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
        <a target="_blank" href="https://www.instagram.com/genieincodebottle/"><img src="https://img.shields.io/badge/@genieincodebottle-C13584?style=flat-square&labelColor=C13584&logo=instagram&logoColor=white&link=https://www.instagram.com/eduardopiresbr/"></a>
    </div>
   
""", unsafe_allow_html=True)