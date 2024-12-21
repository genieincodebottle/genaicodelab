import streamlit as st
import os
from .rag_descriptions import RAGDescriptions, RAGDescription
from typing import List, Dict, Optional

def check_image_exists(image_path: str) -> bool:
    """Check if image file exists at the given path."""
    return os.path.exists(image_path) if image_path else False

def display_resource_links(links: Optional[Dict[str, str]] = None):
    """Display resource links in a formatted way."""
    if links:
        st.markdown("**📚 Resources:**")
        for title, url in links.items():
            st.markdown(f"- [{title}]({url})")

def display_rag_info_tooltip(rag_description: RAGDescription, rag_type: str):
    """Display RAG information with tooltip using Streamlit."""
    st.markdown(f"**{rag_type}** ")
    with st.expander("📖 Learn More", expanded=False):
        # Description with optional diagram
        has_image = check_image_exists(rag_description.diagram_path)
        
        if has_image:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(rag_description.description)
            with col2:
                try:
                    st.image(rag_description.diagram_path, 
                            caption=f"{rag_type} Diagram",
                            use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        else:
            st.write(rag_description.description)
        
        # Features and Use Cases
        col_features, col_uses = st.columns(2)
        with col_features:
            st.markdown("**🎯 Key Features:**")
            for feature in rag_description.key_features:
                st.markdown(f"- {feature}")
        
        with col_uses:
            st.markdown("**💡 Use Cases:**")
            for use_case in rag_description.use_cases:
                st.markdown(f"- {use_case}")
        
        # Display resource links if available
        if hasattr(rag_description, 'resources'):
            st.markdown("---")
            display_resource_links(rag_description.resources)

def display_selected_rag_info(selected_rag_type: str):
    """Display information for the selected RAG type in the main page."""
    rag_desc = RAGDescriptions.RAG_TYPES.get(selected_rag_type)
    if rag_desc:
        display_rag_info_tooltip(rag_desc, selected_rag_type)

def init_rag_info_styles():
    """Initialize custom styles for RAG information display."""
    st.markdown("""
        <style>
        /* Fix for title visibility */
        .main > div:first-child {
            padding-top: 3.5rem !important;
        }
        
        /* RAG info styles */
        .stExpander {
            border: none;
            box-shadow: none;
            margin-bottom: 0.5rem;
            max-width: 100%;
            width: 100%;
        }
        .stExpander > div:first-child {
            border-radius: 4px;
            background-color: #f8f9fa;
        }
        .stMarkdown p {
            margin-bottom: 0.2rem;
        }
        div[data-testid="stExpander"] {
            width: 100%;
        }
        
        /* Title specific adjustments */
        .main .block-container {
            padding-top: 1rem;
            margin-top: 2rem;
        }
        
        /* Link styles */
        .resource-link {
            color: #0366d6;
            text-decoration: none;
        }
        .resource-link:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

def main_title():
    """Add additional spacing before title."""
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)