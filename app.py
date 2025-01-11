import os
import subprocess
import streamlit as st

st.set_page_config(
    page_title="GenAI Code Lab",
    page_icon="🧪",
    layout="wide"
)

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
st.image("static/images/logo.png", width=200)
st.markdown("""
        Welcome to GenAI Code Lab – let's explore and create together :)
    """)

with st.expander("GenAI Evolution"):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        # The Evolution of Generative AI  

        **What is Generative AI?**  
        Generative AI refers to artificial intelligence systems that create new content, such as text, images, audio, 
                    and video, by learning patterns and structures from existing data. It uses models based on 
                    Neural Networks, such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), 
                    and transformer based architectures (e.g. GPT, Claude, Gemini, Llama, DeepSeek etc) to generate 
                    outputs that resemble human made content. These models leverage unsupervised or semi-supervised 
                    learning techniques to produce creative, realistic results across various domains. 
        ## 1950s-1960s: Early Foundations  
        - **1958**: Perceptron, the first neural network, introduced by Frank Rosenblatt.  
        - **1960s**: Basic pattern recognition systems and rule-based approaches emerged, focusing on simple tasks like character recognition and logical reasoning.  
        ## 1970s-1990s: Knowledge Engineering  
        - **1970s**: Expert systems like MYCIN were developed for domain-specific reasoning, marking the rise of symbolic AI.  
        - **1980s**: Symbolic AI and logic programming (e.g., Prolog) gained prominence, focusing on knowledge representation.  
        - **1990s**: Statistical NLP techniques matured, including Hidden Markov Models and early parsing algorithms.  
        ## 2000s-2010s: Statistical Learning and Deep Learning  
        - **Early 2000s**: Statistical machine learning methods like SVMs and decision trees became mainstream.  
        - **2006**: Geoffrey Hinton popularized "deep learning" with deep belief networks.  
        - **2012**: AlexNet’s victory in the ImageNet competition showcased the power of deep neural networks.  
        - **2013**: Variational Autoencoders (VAEs) introduced by Kingma and Welling enabled probabilistic generative modeling.  
        - **2014**: Ian Goodfellow introduced Generative Adversarial Networks (GANs), enabling high-quality image synthesis.  
        ## 2014-2018: Key Developments Leading to the Transformer Era  
        - **2017**: The Transformer architecture paper *"Attention is All You Need"* introduced by Vaswani et al. revolutionized sequence-to-sequence tasks, laying the groundwork for models like BERT and GPT.  
        - **2018**: BERT introduced transformer-based contextual embeddings, enabling state-of-the-art results in NLP tasks.  
        ## 2019-2021: Transformer Revolution  
        - **2019**: GPT-2 demonstrated advanced text generation with 1.5 billion parameters.  
        - **2020**: GPT-3’s 175 billion parameters set new benchmarks for large-scale language modeling.  
        - **2021**: OpenAI released DALL-E and CLIP, integrating multimodal capabilities across text and images.  
        ## 2022-Present: Advanced Generative AI and RAG  
        - **2022**: ChatGPT, based on GPT-3.5, popularized conversational AI by making it widely accessible. 
                    Prompt Engineering emerged as a critical technique for optimizing outputs from large language models (LLMs).  
        - **2023**: Advanced Prompt Engineering and Retrieval-Augmented Generation (RAG) gained traction, combining LLMs with 
                    external knowledge bases for enhanced factual correctness. Also, Multimodal RAG began to emerge, enabling 
                    models to retrieve and process knowledge across text, images, and other formats, though its adoption is still growing.  
        - **2024**: AI agents capable of performing autonomous, multi-step tasks across domains gain increasing attention, 
                    moving toward more generalized and interactive applications. Cache-Augmented RAG, while speculative, is seen as 
                    a promising approach to improve efficiency by reusing frequently accessed knowledge.    
        ## Present & Future (2025 and Beyond)  
        - Advanced multi-agent systems are being developed for collaborative AI tasks.  
        - Autonomous Agentic AI systems with enhanced reasoning and decision-making capabilities continue to evolve, 
                    addressing challenges like alignment and safety.  
        - Ongoing refinements in RAG frameworks and multimodal systems aim to deepen integration with real world knowledge.  
        """)
    
    with col2:
        st.image("static/images/genai_tech_stack.png", 
                caption="Evolution of Generative AI",
                use_container_width=True)
        st.markdown("""
            <div align="left">
                <a target="_blank" href="https://github.com/genieincodebottle/generative-ai/blob/main/docs/essential-terms-genai.pdf">📖 GenAI Glossary of Terms</a>&nbsp;
                <br/>
                <a target="_blank" href="https://github.com/genieincodebottle/generative-ai/blob/main/docs/GenAI_Interview_Questions-Draft.pdf">💬 GenAI Interview Q & A</a>
                <br/>
                <a target="_blank" href="https://github.com/genieincodebottle/generative-ai/blob/main/docs/genai-with-aws-cloud.pdf">☁️ GenAI with AWS Cloud</a>
                <br/>
                <a target="_blank" href="https://github.com/genieincodebottle/generative-ai/blob/main/docs/genai-with-azure-cloud.pdf">☁️ GenAI with Azure Cloud</a>
                <br/>
                <a target="_blank" href="https://github.com/genieincodebottle/generative-ai/blob/main/docs/genai-with-vertexai.pdf">☁️ GenAI with VertexAI Cloud</a>
            </div>
        """, unsafe_allow_html=True)
        
    
# Main Features Section
st.markdown("### ✨ Resources")

def create_feature_card(title, description, button_label, button_key, is_disabled=True, on_click=None, learn_more_content=None, link=None):
    st.markdown(f"""
        <div style='height: 50px; font-weight: bold; overflow: auto; text-align: left;'>    
            {title}
        </div>
        """, unsafe_allow_html=True)
    container = st.container()
    with container:
        st.markdown(f"""
            <div style='height: 70px; overflow: auto; text-align: left;'>
                {description}
            </div>
        """, unsafe_allow_html=True)
        
        if learn_more_content:
            with st.expander("What is this?"):
                st.markdown(learn_more_content, unsafe_allow_html=True)
                st.markdown(link, unsafe_allow_html=True)
        
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

col1, spacer1, col2, spacer2, col3, spacer3, col4 = st.columns([2, 0.1, 2, 0.1, 2, 0.1, 2])

with col1:
    create_feature_card(
        "🤖 Noob to Autonomous Agents",
        "Guide to learning and building different Agentic Patterns using LangGrapg, CrewAI etc",
        "Launch",
        "agentic_button",
        False,
        launch_agents,
        """AI Agents are the next evolution of Large Language Models (LLMs). Instead of just processing text, they can:
        Actively solve complex problems, Make strategic decisions, Use tools and APIs, Learn and adapt from interactions, 
        Work autonomously. Think of an AI Agent as an LLM with hands and a brain, it can understand, plan and interact 
        with the real world through tools and APIs.
        """,
        """https://github.com/genieincodebottle/genaicodelab/blob/main/agents/agentic_patterns/README.md"""
    )
    
with col2:
    create_feature_card(
        "🌟 Advance Agentic-RAG",
        "Guide to learning and building different Advance Agentic-RAG",
        "🌟 Stay Tuned",
        "agentic_rag_button",
        True,
        None,
        """Agentic RAG enhances traditional RAG by integrating intelligent agents for dynamic decision-making, 
        real-time adaptability, and context-aware responses. It ensures accurate data retrieval, reduces hallucinations, 
        and handles complex queries efficiently with customizable workflows."""
    )

with col3:
    create_feature_card(
        "🌐 Multimodal RAG",
        "Guide to learning and building Multimodal RAG",
        "🌟 Stay Tuned",
        "multimodal_rag_button",
        True,
        None,
        """Multimodal Retrieval Augmented Generation (RAG) enhances AI models by enabling them to retrieve and process 
        information from various data types, such as text, images, audio, and video resulting in more accurate and 
        context rich responses. This approach is particularly beneficial for applications requiring the analysis of 
        diverse data sources"""
    )

with col4:
    create_feature_card(
        "⚡ Cache-Augumented Generation",
        "Guide to learning and building different Cache-Augumented Generation Techniques",
        "🌟 Stay Tuned",
        "cag_button",
        True,
        None,
        """Cache-Augmented Generation (CAG) is an approach that enhances large language models by preloading relevant 
        documents into the model's extended context window, eliminating the need for 
        real-time retrieval during inference. This method reduces latency and simplifies the architecture compared 
        to Retrieval-Augmented Generation (RAG), making it particularly effective for tasks involving static datasets 
        and low-latency requirements."""
    )

col1, spacer1, col2, spacer2, col3, spacer3, col4= st.columns([2, 0.1, 2, 0.1, 2, 0.1, 2])

with col1:
    create_feature_card(
        "🛠️ Advanced Prompt Engineering",
        "Guide to learning and building different Prompt Engineering Techniques",
        "🌟 Stay Tuned",
        "prompt_engineering_button"
    )
    
with col2:
    create_feature_card(
        "🌟 Prompt Guard, Prompt Caching etc",
        "Guide to different important topics",
        "🌟 Stay Tuned",
        "misc_button",
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