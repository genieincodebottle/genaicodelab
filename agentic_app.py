import streamlit as st
from typing import List, Dict
import time
import plotly.graph_objects as go
from utils.document_processors import DocumentProcessor
from agentic_rag.basic_rag import BasicRAG
from agentic_rag.adaptive_rag import AdaptiveRAG
from agentic_rag.corrective_rag import CorrectiveRAG
from agentic_rag.reranking_rag import ReRankingRAG
from agentic_rag.hybrid_search_rag import HybridSearchRAG
from agentic_rag.multi_index_rag import MultiIndexRAG
from agentic_rag.query_expansion_rag import QueryTransformRAG
from agentic_rag.self_adaptive_rag import SelfAdaptiveRAG
from agentic_rag.hyde_rag import HyDERAG
from utils.rag_evaluation import evaluate_single_query
from dataclasses import dataclass
from utils.rag_info import display_selected_rag_info, init_rag_info_styles, main_title

# Constants
@dataclass(frozen=True)
class ModelConfig:
    PROVIDERS = ["gemini", "openai", "claude", "ollama", "groq"]
    MODELS = {
        "gemini": ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
        "openai": ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"],
        "claude": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "ollama": ["llama3.2:1b", "llama3.1:latest", "mistral", "phi3.5", "phi3"],
        "groq": ["llama3-8b-8192", "llama3-70b-8192", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"]
    }

def get_contexts_from_rag_system(query: str) -> List[str]:
    """Helper function to get contexts from either RAG system type."""
    if st.session_state.rag_type == "Basic RAG":
        return st.session_state.rag_system.basic_rag_tools._retrieve_documents(query)
    elif st.session_state.rag_type == "Adaptive RAG":
        return st.session_state.rag_system.rag_tools._retrieve_documents(query)
    elif st.session_state.rag_type == "Corrective RAG":
        return st.session_state.rag_system.correction_tools._retrieve_documents(query)
    elif st.session_state.rag_type == "Reranking RAG":
        return st.session_state.rag_system.reranking_tools._retrieve_documents(query)
    elif st.session_state.rag_type == "Hybrid Search RAG":
        return st.session_state.rag_system._retrieve_documents(query)
    elif st.session_state.rag_type == "Multi Index RAG":
        return st.session_state.rag_system.multi_tools._retrieve_documents(query)
    elif st.session_state.rag_type == "Query Expansion RAG":
        return st.session_state.rag_system.transform_tools._retrieve_documents(query)
    elif st.session_state.rag_type == "Self Adaptive RAG":
        return st.session_state.rag_system.adaptive_rag_tools._retrieve_documents(query)
    elif st.session_state.rag_type == "HyDE RAG":
        return st.session_state.rag_system.hyde_tools._retrieve_documents(
            query=query,
            method=st.session_state.get('retrieval_method', 'hyde'),
            hypothetical_doc=st.session_state.get('hypothetical_doc'),
            k_docs=st.session_state.get('k_docs', 3),
            similarity_threshold=st.session_state.get('similarity_threshold', 0.7),
            hybrid_alpha=st.session_state.get('hybrid_alpha', 0.5)
        )
  
model_config = ModelConfig()
RAG_TYPES = ["Basic RAG", "Adaptive RAG", "Corrective RAG", "Reranking RAG", "Hybrid Search RAG", "Multi Index RAG", "Query Expansion RAG", "Self Adaptive RAG", "HyDE RAG"]

# Initialize session state
def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'model_provider' not in st.session_state:
        st.session_state.model_provider = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'rag_type' not in st.session_state:
        st.session_state.rag_type = "Basic RAG"
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "upload"
    if 'api_initialized' not in st.session_state:
        st.session_state.api_initialized = False
        
init_session_state()

# After initializing session state
init_rag_info_styles()

# Helper Functions
@st.cache_data
def plot_query_metrics(scores: Dict[str, float]) -> go.Figure:
    """Create a cached bar chart for query metrics."""
    fig = go.Figure([go.Bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        text=[f"{v:.3f}" for v in scores.values()],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Query Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        xaxis={'tickangle': 45}
    )
    
    return fig

def initialize_rag_system() -> bool:
    """Initialize the RAG system based on selected configuration."""
    try:
        if st.session_state.rag_type == "Basic RAG":
            st.session_state.rag_system = BasicRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        if st.session_state.rag_type == "Adaptive RAG":
            st.session_state.rag_system = AdaptiveRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        elif st.session_state.rag_type == "Corrective RAG":
            st.session_state.rag_system = CorrectiveRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        elif st.session_state.rag_type == "Reranking RAG":
            st.session_state.rag_system = ReRankingRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        elif st.session_state.rag_type == "Hybrid Search RAG":
            st.session_state.rag_system = HybridSearchRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        elif st.session_state.rag_type == "Multi Index RAG":
            st.session_state.rag_system = MultiIndexRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        elif st.session_state.rag_type == "Query Expansion RAG":
            st.session_state.rag_system = QueryTransformRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        elif st.session_state.rag_type == "Self Adaptive RAG":
            st.session_state.rag_system = SelfAdaptiveRAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )
        elif st.session_state.rag_type == "HyDE RAG":
            st.session_state.rag_system = HyDERAG(
                st.session_state.model_provider,
                st.session_state.model_name
            )

        # If documents exist, reload them
        if st.session_state.documents:
            st.session_state.rag_system.load_documents(st.session_state.documents)
            
        st.session_state.initialized = True
        st.session_state.api_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        st.session_state.initialized = False
        st.session_state.api_initialized = False
        return False

def show_temp_success_message():
    """Show a temporary success message in the sidebar."""
    # Create a placeholder in the sidebar
    placeholder = st.sidebar.empty()
    
    # Show the success message
    placeholder.success("Configuration updated successfully!")
    
    # Wait for 3 seconds
    time.sleep(4)
    
    # Clear the message
    placeholder.empty()

def process_documents(docs: List[str]) -> bool:
    """Process documents with error handling."""
    try:
        if not docs:
            st.warning("No documents provided")
            return False
        st.session_state.rag_system.load_documents(docs)
        st.session_state.documents = docs
        return True
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def display_document_info(docs: List[str]) -> None:
    """Display information about processed documents."""
    with st.expander("View Processed Documents"):
        for i, doc in enumerate(docs, 1):
            st.text_area(
                f"Document Chunk {i}",
                value=doc[:500] + "..." if len(doc) > 500 else doc,
                height=100,
                disabled=True
            )

main_title()
# Application Title
st.title("Advanced Agentic-RAG System")

display_selected_rag_info(st.session_state.get('rag_type', "Basic RAG"))

# Sidebar configuration
with st.sidebar:
    
    st.header("Configuration")
    
    # RAG Type Selection with callback
    def on_rag_type_change():
        st.session_state.rag_type = st.session_state.rag_type_selector
        success = initialize_rag_system()
        if success:
            show_temp_success_message()
        return success

    rag_type = st.selectbox(
        "Select RAG Type",
        options=RAG_TYPES,
        index=RAG_TYPES.index(st.session_state.get('rag_type', "Basic RAG")),
        on_change=on_rag_type_change,
        key="rag_type_selector"
    )
    
    # Model Selection with callback
    def on_model_change():
        success = initialize_rag_system()
        if success:
            show_temp_success_message()
        return success

    model_provider = st.selectbox(
        "Select Model Provider",
        options=model_config.PROVIDERS,
        #on_change=on_model_change,
        key="model_provider_selector"
    )
    
    model_name = st.selectbox(
        "Select Model",
        options=model_config.MODELS[model_provider],
        on_change=on_model_change,
        key="model_name_selector"
    )

    reuse_same_model_for_eval = st.checkbox(
        f"Use same {model_provider.title()} API key for evaluation",
        value=False
    )
    # Update session state
    st.session_state.model_provider = model_provider
    st.session_state.model_name = model_name
    
    if st.button("Initialize App"):
        if initialize_rag_system():
            st.success("Configuration set successfully!")
        else:
            st.warning("Please check API keys in .env file" if model_provider != "ollama" 
                      else "Please make sure Ollama is installed and running locally")

        # Navigation
    st.markdown("---")
    st.header("Navigation")
    cols = st.columns(2)
    with cols[0]:
        if st.button("📄 Document Upload",
                    type="primary" if st.session_state.current_page == "upload" else "secondary"):
            st.session_state.current_page = "upload"
    with cols[1]:
        if st.button("❓ Query Documents",
                    type="primary" if st.session_state.current_page == "query" else "secondary"):
            st.session_state.current_page = "query"

# Main Content
if not st.session_state.api_initialized:
    st.warning("Please Initialize App in the sidebar to begin")
else:
    # Document Upload Page
    if st.session_state.current_page == "upload":
        #st.markdown("<h2 class='stHeader'>📄 Document Upload and Processing</h2>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=['pdf', 'csv', 'txt'],
            accept_multiple_files=True
        )
            
        if uploaded_files and st.button("Process Files"):
            all_documents = []
            progress_bar = st.progress(0)
                
            for i, file in enumerate(uploaded_files):
                st.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                    
                try:
                    docs = DocumentProcessor.get_documents_from_file(file)
                    all_documents.extend(docs)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
            if all_documents and process_documents(all_documents):
                st.success("All files processed successfully!")
                st.session_state.current_page = "query"
        
        
        # Display loaded documents
        if st.session_state.documents:
            with st.expander("View Loaded Documents"):
                for i, doc in enumerate(st.session_state.documents, 1):
                    st.text(f"{i}. {doc[:200]}..." if len(doc) > 200 else f"{i}. {doc}")
    
    # Query Page
    elif st.session_state.current_page == "query":
        #st.markdown("<h2 class='stHeader'>❓ Query Documents</h2>", unsafe_allow_html=True)
        
        if not st.session_state.documents:
            st.warning("No documents loaded. Please upload documents first.")
            if st.button("Go to Document Upload"):
                st.session_state.current_page = "upload"
        else:
            with st.expander("View Loaded Documents"):
                for i, doc in enumerate(st.session_state.documents, 1):
                    st.text(f"{i}. {doc[:200]}..." if len(doc) > 200 else f"{i}. {doc}")
            
            query = st.text_input("Enter your question")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                max_iterations = st.slider("Maximum iterations", 1, 5, 3)
            with col2:
                auto_evaluate = st.checkbox("Enable Evaluation", value=True)
            with col3:
                use_ground_truth = st.checkbox("Add Ground Truth", value=False)
            
            ground_truth = st.text_area("Ground Truth Answer", height=100) if use_ground_truth else None
            
            if st.session_state.rag_type == "HyDE RAG":
                with st.expander("HyDE RAG Settings"):
                    # Document Generation Settings
                    st.subheader("Document Generation")
                    doc_style = st.selectbox(
                        "Hypothetical Document Style",
                        options=["scientific", "technical", "narrative"],
                        help="Style of the generated hypothetical document"
                    )
                    
                    # Retrieval Settings
                    st.subheader("Retrieval Settings")
                    retrieval_cols = st.columns(2)
                    
                    with retrieval_cols[0]:
                        retrieval_method = st.selectbox(
                            "Retrieval Method",
                            options=["hyde", "hybrid", "standard"],
                            help="Method to use for document retrieval"
                        )
                        
                        k_docs = st.slider(
                            "Number of Documents",
                            min_value=1,
                            max_value=10,
                            value=3,
                            help="Number of documents to retrieve"
                        )
                    
                    with retrieval_cols[1]:
                        similarity_threshold = st.slider(
                            "Similarity Threshold",
                            min_value=0.1,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            help="Maximum distance threshold for retrieved documents (lower = more similar)"
                        )
                        
                        if retrieval_method == "hybrid":
                            hybrid_alpha = st.slider(
                                "Hybrid Alpha",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.5,
                                step=0.1,
                                help="Weight between HyDE and standard retrieval (0=HyDE only, 1=standard only)"
                            )
                        else:
                            hybrid_alpha = 0.5
            
            
            if st.button("Get Answer", type="primary") and query:
                with st.spinner("Processing query..."):
                    try:
                        # Get RAG response
                        if st.session_state.rag_type == "HyDE RAG":
                            result = st.session_state.rag_system.run(
                                query=query,
                                max_iterations=max_iterations,
                                doc_style=doc_style,
                                k_docs=k_docs,
                                similarity_threshold=similarity_threshold,
                                hybrid_alpha=hybrid_alpha,
                                retrieval_method=retrieval_method
                            )
                        else:
                            result = st.session_state.rag_system.run(
                                query,
                                max_iterations=max_iterations
                            )
                        
                        # Create tabs for results
                        result_tabs = st.tabs(["Answer", "Evaluation", "Query Details"])
                        
                        # Answer Tab
                        with result_tabs[0]:
                            st.markdown("### Final Answer")
                            st.success(result['final_answer'])
                            if use_ground_truth and ground_truth:
                                st.markdown("### Ground Truth")
                                st.info(ground_truth)
                        
                        # Evaluation Tab
                        with result_tabs[1]:
                            if auto_evaluate:
                                try:
                                    contexts = get_contexts_from_rag_system(query)
                                    
                                    eval_results = evaluate_single_query(
                                        model_provider=model_provider,
                                        model_name=model_name,
                                        question=query,
                                        answer=result['final_answer'],
                                        contexts=contexts,
                                        resuse_same_model_for_eval=reuse_same_model_for_eval,
                                        ground_truth=ground_truth if use_ground_truth else None
                                    )
                                    
                                    # Display metrics
                                    st.subheader("Response Quality Metrics")
                                    metric_cols = st.columns(2)
                                    
                                    with metric_cols[0]:
                                        st.metric("Answer Relevancy", f"{eval_results['answer_relevancy']:.3f}")
                                    with metric_cols[1]:
                                        st.metric("Faithfulness", f"{eval_results['faithfulness']:.3f}")
                                    
                                    if use_ground_truth and ground_truth:
                                        st.metric("Context Precision", f"{eval_results.get('context_precision', 0.0):.3f}")
                                    
                                    # Plot metrics
                                    st.plotly_chart(plot_query_metrics(eval_results), use_container_width=True)
                                    
                                    with st.expander("📊 Understanding the Metrics"):
                                        st.markdown("""
                                        - **Answer Relevancy**: Measures how well the answer addresses the question
                                        - **Faithfulness**: Measures how factually accurate the answer is based on the context
                                        - **Context Precision**: Measures the accuracy of answer against ground truth (when provided)
                                        """)
                                except Exception as e:
                                    st.error(f"Error during evaluation: {str(e)}")
                            else:
                                st.info("Evaluation is disabled. Enable it using the checkbox above.")
                        
                        # Query Details Tab
                        with result_tabs[2]:
                            st.markdown("#### Query Processing Details")
                            st.info(f"Original Query: {query}")
                            
                            if st.session_state.rag_type == "Basic RAG":
                                if 'current_query' in result:
                                    st.info("Current Query: " + str(result['current_query']))
                                if 'retrieved_docs' in result:
                                    st.subheader("Retrieved Documents")
                                    for i, doc in enumerate(result['retrieved_docs'], 1):
                                        with st.expander(f"Retrieved Document {i}"):
                                            st.text(doc)
                            elif st.session_state.rag_type == "Adaptive RAG":
                                st.info(f"Final Query: {result.get('final_query', query)}")
                            elif st.session_state.rag_type == "Corrective RAG":
                                if 'critique' in result:
                                    st.info("Critique: " + result['critique'])
                            elif st.session_state.rag_type == "Reranking RAG":
                                if 'initial_docs' in result:
                                    for i, doc in enumerate(result['initial_docs'], 1):
                                        with st.expander(f"Initial Document {i}"):
                                            st.text(doc)
                                if 'reranked_docs' in result:
                                    for i, doc in enumerate(result['reranked_docs'], 1):
                                        with st.expander(f"Reranked Document {i}"):
                                            st.text(doc)
                            elif st.session_state.rag_type == "Hybrid Search RAG":
                                if 'bm25_docs' in result:
                                    st.subheader("BM25 Documents")
                                    for i, doc in enumerate(result['bm25_docs'], 1):
                                        with st.expander(f"BM25 Document {i}"):
                                            st.text(doc)
                                if 'vector_docs' in result:
                                    st.subheader("Vector Search Documents")
                                    for i, doc in enumerate(result['vector_docs'], 1):
                                        with st.expander(f"Vector Document {i}"):
                                            st.text(doc)
                                if 'ensemble_docs' in result:
                                    st.subheader("Ensemble Documents")
                                    for i, doc in enumerate(result['ensemble_docs'], 1):
                                        with st.expander(f"Ensemble Document {i}"):
                                            st.text(doc)
                                if 'hybrid_explanation' in result:
                                    st.info("Hybrid Search Explanation: " + result['hybrid_explanation'])
                            elif st.session_state.rag_type == "Multi Index RAG":
                                if 'source_docs' in result:
                                    st.subheader("Source Documents")
                                    for i, doc in enumerate(result['source_docs'], 1):
                                        with st.expander(f"Source Document {i}"):
                                            st.text(doc)
                                if 'source_explanation' in result:
                                    st.info("Source Explanation: " + result['source_explanation'])
                            elif st.session_state.rag_type == "Query Expansion RAG":
                                if 'original_query' in result:
                                    st.info("Original Query: " + result['original_query'])
                                if 'transformed_queries' in result:
                                    st.info("Transformed Queries: " + result['transformed_queries'])
                                if 'retrieved_docs' in result:
                                    st.info("Retrieved Docs: " + result['retrieved_docs'])
                            elif st.session_state.rag_type == "Self Adaptive RAG":
                                if 'query_complexity' in result:
                                    st.info("Query Complexity: " + str(result['query_complexity']))
                                if 'query_type' in result:
                                    st.info("Query Type: " + result['query_type'])
                                if 'retrieval_strategy' in result:
                                    st.info("Retrieval Strategies: " + result['retrieval_strategy'])   
                            elif st.session_state.rag_type == "HyDE RAG":
                                if 'hypothetical_doc' in result:
                                    st.info("Hypothetical Doc: " + str(result['hypothetical_doc']))
                                if 'retrieval_method' in result:
                                    st.info("Retrieval Methods: " + result['retrieval_method'])
                            

                            st.info(f"Number of Iterations: {result['iterations']}")
                            
                            # Display context based on RAG type
                            if st.session_state.rag_type == "Basic RAG":
                                 contexts = st.session_state.rag_system.basic_rag_tools._retrieve_documents(query)
                            elif st.session_state.rag_type == "Adaptive RAG":
                                contexts = st.session_state.rag_system.rag_tools._retrieve_documents(query)
                            elif st.session_state.rag_type == "Corrective RAG":
                                contexts = st.session_state.rag_system.correction_tools._retrieve_documents(query)
                            elif st.session_state.rag_type == "Reranking RAG":
                                contexts = st.session_state.rag_system.reranking_tools._retrieve_documents(query)
                            elif st.session_state.rag_type == "Hybrid Search RAG":
                                contexts = st.session_state.rag_system._retrieve_documents(query)
                            elif st.session_state.rag_type == "Multi Index RAG":
                                contexts = st.session_state.rag_system.multi_tools._retrieve_documents(query)
                            elif st.session_state.rag_type == "Query Expansion RAG":
                                 contexts = st.session_state.rag_system.transform_tools._retrieve_documents(query)
                            elif st.session_state.rag_type == "Self Adaptive RAG":
                                 contexts = st.session_state.rag_system.adaptive_rag_tools._retrieve_documents(query)
                            elif st.session_state.rag_type == "HyDE RAG":
                                 contexts = st.session_state.rag_system.hyde_tools._retrieve_documents(
                                        query,
                                        method=st.session_state.get('retrieval_method', 'hyde'),
                                        hypothetical_doc=st.session_state.get('hypothetical_doc'),
                                        k_docs=st.session_state.get('k_docs', 3),
                                        similarity_threshold=st.session_state.get('similarity_threshold', 0.7),
                                        hybrid_alpha=st.session_state.get('hybrid_alpha', 0.5)
                                    )
                            

                            if contexts:
                                with st.expander("View Retrieved Context"):
                                    for i, doc in enumerate(contexts, 1):
                                        st.markdown(f"**Context {i}:**")
                                        st.text(doc)
                                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Advanced RAG System - Combining Adaptive and Corrective Approaches</p>
    </div>
    """,
    unsafe_allow_html=True
)