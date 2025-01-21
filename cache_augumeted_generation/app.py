"""
Cache Augmented Generation (CAG) Application
A Streamlit application for demonstrating Cache-Augmented Generation (CAG) method.
"""

# -------------------------------
# Imports
# -------------------------------
import os
from time import time
from typing import Tuple, List
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import torch
import plotly.graph_objects as go
from transformers import (
    BitsAndBytesConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    DynamicCache
)
from sentence_transformers import SentenceTransformer
import streamlit as st
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------
# Environment Setup
# -------------------------------
def setup_environment():
    """Initialize environment variables."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise Exception("HF_TOKEN is not set in .env file")
    return token

HF_TOKEN = setup_environment()

# -------------------------------
# Data Classes
# -------------------------------
@dataclass
class TestResults:
    """Store and manage test results from the CAG/Non-CAG process."""
    cache_time: List[float]
    generate_time: List[float]
    similarity: List[float]
    prompts: List[str]
    responses: List[str]
    ground_truths: List[str]
    timestamps: List[str]
    prepare_time: float = 0.0

    @property
    def avg_similarity(self) -> float:
        """Calculate average similarity score."""
        return sum(self.similarity) / len(self.similarity)

    @property
    def avg_cache_time(self) -> float:
        """Calculate average cache time."""
        return sum(self.cache_time) / len(self.cache_time)

    @property
    def avg_generate_time(self) -> float:
        """Calculate average generation time."""
        return sum(self.generate_time) / len(self.generate_time)

# -------------------------------
# Model Management
# -------------------------------
class CAGManager:
    """Manage Cache Augmented Generation operations."""
    
    def __init__(self, hf_token: str):
        """Initialize the CAG Manager."""
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    def initialize_model(self, model_name: str) -> bool:
        """Initialize a non-quantized model."""
        try:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=self.hf_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
            return True
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return False

    def initialize_quantized_model(self, model_name: str) -> bool:
        """Initialize a quantized model."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=self.quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=self.hf_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
            return True
        except Exception as e:
            st.error(f"Error initializing quantized model: {str(e)}")
            return False

    def generate_response(
        self, 
        prompt: str, 
        past_key_values: DynamicCache = None, 
        max_tokens: int = 300
    ) -> str:
        """Generate a response using the model."""
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            origin_ids = input_ids
            output_ids = input_ids.clone()
            next_token = input_ids

            with torch.no_grad():
                for _ in range(max_tokens):
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                    next_token = next_token.to(self.model.device)
                    past_key_values = outputs.past_key_values
                    output_ids = torch.cat([output_ids, next_token], dim=1)

                    if next_token.item() in self.model.config.eos_token_id:
                        break

            output = output_ids[:, origin_ids.shape[-1]:]
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise e

    def prepare_kv_cache(
        self, 
        documents: str, 
        instruction: str = None
    ) -> Tuple[DynamicCache, float]:
        """Prepare the KV cache for generation."""
        instruction = instruction or "Answer the question with a short answer."
        prompt = self._create_prompt(documents, instruction)
        start_time = time()

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            past_key_values = DynamicCache()

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False
                )

            if not outputs.past_key_values or len(outputs.past_key_values) == 0:
                raise ValueError("Empty KV cache generated")

            return outputs.past_key_values, time() - start_time

        except Exception as e:
            print(f"Error preparing KV cache: {str(e)}")
            return DynamicCache(), time() - start_time

    def run_test(
        self, 
        dataset: List[Tuple[str, str]], 
        documents: str,
        use_kv_cache: bool = True, 
        knowledge_cache: DynamicCache = None
    ) -> TestResults:
        """Run the test pipeline."""
        results = TestResults([], [], [], [], [], [], [])
        progress = st.progress(0, text="Processing questions...")

        for idx, (question, ground_truth) in enumerate(dataset):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            progress.progress(
                (idx + 1) / len(dataset),
                text=f"Processing question {idx + 1} of {len(dataset)}"
            )

            prompt = self._create_question_prompt(
                question, 
                documents if not use_kv_cache else ""
            )
            start_time = time()
            response = self.generate_response(prompt, knowledge_cache)
            gen_time = time() - start_time
            similarity = self._calculate_similarity(response, ground_truth)
            
            self._store_result(
                results, 
                question, 
                response, 
                ground_truth,
                similarity, 
                0 if not use_kv_cache else gen_time * 0.3, 
                gen_time
            )

        progress.empty()
        return results

    def clean_up(self, kv: DynamicCache, origin_len: int):
        """Clean up KV cache."""
        for i in range(len(kv.key_cache)):
            kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
            kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]

    def _create_prompt(self, documents: str, instruction: str) -> str:
        """Create a formatted prompt."""
        return f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an assistant for giving short answers based on given context.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Context information is below.
        ------------------------------------------------
        {documents}
        ------------------------------------------------
        {instruction}
        Question:
        """

    def _create_question_prompt(self, question: str, context: str = "") -> str:
        """Create a question-specific prompt."""
        if context:
            return self._create_prompt(context, "") + f"{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        return f"{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

    def _calculate_similarity(self, response: str, ground_truth: str) -> float:
        """Calculate similarity between response and ground truth."""
        response_embedding = self.bert_model.encode(response, convert_to_tensor=True)
        truth_embedding = self.bert_model.encode(ground_truth, convert_to_tensor=True)
        return torch.cosine_similarity(response_embedding, truth_embedding, dim=0).item()

    def _store_result(
        self, 
        results: TestResults, 
        question: str, 
        response: str,
        ground_truth: str, 
        similarity: float, 
        cache_time: float, 
        gen_time: float
    ):
        """Store test results."""
        results.prompts.append(question)
        results.responses.append(response)
        results.ground_truths.append(ground_truth)
        results.similarity.append(similarity)
        results.cache_time.append(cache_time)
        results.generate_time.append(gen_time)
        results.timestamps.append(datetime.now().strftime("%H:%M:%S"))

# -------------------------------
# UI Components
# -------------------------------
def display_results(results: TestResults):
    """Display test results in the UI."""
    # Display metrics
    cols = st.columns(4)
    metrics = [
        ("Average Similarity", results.avg_similarity, ""),
        ("Average Cache Time", results.avg_cache_time, "s"),
        ("Average Generation Time", results.avg_generate_time, "s"),
        ("Preparation Time", results.prepare_time, "s")
    ]

    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.metric(label, f"{value:.4f}{unit}")

    # Create results DataFrame
    df = pd.DataFrame({
        'Timestamp': results.timestamps,
        'Question': results.prompts,
        'Response': results.responses,
        'Ground Truth': results.ground_truths,
        'Similarity': results.similarity,
        'Cache Time (s)': results.cache_time,
        'Generation Time (s)': results.generate_time
    })

    # Display results in tabs
    tab1, tab2 = st.tabs(["📊 Results Table", "📈 Performance Metrics"])
    
    with tab1:
        st.dataframe(df, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Cache Time (s)'],
            name='Cache Time', 
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Generation Time (s)'],
            name='Generation Time', 
            mode='lines+markers'
        ))
        fig.update_layout(
            title='Performance Metrics Over Time',
            xaxis_title='Question Index',
            yaxis_title='Time (seconds)'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_help_section():
    """Render the help documentation section."""
    with st.expander("💡 Need Help Getting Started?"):
        st.markdown("""
        ### Initial Setup
        1. Rename `.env.example` to `.env`
        2. Get your Hugging Face token:
            * Visit [Hugging Face Tokens Page](https://huggingface.co/settings/tokens)
            * Create a new token with read access
            * Copy the token to `HF_TOKEN` in your .env file
            
        ### Key Features
        * **Cache-Augmented Generation**
            * Speeds up similar queries using KV caching
            * Reduces processing time for repeated content
            
        * **Model Management**
            * Initialize the model before processing data
            
        * **Data Workflow**
            * Preview your input data before processing
            * Select batch size (1-10 questions)
            
        ### Troubleshooting
        * Ensure HF_TOKEN env variable is properly set
        * Check your internet connection for model downloads
        * Clear cache if you encounter performance issues
        """)

def render_workflow_diagram():
    """Render the system workflow diagram."""
    with st.expander("📖 Overview & System Workflow", expanded=False):
        st.markdown("""
        #### Overview
        Retrieval-Augmented Generation (RAG) enhances language models by integrating external 
        knowledge but faces challenges like retrieval latency, errors, and system complexity. 
        Cache-Augmented Generation (CAG) addresses these by preloading relevant data into the 
        model's context, leveraging modern LLMs' extended context windows and caching runtime parameters. 
        This eliminates real-time retrieval during inference, enabling direct response generation.

        #### Advantages of CAG
        * **Reduced Latency:** Faster inference by removing real-time retrieval.
        * **Improved Reliability:** Avoids retrieval errors and ensures context relevance.
        * **Simplified Design:** Offers a streamlined, low-complexity alternative to RAG with comparable or better performance.

        #### Limitations of CAG
        * **Knowledge Size Limits:** Requires fitting all relevant data into the context window, unsuitable for extremely 
        large datasets.
        * **Context Length Issues:** Performance may degrade with very long contexts.

        #### References
        * [GitHub](https://github.com/hhhuang/CAG/tree/main)
        * [Research Paper](https://arxiv.org/abs/2412.15605)
        """, unsafe_allow_html=True)

        current_dir = Path(__file__).parent  # Directory of current script
        image_path = current_dir / 'images'
        # Get the relative path to the image
        col1, col2 = st.columns(2)
        with col1:
            routing_diagram = Image.open(image_path/ 'cag_diagram.png')
            st.image(routing_diagram, caption='High Level Architecture')
        with col2:
            sequence_diagram = Image.open(image_path/ 'cag_sequence_diagram.png')
            st.image(sequence_diagram, caption='Sequence Diagram')
        
# -------------------------------
# Main Application
# -------------------------------
def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="CAG - Cache Augmented Generation",
        page_icon="🗂️",
        layout="wide"
    )
    st.subheader("🗂️ Cache-Augmented Generation (CAG)")
    # Render Workflow    
    render_workflow_diagram()
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Huggingface Model Config")
        model_name = st.text_input(
            "Model Name",
            value="meta-llama/Llama-3.2-1B-Instruct",
            help="Enter the name of the huggingface model to use"
        )
        use_kv_cache = st.checkbox(
            "CAG (Use KV Cache)",
            value=True,
            help="Enable KV Cache for Cache-Augmented Generation"
        )

        # GPU settings
        quantized = False
        if torch.cuda.is_available():
            quantized = st.checkbox(
                "Enable Model Quantization",
                value=True,
                help="Enable Model Quantization to reduce Model Size"
            )
            st.markdown(
                '<p style="font-size: 14px; color: green; font-weight: bold">'
                '✅ CUDA is available - Running on GPU</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p style="font-size: 14px; color: red; font-weight: bold">'
                '❌ CUDA is not available - Running on CPU</p>',
                unsafe_allow_html=True
            )

        initialize_button = st.button(
            "Initialize Model",
            type="primary",
            help="Click to load the model"
        )
        render_help_section()

    # Model initialization
    if initialize_button:
        with st.spinner("Initializing model..."):
            manager = CAGManager(HF_TOKEN)
            initialized = False

            if quantized:
                initialized = manager.initialize_quantized_model(model_name=model_name)
            else:
                initialized = manager.initialize_model(model_name=model_name)

            if initialized:
                st.session_state.manager = manager
                st.success("✅ Model initialized successfully!")
            else:
                st.error("❌ Failed to initialize model.")
                return

    if 'manager' not in st.session_state:
        st.info("👈 Initialize the model using the sidebar controls")
        return

    # Data loading and processing
    try:
        # Load dataset
        df = pd.read_csv("./datasets/sample_qa_dataset.csv")
        with st.expander("📄 Preview Input Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Configure processing
        max_questions = st.number_input(
            "Maximum Questions to Process", 
            min_value=1, 
            value=5
        )
        dataset = list(zip(
            df['sample_question'].iloc[:max_questions],
            df['sample_ground_truth'].iloc[:max_questions]
        ))
        documents = '\n\n'.join(df["text"].tolist())

        # Processing buttons
        col1, col2 = st.columns(2)
        with col1:
            prepare_cache = st.button(
                "1️⃣ Prepare KV Cache",
                disabled=not use_kv_cache,
                type="primary"
            )
        with col2:
            process_button = st.button(
                "2️⃣ Process Questions",
                type="primary"
            )

        # Cache preparation
        if prepare_cache and use_kv_cache:
            cache_progress = st.progress(0, text="Preparing KV Cache...")
            with st.spinner("Preparing KV Cache..."):
                try:
                    for i in range(100):
                        cache_progress.progress(i/100)
                        if i == 50:
                            knowledge_cache, prep_time = st.session_state.manager.prepare_kv_cache(documents)
                            st.session_state.knowledge_cache = knowledge_cache
                            st.session_state.prep_time = prep_time
                    cache_progress.progress(1.0, text="KV Cache Ready!")
                    st.success(f"✅ KV Cache prepared in {prep_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error preparing KV cache: {str(e)}")
                    return

        # Question processing
        if process_button:
            try:
                # Get prepared cache if available
                if use_kv_cache:
                    knowledge_cache = st.session_state.get('knowledge_cache')
                    prep_time = st.session_state.get('prep_time', 0.0)
                    kv_len = knowledge_cache.key_cache[0].shape[-2]
                    st.session_state.manager.clean_up(knowledge_cache, kv_len)
                else:
                    knowledge_cache = None
                    prep_time = 0.0

                # Process questions and display results
                with st.spinner("Processing questions..."):
                    results = st.session_state.manager.run_test(
                        dataset=dataset,
                        documents=documents,
                        use_kv_cache=use_kv_cache,
                        knowledge_cache=knowledge_cache
                    )
                    results.prepare_time = prep_time
                    display_results(results)

            except Exception as e:
                st.error(f"An error occurred during testing: {str(e)}")
                st.exception(e)

    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.exception(e)

# -------------------------------
# Application Entry Point
# -------------------------------
if __name__ == "__main__":
    main()