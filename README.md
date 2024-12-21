# Advanced Agentic-RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that implements multiple advanced RAG architectures and strategies. This system combines various RAG approaches to provide enhanced document understanding and question-answering capabilities.

## 🌟 Features

The system implements nine different RAG architectures:

1. **Basic RAG**: Standard implementation of retrieval-augmented generation
   - Simple document retrieval and answer generation
   - Baseline for comparing other implementations

2. **Adaptive RAG**: Dynamically adjusts retrieval and generation strategies
   - Query reformulation based on context
   - Iterative refinement of answers
   - Maximum iteration control for optimization

3. **Corrective RAG**: Implements self-correction mechanisms
   - Initial response generation
   - Response critique generation
   - Final response refinement with additional context
   - Multi-stage verification process

4. **Reranking RAG**: Uses FlashRank for advanced document reranking
   - Initial retrieval phase
   - Document reranking using FlashRank
   - Enhanced relevance scoring
   - Improved context selection

5. **Hybrid Search RAG**: Combines multiple search strategies
   - BM25 keyword-based search
   - Vector similarity search
   - Ensemble retrieval approach
   - Detailed search explanation generation

6. **Multi Index RAG**: Manages multiple vector stores
   - Source-specific document retrieval
   - Cross-source answer generation
   - Source contribution analysis
   - Comprehensive context management

7. **Query Expansion RAG**: Implements query transformation techniques
   - Query variation generation
   - Multi-query document retrieval
   - Result aggregation and deduplication
   - Enhanced retrieval coverage

8. **Self Adaptive RAG**: Automatically adapts to query characteristics
   - Query complexity assessment
   - Strategy selection based on query type
   - Dynamic retrieval adjustment
   - Performance optimization

9. **HyDE RAG**: Implements Hypothetical Document Embeddings
   - Hypothetical document generation
   - Multiple retrieval methods (hyde/hybrid/standard)
   - Configurable similarity thresholds
   - Hybrid retrieval weighting

## 🛠️ Technical Architecture

### Core Components

- **Document Processing**: 
  - Handles PDF, CSV, and TXT files
  - Implements chunk size and overlap controls
  - Supports multiple document sources

- **Vector Storage**:
  - Uses Chroma for vector storage
  - Persistent storage with automatic management
  - Multiple collection support

- **Embedding Models**:
  - Nomic AI embeddings (nomic-embed-text-v1.5)
  - Support for different embedding models

- **LLM Support**:
  - Multiple provider support (OpenAI, Anthropic, Google, Ollama, Groq)
  - Model selection flexibility
  - Consistent API interface

### Web Interface

- **Streamlit-based UI** with features:
  - Document upload and processing
  - RAG type selection
  - Model provider/model selection
  - Query interface with evaluation
  - Detailed results visualization
  - Performance metrics display

## 📋 Requirements

```
streamlit
PyPDF2 
pandas 
beautifulsoup4 
trafilatura 
validators 
requests
chromadb
ragas 
datasets 
plotly
langgraph 
langchain 
langchain-google-genai 
langchain-community
sentence-transformers
langchain-huggingface
langchain-anthropic 
langchain-openai
langchain-ollama
langchain-groq
ollama
einops
python-dotenv
flashrank
rank_bm25
```

## 🚀 Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd advanced-rag-system
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key
   ```

5. **Run the Application**:
   ```bash
   streamlit run home.py
   ```

## 💻 Usage

1. **Start the Application**:
   - Launch the application using `streamlit run home.py`
   - Click on "Launch Agentic-RAG" in the home interface

2. **Configure the System**:
   - Select desired RAG type from the sidebar
   - Choose model provider and specific model
   - Initialize the application

3. **Upload Documents**:
   - Use the document upload interface
   - Support for PDF, CSV, and TXT files
   - Multiple file upload supported

4. **Query Documents**:
   - Enter questions in the query interface
   - Adjust iteration settings if needed
   - Enable evaluation for performance metrics
   - Add ground truth for accuracy comparison

5. **View Results**:
   - Review answers in the results tabs
   - Check evaluation metrics if enabled
   - Examine retrieved contexts
   - Analyze query processing details

## 🔍 Evaluation Features

- Answer relevancy scoring
- Faithfulness measurement
- Context precision metrics
- Performance visualization
- Ground truth comparison
- Detailed metric explanations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## 📄 License

[License information here]

## 👥 Authors

[Author information here]
