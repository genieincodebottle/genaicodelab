import os
from dataclasses import dataclass
from typing import List, Dict, Optional

# Get the current directory of this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define path to images directory (assuming it's in a 'static' folder at project root)
IMAGES_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'static', 'images')

@dataclass
class RAGDescription:
    title: str
    description: str
    key_features: List[str]
    use_cases: List[str]
    diagram_path: Optional[str] = None
    resources: Optional[Dict[str, str]] = None

class RAGDescriptions:
    MAIN_DESCRIPTION = RAGDescription(
        title="Agentic RAG System",
        description="""
        Agentic RAG (Retrieval-Augmented Generation) is an advanced AI system that combines LangGraph-based 
        agents with sophisticated document retrieval and generation capabilities. It enables dynamic, 
        context-aware responses while maintaining accuracy and relevance through various specialized RAG approaches.
        """,
        key_features=[
            "LangGraph-based agent coordination",
            "Dynamic context retrieval",
            "Adaptive response generation",
            "Multi-strategy document processing"
        ],
        use_cases=[
            "Complex document analysis",
            "Intelligent query processing",
            "Context-aware response generation",
            "Dynamic information retrieval"
        ],
        diagram_path=os.path.join(IMAGES_DIR, 'main_rag.png'),
        resources={
            "LangGraph Documentation": "https://python.langchain.com/docs/langgraph",
            "RAG Overview": "https://python.langchain.com/docs/use_cases/question_answering/",
            "Agentic RAG Paper": "https://arxiv.org/abs/2312.04548"
        }
    )

    RAG_TYPES: Dict[str, RAGDescription] = {
        "Basic RAG": RAGDescription(
            title="Basic RAG",
            description="""
            A foundational RAG implementation that performs straightforward document retrieval 
            and response generation using vector similarity search. This approach forms the basis 
            for more complex RAG systems, providing essential functionality for document retrieval 
            and answer generation.
            """,
            key_features=[
                "Vector-based similarity search",
                "Direct document retrieval",
                "Single-pass generation",
                "Basic context integration",
                "Straightforward implementation"
            ],
            use_cases=[
                "Simple question answering",
                "Document summarization",
                "Basic information retrieval",
                "Straightforward queries",
                "Initial RAG implementation"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'basic_rag.png'),
            resources={
                "Basic RAG Tutorial": "https://python.langchain.com/docs/use_cases/question_answering/quickstart",
                "Vector Store Guide": "https://python.langchain.com/docs/integrations/vectorstores/",
                "Implementation Details": "https://blog.langchain.dev/building-ragventures/"
            }
        ),
        
        "Adaptive RAG": RAGDescription(
            title="Adaptive RAG",
            description="""
            An advanced RAG system that dynamically adjusts its retrieval and generation strategies 
            based on query complexity and context requirements. It uses sophisticated algorithms to 
            analyze queries and adapt its approach in real-time, ensuring optimal performance for 
            different types of questions and document contexts.
            """,
            key_features=[
                "Dynamic strategy adjustment",
                "Query complexity analysis",
                "Adaptive context retrieval",
                "Flexible response generation",
                "Real-time optimization"
            ],
            use_cases=[
                "Complex analytical queries",
                "Dynamic document analysis",
                "Context-sensitive research",
                "Adaptive information retrieval",
                "Multi-step reasoning tasks"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'adaptive_rag.png'),
            resources={
                "Adaptive RAG Guide": "https://python.langchain.com/docs/use_cases/question_answering/adaptive_retrieval",
                "Strategy Optimization": "https://www.pinecone.io/learn/adaptive-retrieval/",
                "Research Paper": "https://arxiv.org/abs/2402.08865"
            }
        ),

        "Corrective RAG": RAGDescription(
            title="Corrective RAG",
            description="""
            Corrective RAG is a technique that introduces an additional step to verify and correct 
            the information retrieved before generating the final response. This method aims to reduce 
            errors and inconsistencies in the generated output by cross-checking the retrieved information 
            against known facts or trusted sources. It often involves a separate model or module dedicated 
            to fact-checking and error correction.
            """,
            key_features=[
                "Iterative response validation",
                "Self-correction mechanisms",
                "Fact-checking integration",
                "Context verification",
                "Confidence scoring"
            ],
            use_cases=[
                "High-accuracy requirements",
                "Fact-based responses",
                "Critical information retrieval",
                "Quality-sensitive applications",
                "Research validation"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'corrective_rag.png'),
            resources={
                "Self-Correction Guide": "https://python.langchain.com/docs/use_cases/question_answering/quality_control",
                "Validation Methods": "https://www.pinecone.io/learn/self-correcting-rag/",
                "Research Implementation": "https://arxiv.org/abs/2401.13388"
            }
        ),

        "Reranking RAG": RAGDescription(
            title="Reranking RAG",
            description="""
            Enhances retrieval quality by implementing a two-stage process: initial retrieval followed 
            by context-aware reranking of results. This approach significantly improves the relevance 
            and quality of retrieved documents by applying sophisticated ranking algorithms to refine 
            the initial search results.
            """,
            key_features=[
                "Two-stage retrieval process",
                "Context-aware reranking",
                "Relevance optimization",
                "Dynamic scoring system",
                "Multi-factor ranking"
            ],
            use_cases=[
                "Precision-critical retrieval",
                "Large document collections",
                "Research document analysis",
                "Complex information needs",
                "Semantic search refinement"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'reranking_rag.png'),
            resources={
                "Reranking Tutorial": "https://python.langchain.com/docs/use_cases/question_answering/reranking",
                "Advanced Techniques": "https://www.pinecone.io/learn/cohere-rerank/",
                "Implementation Guide": "https://blog.langchain.dev/advanced-rag-reranking/"
            }
        ),

        "Hybrid Search RAG": RAGDescription(
            title="Hybrid Search RAG",
            description="""
            Combines multiple search strategies including vector search and keyword-based methods 
            to optimize retrieval effectiveness. This hybrid approach leverages the strengths of 
            different search methodologies to provide more comprehensive and accurate results.
            """,
            key_features=[
                "Multiple search strategies",
                "Hybrid scoring system",
                "Optimized retrieval balance",
                "Strategy combination",
                "Adaptive weighting"
            ],
            use_cases=[
                "Diverse content types",
                "Mixed query requirements",
                "Complex search needs",
                "Comprehensive retrieval",
                "Multi-format documents"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'hybrid_search_rag.png'),
            resources={
                "Hybrid Search Guide": "https://python.langchain.com/docs/use_cases/question_answering/hybrid_search",
                "Implementation Details": "https://www.pinecone.io/learn/hybrid-search/",
                "Research Paper": "https://arxiv.org/abs/2310.11511"
            }
        ),

        "Multi Index RAG": RAGDescription(
            title="Multi Index RAG",
            description="""
            Utilizes multiple specialized indexes for different types of content and queries, enabling 
            more targeted and effective retrieval. This approach maintains separate indexes optimized 
            for specific content types or query patterns, improving overall search effectiveness.
            """,
            key_features=[
                "Multiple specialized indexes",
                "Content-type awareness",
                "Targeted retrieval",
                "Index selection logic",
                "Optimized storage"
            ],
            use_cases=[
                "Mixed content types",
                "Specialized queries",
                "Domain-specific retrieval",
                "Multi-source information",
                "Structured data queries"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'multi_index_rag.png'),
            resources={
                "Multi-Index Tutorial": "https://python.langchain.com/docs/use_cases/question_answering/multi_index",
                "Index Optimization": "https://www.pinecone.io/learn/multi-index/",
                "Technical Guide": "https://blog.langchain.dev/multi-index-rag/"
            }
        ),

        "Query Expansion RAG": RAGDescription(
            title="Query Expansion RAG",
            description="""
            Enhances retrieval by automatically expanding and reformulating queries to capture more 
            relevant context and information. This system generates multiple variations of the original 
            query to improve coverage and catch relevant documents that might be missed by a single query.
            """,
            key_features=[
                "Query reformulation",
                "Semantic expansion",
                "Context enhancement",
                "Multiple query versions",
                "Synonym integration"
            ],
            use_cases=[
                "Complex information needs",
                "Ambiguous queries",
                "Comprehensive retrieval",
                "Semantic search",
                "Research exploration"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'query_expansion_rag.png'),
            resources={
                "Query Expansion Guide": "https://python.langchain.com/docs/use_cases/question_answering/query_expansion",
                "Implementation Details": "https://www.pinecone.io/learn/query-expansion/",
                "Research Paper": "https://arxiv.org/abs/2312.06674"
            }
        ),

        "Self Adaptive RAG": RAGDescription(
            title="Self Adaptive RAG",
            description="""
            A sophisticated system that learns and adapts its strategies based on query patterns and 
            response effectiveness. It incorporates machine learning techniques to continuously improve 
            its performance by learning from past interactions and results.
            """,
            key_features=[
                "Strategy learning",
                "Performance adaptation",
                "Pattern recognition",
                "Dynamic optimization",
                "Continuous improvement"
            ],
            use_cases=[
                "Learning environments",
                "Pattern-based queries",
                "Adaptive systems",
                "Performance-critical applications",
                "Interactive systems"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'self_adaptive_rag.png'),
            resources={
                "Self-Adaptive Guide": "https://python.langchain.com/docs/use_cases/question_answering/self_adaptive",
                "Implementation Tutorial": "https://www.pinecone.io/learn/self-adaptive-rag/",
                "Research Paper": "https://arxiv.org/abs/2401.08406"
            }
        ),

        "HyDE RAG": RAGDescription(
            title="HyDE RAG",
            description="""
            Hypothetical Document Embedding RAG generates synthetic documents to improve retrieval 
            accuracy and context understanding. This innovative approach creates hypothetical ideal 
            documents that would perfectly answer a query, then uses these to guide the retrieval process.
            """,
            key_features=[
                "Hypothetical document generation",
                "Enhanced embedding search",
                "Context simulation",
                "Improved retrieval accuracy",
                "Synthetic content integration"
            ],
            use_cases=[
                "Sparse information scenarios",
                "Novel queries",
                "Context generation",
                "Improved accuracy needs",
                "Complex information synthesis"
            ],
            diagram_path=os.path.join(IMAGES_DIR, 'hyde_rag.png'),
            resources={
                "HyDE RAG Tutorial": "https://python.langchain.com/docs/use_cases/question_answering/hyde",
                "Technical Details": "https://www.pinecone.io/learn/hypothetical-document-embeddings/",
                "Research Paper": "https://arxiv.org/abs/2212.10496"
            }
        )
    }