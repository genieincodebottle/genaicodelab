from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Any
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain.tools import Tool
import operator
import chromadb
import tempfile
import os
import logging
from dotenv import load_dotenv
from utils.llm_manager import LLMManager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Define state types for self-adaptive RAG
class SelfAdaptiveState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_query: str
    query_complexity: Optional[int]
    query_type: Optional[str]
    retrieval_strategy: Optional[str]
    retrieved_docs: Optional[List[str]]
    iteration: int
    max_iterations: int
    final_answer: Optional[str]
    feedback_metrics: Optional[Dict]

class SelfAdaptiveTools:
    def __init__(self, vectorstore: Any, llm: Any, model_name: Any):
        self.vectorstore = vectorstore
        self.llm = llm
        self.model_name = model_name
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="assess_query",
                description="Analyzes query complexity and determines retrieval strategy",
                func=self._assess_query
            ),
            Tool(
                name="retrieve_documents",
                description="Retrieves relevant documents based on strategy",
                func=self._retrieve_documents
            ),
            Tool(
                name="generate_answer",
                description="Generates answer based on retrieved context",
                func=self._generate_answer
            )
        ]
    
    def _assess_query(self, query: str) -> Dict:
        prompt = f"""Analyze the following query and provide:
        1. Complexity (rate from 1-5, where 1 is very simple and 5 is very complex)
        2. Query type (e.g., factual, analytical, open-ended)
        3. Suggested retrieval strategy (standard/query_expansion/hyde)
        
        Query: {query}
        
        Analysis:"""
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else response
        
        # Parse the response
        lines = content.split('\n')
        complexity = int(lines[0].split(':')[-1].strip())
        query_type = lines[1].split(':')[-1].strip()
        strategy = lines[2].split(':')[-1].strip()
        
        return {
            "complexity": complexity,
            "query_type": query_type,
            "retrieval_strategy": strategy
        }
    
    def _retrieve_documents(self, query: str, strategy: str = "standard") -> List[str]:
        if strategy == "standard":
            docs = self.vectorstore.similarity_search(query, k=3)
        elif strategy == "query_expansion":
            expanded_query = self._expand_query(query)
            docs = self.vectorstore.similarity_search(expanded_query, k=4)
        elif strategy == "hyde":
            hyde_query = self._generate_hyde_query(query)
            docs = self.vectorstore.similarity_search(hyde_query, k=4)
        else:
            docs = self.vectorstore.similarity_search(query, k=3)
            
        return [doc.page_content for doc in docs]
    
    def _expand_query(self, query: str) -> str:
        prompt = f"Expand the following query with relevant keywords:\nQuery: {query}\nExpanded query:"
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else response
    
    def _generate_hyde_query(self, query: str) -> str:
        prompt = f"""Generate a hypothetical document that would perfectly answer this query:
        Query: {query}
        Hypothetical Document:"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else response
    
    def _generate_answer(self, query: str, context: str, complexity: int) -> str:
        prompt = f"""Based on the query complexity of {complexity}, provide a {
            'concise' if complexity <= 2 else 'detailed'
        } answer to the following query using the given context:
        
        Query: {query}
        Context: {context}
        
        Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else response

class SelfAdaptiveRAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Self-Adaptive RAG system."""
        llm_manager = LLMManager()
        self.llm = llm_manager.initialize_llm(model_provider, model_name)
        self.model_name = model_name
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        
        self.persist_dir = os.path.join(tempfile.gettempdir(), "chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.vectorstore = None
        self.adaptive_rag_tools = None
        self.workflow = None
    
    def load_documents(self, documents: List[str]):
        """Process and load documents into the vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        texts = text_splitter.create_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name="self_adaptive_rag_collection",
            persist_directory=self.persist_dir
        )
        
        self.adaptive_rag_tools = SelfAdaptiveTools(self.vectorstore, self.llm, self.model_name)
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the Self-Adaptive RAG system."""
        
        def should_continue(state: SelfAdaptiveState) -> bool:
            """Determine if the workflow should continue iterating."""
            return (
                state["iteration"] < state["max_iterations"]
                and len(state.get("final_answer", "")) < 100
            )
        
        def assess_node(state: SelfAdaptiveState) -> SelfAdaptiveState:
            """Assess query complexity and determine retrieval strategy."""
            try:
                assessment = self.adaptive_rag_tools._assess_query(state["current_query"])
                state["query_complexity"] = assessment["complexity"]
                state["query_type"] = assessment["query_type"]
                state["retrieval_strategy"] = assessment["retrieval_strategy"]
            except Exception as e:
                logger.error(f"Error in assess_node: {str(e)}")
                state["query_complexity"] = 3
                state["query_type"] = "unknown"
                state["retrieval_strategy"] = "standard"
            return state
        
        def retrieve_node(state: SelfAdaptiveState) -> SelfAdaptiveState:
            """Retrieve documents based on the determined strategy."""
            try:
                docs = self.adaptive_rag_tools._retrieve_documents(
                    state["current_query"],
                    state["retrieval_strategy"]
                )
                state["retrieved_docs"] = docs
            except Exception as e:
                logger.error(f"Error in retrieve_node: {str(e)}")
                state["retrieved_docs"] = []
            return state
        
        def answer_node(state: SelfAdaptiveState) -> SelfAdaptiveState:
            """Generate the final answer."""
            try:
                if state["retrieved_docs"]:
                    context = " ".join(state["retrieved_docs"])
                    answer = self.adaptive_rag_tools._generate_answer(
                        query=state["current_query"],
                        context=context,
                        complexity=state["query_complexity"]
                    )
                    state["final_answer"] = answer
                else:
                    state["final_answer"] = "No relevant documents found to answer the query."
            except Exception as e:
                logger.error(f"Error in answer_node: {str(e)}")
                state["final_answer"] = f"Error generating answer: {str(e)}"
            
            state["iteration"] += 1
            return state
        
        # Create the graph
        workflow = StateGraph(SelfAdaptiveState)
        
        # Add nodes
        workflow.add_node("assess", assess_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("answer", answer_node)
        
        # Add edges
        workflow.add_edge("assess", "retrieve")
        workflow.add_edge("retrieve", "answer")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "answer",
            should_continue,
            {
                True: "assess",
                False: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("assess")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(self, query: str, max_iterations: int = 3) -> Dict:
        """Run the self-adaptive RAG process."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        initial_state = {
            "messages": [],
            "current_query": query,
            "query_complexity": None,
            "query_type": None,
            "retrieval_strategy": None,
            "retrieved_docs": None,
            "iteration": 0,
            "max_iterations": max_iterations,
            "final_answer": None,
            "feedback_metrics": None
        }
        
        final_state = self.workflow.invoke(initial_state)
        return {
            "final_answer": final_state["final_answer"],
            "iterations": final_state["iteration"],
            "query_complexity": final_state["query_complexity"],
            "query_type": final_state["query_type"],
            "retrieval_strategy": final_state["retrieval_strategy"]
        }