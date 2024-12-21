from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
import operator
import chromadb
import tempfile
import os
import logging
from utils.llm_manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

class BasicRAGState(TypedDict):
    """State definition for Basic RAG workflow"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_query: str
    retrieved_docs: Optional[List[str]]
    final_answer: Optional[str]
    iteration: int
    max_iterations: int

class BasicRAGTools:
    """Tools for basic RAG operations"""
    def __init__(self, vectorstore: Any, llm: Any):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def _retrieve_documents(self, query: str) -> List[str]:
        """Retrieve relevant documents based on query"""
        try:
            docs = self.vectorstore.similarity_search(query, k=4)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer based on query and context"""
        try:
            prompt = f"""Please answer the following query based on the given context:
            
            Query: {query}
            Context: {context}
            
            Answer:"""
            
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else response
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

class BasicRAG:
    """LangGraph-based Basic RAG implementation"""
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Basic RAG system"""
        llm_manager = LLMManager()
        self.llm = llm_manager.initialize_llm(model_provider, model_name)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        
        # Create a persistent directory for Chroma
        self.persist_dir = os.path.join(tempfile.gettempdir(), "basic_chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        self.vectorstore = None
        self.basic_rag_tools = None
        self.workflow = None

    def load_documents(self, documents: List[str]):
        """Process and load documents into vector store"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            texts = text_splitter.create_documents(documents)
            
            # Initialize Chroma vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name="basic_rag_collection",
                persist_directory=self.persist_dir
            )
            
            # Initialize RAG tools
            self.basic_rag_tools = BasicRAGTools(self.vectorstore, self.llm)
            
            # Create workflow
            self._create_workflow()
            logger.info(f"Loaded {len(texts)} document chunks into vector store")
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def _create_workflow(self):
        """Create the LangGraph workflow"""
        def retrieve_node(state: BasicRAGState) -> BasicRAGState:
            """Node for document retrieval"""
            try:
                docs = self.basic_rag_tools._retrieve_documents(state["current_query"])
                state["retrieved_docs"] = docs
            except Exception as e:
                logger.error(f"Error in retrieve node: {str(e)}")
                state["retrieved_docs"] = []
            return state
        
        def answer_node(state: BasicRAGState) -> BasicRAGState:
            """Node for answer generation"""
            try:
                if state["retrieved_docs"]:
                    context = " ".join(state["retrieved_docs"])
                    answer = self.basic_rag_tools._generate_answer(
                        query=state["current_query"],
                        context=context
                    )
                    state["final_answer"] = answer
                else:
                    state["final_answer"] = "No relevant documents found to answer the query."
                
                # Increment iteration counter
                state["iteration"] = state["iteration"] + 1
            except Exception as e:
                logger.error(f"Error in answer node: {str(e)}")
                state["final_answer"] = f"Error generating answer: {str(e)}"
            return state
        
        # Create the graph
        workflow = StateGraph(BasicRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("answer", answer_node)
        
        # Add edges
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Compile workflow
        self.workflow = workflow.compile()

    def run(self, query: str, max_iterations: int = 1) -> Dict[str, Any]:
        """Run the RAG process"""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        try:
            # Initialize state
            initial_state = {
                "messages": [],
                "current_query": query,
                "retrieved_docs": None,
                "final_answer": None,
                "iteration": 0,
                "max_iterations": max_iterations
            }
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "final_answer": final_state["final_answer"],
                "retrieved_docs": final_state["retrieved_docs"],
                "current_query": final_state["current_query"],
                "iterations": final_state["iteration"]
            }
        except Exception as e:
            logger.error(f"Error running RAG process: {str(e)}")
            return {
                "final_answer": f"Error: {str(e)}",
                "retrieved_docs": [],
                "current_query": query
            }