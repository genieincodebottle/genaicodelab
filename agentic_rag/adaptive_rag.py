import os
import logging
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain.tools import Tool
import operator
import chromadb
import tempfile
from utils.llm_manager import LLMManager

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", 200))

# Define state types
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_query: str
    retrieved_docs: Optional[List[str]]
    iteration: int
    max_iterations: int
    final_answer: Optional[str]

class RAGTools:
    def __init__(self, vectorstore: Any, llm: Any, model_name: Any):
        self.vectorstore = vectorstore
        self.llm = llm
        self.tools = self._create_tools()
        self.model_name = model_name
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="query_reformulation",
                description="Reformulates the query based on context and previous iterations",
                func=self._query_reformulation
            ),
            Tool(
                name="retrieve_documents",
                description="Retrieves relevant documents based on the query",
                func=self._retrieve_documents
            ),
            Tool(
                name="generate_answer",
                description="Generates an answer based on the query and retrieved context",
                func=self._generate_answer
            )
        ]
    
    def _query_reformulation(self, query: str, context: str) -> str:
        prompt = f"""Given the original query and the current context, please reformulate the query to be more specific and targeted:
        
        Original query: {query}
        Current context: {context}
        
        Reformulated query:"""
        
        response = self.llm.invoke(prompt)
        if self.model_name == "llama3.1-latest" or self.model_name == "llama3.2:1b":
            return response
        return response.content
    
    def _retrieve_documents(self, query: str) -> List[str]:
        docs = self.vectorstore.similarity_search(query, k=4)
        return [doc.page_content for doc in docs]
    
    def _generate_answer(self, query: str, context: str) -> str:
        prompt = f"""Please answer the following query based on the given context:
        
        Query: {query}
        Context: {context}
        
        Answer:"""
        
        response = self.llm.invoke(prompt)
        if self.model_name == "llama3.1-latest" or self.model_name == "llama3.2:1b":
            return response
        return response.content

class AdaptiveRAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Agentic Adaptive RAG system."""
        llm_manager = LLMManager()
        self.llm = llm_manager.initialize_llm(model_provider, model_name)
        
        self.model_name = model_name
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        
        # Create a persistent directory for Chroma
        self.persist_dir = os.path.join(tempfile.gettempdir(), "chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        self.vectorstore = None
        self.rag_tools = None
        self.workflow = None
        
    def load_documents(self, documents: List[str]):
        """Process and load documents into the vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        texts = text_splitter.create_documents(documents)
        
        # Initialize Chroma vectorstore with persistent client
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name="rag_collection",
            persist_directory=self.persist_dir
        )
        
        # Initialize RAG tools
        self.rag_tools = RAGTools(self.vectorstore, self.llm, self.model_name)
        
        # Create the workflow
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the RAG system."""
        
        def should_continue(state: AgentState) -> bool:
            return (
                state["iteration"] < state["max_iterations"] 
                and len(state.get("final_answer", "")) < 100
            )
        
        def retrieve_node(state: AgentState) -> AgentState:
            try:
                docs = self.rag_tools._retrieve_documents(state["current_query"])
                state["retrieved_docs"] = docs
            except Exception as e:
                print(f"Error in retrieve_node: {str(e)}")
                state["retrieved_docs"] = []
            return state
        
        def reformulate_node(state: AgentState) -> AgentState:
            if state["iteration"] > 0 and state["retrieved_docs"]:
                try:
                    context = " ".join(state["retrieved_docs"])
                    new_query = self.rag_tools._query_reformulation(
                        query=state["current_query"],
                        context=context
                    )
                    state["current_query"] = new_query
                except Exception as e:
                    print(f"Error in reformulate_node: {str(e)}")
            return state
        
        def answer_node(state: AgentState) -> AgentState:
            try:
                if state["retrieved_docs"]:
                    context = " ".join(state["retrieved_docs"])
                    answer = self.rag_tools._generate_answer(
                        query=state["current_query"],
                        context=context
                    )
                    state["final_answer"] = answer
                else:
                    state["final_answer"] = "No relevant documents found to answer the query."
            except Exception as e:
                print(f"Error in answer_node: {str(e)}")
                state["final_answer"] = f"Error generating answer: {str(e)}"
            
            state["iteration"] += 1
            return state
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reformulate", reformulate_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("answer", answer_node)
        
        # Add edges
        workflow.add_edge("reformulate", "retrieve")
        workflow.add_edge("retrieve", "answer")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "answer",
            should_continue,
            {
                True: "reformulate",
                False: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("reformulate")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Run the adaptive RAG process."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        # Initialize state
        initial_state = {
            "messages": [],
            "current_query": query,
            "retrieved_docs": None,
            "iteration": 0,
            "max_iterations": max_iterations,
            "final_answer": None
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "iterations": final_state["iteration"],
            "final_query": final_state["current_query"]
        }
