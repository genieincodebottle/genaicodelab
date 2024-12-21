import os
import logging
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
import operator
import chromadb
import tempfile
from utils.llm_manager import LLMManager

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

class CorrectiveAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    retrieved_docs: Optional[List[str]]  
    rag_initial_response: Optional[str]   
    rag_critique: Optional[str]          
    additional_context: Optional[List[str]]  
    rag_final_answer: Optional[str]      
    iteration: int
    max_iterations: int

class CorrectionTools:
    def __init__(self, vectorstore: Any, llm: Any):
        self.vectorstore = vectorstore
        self.llm = llm
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="retrieve_documents",
                description="Retrieves relevant documents based on the query",
                func=self._retrieve_documents
            ),
            Tool(
                name="generate_initial_response",
                description="Generates initial response based on context",
                func=self._generate_initial_response
            ),
            Tool(
                name="generate_critique",
                description="Generates critique of the initial response",
                func=self._generate_critique
            ),
            Tool(
                name="generate_final_response",
                description="Generates final improved response",
                func=self._generate_final_response
            )
        ]
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def _generate_initial_response(self, query: str, context: str) -> str:
        """Generate initial response based on context."""
        prompt = ChatPromptTemplate.from_template(
            "Based on the following context, please answer the query:\n"
            "Context: {context}\nQuery: {query}"
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})
        return response.content if hasattr(response, 'content') else response
    
    def _generate_critique(self, query: str, response: str) -> str:
        """Generate critique of the response."""
        prompt = ChatPromptTemplate.from_template(
            "Please critique the following response to the query. "
            "Identify any potential errors, missing information, or areas for improvement:\n"
            "Query: {query}\nResponse: {response}"
        )
        chain = prompt | self.llm
        critique = chain.invoke({"response": response, "query": query})
        return critique.content if hasattr(critique, 'content') else critique
    
    def _generate_final_response(self, query: str, initial_response: str, 
                               critique: str, additional_context: str) -> str:
        """Generate improved response based on critique and additional context."""
        prompt = ChatPromptTemplate.from_template(
            "Based on the initial response, critique, and additional context, "
            "please provide an improved answer to the query:\n"
            "Initial Response: {initial_response}\nCritique: {critique}\n"
            "Additional Context: {additional_context}\nQuery: {query}"
        )
        chain = prompt | self.llm
        final_response = chain.invoke({
            "initial_response": initial_response,
            "critique": critique,
            "additional_context": additional_context,
            "query": query
        })
        return final_response.content if hasattr(final_response, 'content') else final_response

class CorrectiveRAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Corrective Agentic RAG system."""
               
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
        self.correction_tools = None
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
            collection_name="corrective_rag_collection",
            persist_directory=self.persist_dir
        )
        
        # Initialize correction tools
        self.correction_tools = CorrectionTools(self.vectorstore, self.llm)
        
        # Create the workflow
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the Corrective RAG system."""
        
        def should_continue(state: CorrectiveAgentState) -> bool:
            return (
                state["iteration"] < state["max_iterations"]
                and len(state.get("rag_final_answer", "")) < 100
            )
        
        def retrieve_node(state: CorrectiveAgentState) -> CorrectiveAgentState:
            try:
                docs = self.correction_tools._retrieve_documents(state["query"])
                state["retrieved_docs"] = docs
            except Exception as e:
                logger.error(f"Error in retrieve_node: {str(e)}")
                state["retrieved_docs"] = []
            return state
        
        def initial_response_node(state: CorrectiveAgentState) -> CorrectiveAgentState:
            try:
                if state["retrieved_docs"]:
                    context = "\n".join(state["retrieved_docs"])
                    response = self.correction_tools._generate_initial_response(
                        query=state["query"],
                        context=context
                    )
                    state["rag_initial_response"] = response
                else:
                    state["rag_initial_response"] = "No relevant documents found."
            except Exception as e:
                logger.error(f"Error in initial_response_node: {str(e)}")
                state["rag_initial_response"] = f"Error generating response: {str(e)}"
            return state
        
        def critique_node(state: CorrectiveAgentState) -> CorrectiveAgentState:
            try:
                if state["rag_initial_response"]:
                    critique = self.correction_tools._generate_critique(
                        query=state["query"],
                        response=state["rag_initial_response"]
                    )
                    state["rag_critique"] = critique
                else:
                    state["rag_critique"] = "No initial response to critique."
            except Exception as e:
                logger.error(f"Error in critique_node: {str(e)}")
                state["rag_critique"] = f"Error generating critique: {str(e)}"
            return state
        
        def final_response_node(state: CorrectiveAgentState) -> CorrectiveAgentState:
            try:
                additional_docs = self.correction_tools._retrieve_documents(state["rag_critique"], k=2)
                state["additional_context"] = additional_docs
                
                if additional_docs:
                    additional_context = "\n".join(additional_docs)
                    final_answer = self.correction_tools._generate_final_response(
                        query=state["query"],
                        initial_response=state["rag_initial_response"],
                        critique=state["rag_critique"],
                        additional_context=additional_context
                    )
                    state["rag_final_answer"] = final_answer
                else:
                    state["rag_final_answer"] = state["rag_initial_response"]
            except Exception as e:
                logger.error(f"Error in final_response_node: {str(e)}")
                state["rag_final_answer"] = f"Error generating final answer: {str(e)}"
            
            state["iteration"] += 1
            return state
        
        # Create the graph
        workflow = StateGraph(CorrectiveAgentState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("initial_response", initial_response_node)
        workflow.add_node("critique", critique_node)
        workflow.add_node("final_response", final_response_node)
        
        # Add edges
        workflow.add_edge("retrieve", "initial_response")
        workflow.add_edge("initial_response", "critique")
        workflow.add_edge("critique", "final_response")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "final_response",
            should_continue,
            {
                True: "critique",
                False: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(self, query: str, max_iterations: int = 2) -> Dict[str, Any]:
        """Run the corrective agentic RAG process."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        # Initialize state with new key names
        initial_state = {
            "messages": [],
            "query": query,
            "retrieved_docs": None,
            "rag_initial_response": None,
            "rag_critique": None,
            "additional_context": None,
            "rag_final_answer": None,
            "iteration": 0,
            "max_iterations": max_iterations
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Return with original key names for compatibility
        return {
            "initial_response": final_state["rag_initial_response"],
            "critique": final_state["rag_critique"],
            "final_answer": final_state["rag_final_answer"],
            "iterations": final_state["iteration"]
        }