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
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings import HuggingFaceEmbeddings
import operator
import chromadb
import tempfile
from utils.llm_manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

class ReRankingAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    initial_docs: Optional[List[str]]
    reranked_docs: Optional[List[str]]
    final_answer: Optional[str]

class ReRankingTools:
    def __init__(self, vectorstore: Any, llm: Any):
        self.vectorstore = vectorstore
        self.llm = llm
        # Initialize and rebuild the FlashrankRerank model
        self.compressor = FlashrankRerank()
        self.compressor.model_rebuild()  # Rebuild the model before using it
        
        # Create compression retriever with rebuilt model
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="initial_retrieval",
                description="Performs initial retrieval of documents",
                func=self._initial_retrieval
            ),
            Tool(
                name="rerank_documents",
                description="Re-ranks the retrieved documents",
                func=self._rerank_documents
            ),
            Tool(
                name="generate_response",
                description="Generates final response based on re-ranked documents",
                func=self._generate_response
            )
        ]
    
    def _initial_retrieval(self, query: str, k: int = 5) -> List[str]:
        """Perform initial retrieval of documents."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents for UI compatibility."""
        return self._initial_retrieval(query, k)

    def _rerank_documents(self, query: str) -> List[str]:
        """Re-rank documents using FlashRank."""
        docs = self.compression_retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]
    
    def _generate_response(self, query: str, context: List[str]) -> str:
        """Generate final response based on re-ranked documents."""
        combined_context = "\n\n".join(context)
        prompt = ChatPromptTemplate.from_template(
            "Based on the following context, please answer the query:\n"
            "Context: {context}\nQuery: {query}"
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": combined_context, "query": query})
        return response.content if hasattr(response, 'content') else response

class ReRankingRAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Re-ranking RAG system."""
        llm_manager = LLMManager()
        self.llm = llm_manager.initialize_llm(model_provider, model_name)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        
        # Create a persistent directory for Chroma
        self.persist_dir = os.path.join(tempfile.gettempdir(), "chroma_rerank_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        self.vectorstore = None
        self.reranking_tools = None
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
            collection_name="reranking_collection",
            persist_directory=self.persist_dir
        )
        
        self.reranking_tools = ReRankingTools(self.vectorstore, self.llm)
        
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the Re-ranking RAG system."""
        
        def initial_retrieval_node(state: ReRankingAgentState) -> ReRankingAgentState:
            try:
                docs = self.reranking_tools._initial_retrieval(state["query"])
                state["initial_docs"] = docs
            except Exception as e:
                logger.error(f"Error in initial_retrieval_node: {str(e)}")
                state["initial_docs"] = []
            return state
        
        def rerank_node(state: ReRankingAgentState) -> ReRankingAgentState:
            try:
                if state["initial_docs"]:
                    reranked_docs = self.reranking_tools._rerank_documents(state["query"])
                    state["reranked_docs"] = reranked_docs
                else:
                    state["reranked_docs"] = []
            except Exception as e:
                logger.error(f"Error in rerank_node: {str(e)}")
                state["reranked_docs"] = state["initial_docs"]  # Fallback to initial docs
            return state
        
        def response_node(state: ReRankingAgentState) -> ReRankingAgentState:
            try:
                docs_to_use = state["reranked_docs"] if state["reranked_docs"] else state["initial_docs"]
                if docs_to_use:
                    response = self.reranking_tools._generate_response(
                        query=state["query"],
                        context=docs_to_use
                    )
                    state["final_answer"] = response
                else:
                    state["final_answer"] = "No relevant documents found."
            except Exception as e:
                logger.error(f"Error in response_node: {str(e)}")
                state["final_answer"] = f"Error generating response: {str(e)}"
            return state
        
        # Create the graph
        workflow = StateGraph(ReRankingAgentState)
        
        # Add nodes
        workflow.add_node("initial_retrieval", initial_retrieval_node)
        workflow.add_node("rerank", rerank_node)
        workflow.add_node("response", response_node)
        
        # Add edges
        workflow.add_edge("initial_retrieval", "rerank")
        workflow.add_edge("rerank", "response")
        workflow.add_edge("response", END)
        
        # Set entry point
        workflow.set_entry_point("initial_retrieval")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(self, query: str, max_iterations: int = 1, **kwargs) -> Dict[str, Any]:
        """Run the re-ranking RAG process.
        
        Args:
            query (str): The user's query
            max_iterations (int): Ignored in reranking RAG, included for UI compatibility
            **kwargs: Additional keyword arguments for UI compatibility
        """
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        initial_state = {
            "messages": [],
            "query": query,
            "initial_docs": None,
            "reranked_docs": None,
            "final_answer": None
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "initial_docs": final_state["initial_docs"],
            "reranked_docs": final_state["reranked_docs"],
            "final_answer": final_state["final_answer"],
            "iterations": 1  # Always 1 for reranking RAG
        }