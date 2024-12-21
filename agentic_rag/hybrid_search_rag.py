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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import operator
import chromadb
import tempfile
from utils.llm_manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

class HybridSearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    bm25_docs: Optional[List[str]]
    vector_docs: Optional[List[str]]
    ensemble_docs: Optional[List[str]]
    hybrid_explanation: Optional[str]
    final_answer: Optional[str]

class HybridSearchTools:
    def __init__(self, chunks: List[Any], vectorstore: Any, llm: Any, ensemble_retriever: Any):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.llm = llm
        self.ensemble_retriever = ensemble_retriever
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="hybrid_retrieval",
                description="Performs hybrid retrieval using both BM25 and vector search",
                func=self._hybrid_retrieval
            ),
            Tool(
                name="generate_explanation",
                description="Generates explanation of hybrid search process",
                func=self._generate_explanation
            ),
            Tool(
                name="generate_response",
                description="Generates final response based on retrieved documents",
                func=self._generate_response
            )
        ]
    
    def _hybrid_retrieval(self, query: str) -> Dict[str, List[str]]:
        """Perform hybrid retrieval using both BM25 and vector search."""
        try:
            # Get ensemble results
            ensemble_docs = self.ensemble_retriever.get_relevant_documents(query)
            ensemble_texts = [str(doc.page_content) for doc in ensemble_docs]
            
            # Get vector search results
            vector_docs = self.vectorstore.similarity_search(query, k=5)
            vector_texts = [str(doc.page_content) for doc in vector_docs]
            
            # Get BM25 results
            bm25_retriever = BM25Retriever.from_documents(self.chunks)
            bm25_retriever.k = 5
            bm25_docs = bm25_retriever.get_relevant_documents(query)
            bm25_texts = [str(doc.page_content) for doc in bm25_docs]
            
            return {
                "bm25_docs": bm25_texts,
                "vector_docs": vector_texts,
                "ensemble_docs": ensemble_texts
            }
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return {
                "bm25_docs": [],
                "vector_docs": [],
                "ensemble_docs": []
            }
    
    def _generate_explanation(self, query: str) -> str:
        """Generate explanation of hybrid search process."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Explain how the hybrid search process, combining keyword-based (BM25) and "
                "semantic search (vector embeddings), might have improved the retrieval of "
                "relevant information for answering this query:\n\n"
                "Query: {query}\n\n"
                "Consider the potential benefits of this approach compared to using only "
                "one search method. Focus on how BM25's keyword matching and vector search's "
                "semantic understanding complement each other."
            )
            chain = prompt | self.llm
            explanation = chain.invoke({"query": query})
            return str(explanation.content) if hasattr(explanation, 'content') else str(explanation)
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Unable to generate hybrid search explanation due to an error."
    
    def _generate_response(self, query: str, context: List[str]) -> str:
        """Generate final response based on retrieved documents."""
        try:
            # Ensure all context items are strings and join them
            context_str = "\n\n".join([str(doc) for doc in context])
            
            prompt = ChatPromptTemplate.from_template(
                "Based on the following context retrieved using hybrid search "
                "(combining keyword-based and semantic search), please answer the query:\n"
                "Context: {context}\nQuery: {query}\nAnswer:"
            )
            chain = prompt | self.llm
            response = chain.invoke({"context": context_str, "query": query})
            return str(response.content) if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

class HybridSearchRAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Hybrid Search RAG system."""
        llm_manager = LLMManager()
        self.llm = llm_manager.initialize_llm(model_provider, model_name)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        
        self.persist_dir = os.path.join(tempfile.gettempdir(), "chroma_hybrid_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        self.chunks = None
        self.vectorstore = None
        self.ensemble_retriever = None
        self.hybrid_tools = None
        self.workflow = None
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve documents for UI compatibility."""
        try:
            if not self.ensemble_retriever:
                raise ValueError("Ensemble retriever not initialized")
            docs = self.ensemble_retriever.get_relevant_documents(query)
            return [str(doc.page_content) for doc in docs[:k]]
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            return []
    
    def load_documents(self, documents: List[str]):
        """Process and load documents into the vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.chunks = text_splitter.create_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name="hybrid_search_collection",
            persist_directory=self.persist_dir
        )
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(self.chunks)
        bm25_retriever.k = 5
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                self.vectorstore.as_retriever(search_kwargs={"k": 5})
            ],
            weights=[0.5, 0.5]
        )
        
        self.hybrid_tools = HybridSearchTools(
            self.chunks,
            self.vectorstore,
            self.llm,
            self.ensemble_retriever
        )
        
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the Hybrid Search RAG system."""
        
        def retrieval_node(state: HybridSearchState) -> HybridSearchState:
            try:
                results = self.hybrid_tools._hybrid_retrieval(state["query"])
                state.update(results)
            except Exception as e:
                logger.error(f"Error in retrieval_node: {str(e)}")
                state.update({
                    "bm25_docs": [],
                    "vector_docs": [],
                    "ensemble_docs": []
                })
            return state
        
        def explanation_node(state: HybridSearchState) -> HybridSearchState:
            try:
                explanation = self.hybrid_tools._generate_explanation(state["query"])
                state["hybrid_explanation"] = str(explanation)
            except Exception as e:
                logger.error(f"Error in explanation_node: {str(e)}")
                state["hybrid_explanation"] = f"Error generating explanation: {str(e)}"
            return state
        
        def response_node(state: HybridSearchState) -> HybridSearchState:
            try:
                if state.get("ensemble_docs"):
                    response = self.hybrid_tools._generate_response(
                        query=state["query"],
                        context=state["ensemble_docs"]
                    )
                    state["final_answer"] = str(response)
                else:
                    state["final_answer"] = "No relevant documents found."
            except Exception as e:
                logger.error(f"Error in response_node: {str(e)}")
                state["final_answer"] = f"Error generating response: {str(e)}"
            return state
        
        # Create the graph
        workflow = StateGraph(HybridSearchState)
        
        # Add nodes
        workflow.add_node("retrieval", retrieval_node)
        workflow.add_node("explanation", explanation_node)
        workflow.add_node("response", response_node)
        
        # Add edges
        workflow.add_edge("retrieval", "explanation")
        workflow.add_edge("explanation", "response")
        workflow.add_edge("response", END)
        
        # Set entry point
        workflow.set_entry_point("retrieval")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(self, query: str, max_iterations: int = 1, **kwargs) -> Dict[str, Any]:
        """Run the hybrid search RAG process."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        initial_state = {
            "messages": [],
            "query": query,
            "bm25_docs": None,
            "vector_docs": None,
            "ensemble_docs": None,
            "hybrid_explanation": None,
            "final_answer": None
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "bm25_docs": final_state["bm25_docs"],
            "vector_docs": final_state["vector_docs"],
            "ensemble_docs": final_state["ensemble_docs"],
            "hybrid_explanation": final_state["hybrid_explanation"],
            "final_answer": final_state["final_answer"],
            "iterations": 1  # Always 1 for hybrid search
        }