import os
import logging
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
from utils.llm_manager import LLMManager

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

class HyDEState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_query: str
    hypothetical_doc: Optional[str]
    retrieval_method: str  # 'standard' or 'hyde'
    retrieved_docs: Optional[List[str]]
    iteration: int
    max_iterations: int
    final_answer: Optional[str]
    feedback_metrics: Optional[Dict]

class HyDETools:
    def __init__(self, vectorstore: Any, llm: Any, model_name: Any):
        self.vectorstore = vectorstore
        self.llm = llm
        self.model_name = model_name
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="generate_hypothetical",
                description="Generates a hypothetical document for the query",
                func=self._generate_hypothetical_doc
            ),
            Tool(
                name="retrieve_documents",
                description="Retrieves documents using either standard or HyDE approach",
                func=self._retrieve_documents
            ),
            Tool(
                name="generate_answer",
                description="Generates final answer based on retrieved context",
                func=self._generate_answer
            )
        ]
    
    def _generate_hypothetical_doc(self, query: str, doc_style: str = "scientific") -> str:
        """Generate a hypothetical document that would perfectly answer the query."""
        style_prompts = {
            "scientific": "Write a scientific article excerpt that would answer",
            "technical": "Write a technical documentation excerpt that would answer",
            "narrative": "Write a narrative explanation that would answer"
        }
        
        style_text = style_prompts.get(doc_style, style_prompts["scientific"])
        
        prompt = f"""{style_text} this query perfectly.
        Make it detailed and factual, as if it were from a high-quality source.
        
        Query: {query}
        
        Hypothetical Document:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            return query  # Fallback to original query if generation fails
    
    def _retrieve_documents(
        self, 
        query: str, 
        method: str = 'hyde',
        hypothetical_doc: str = None,
        k_docs: int = 3,
        similarity_threshold: float = 1.5,  # Increased default threshold
        hybrid_alpha: float = 0.5
    ) -> List[str]:
        """Retrieve documents using either standard or HyDE approach."""
        try:
            if method == 'hyde' and hypothetical_doc:
                # HyDE retrieval
                hyde_results_with_scores = self.vectorstore.similarity_search_with_score(
                    hypothetical_doc,
                    k=k_docs * 2  # Get more docs initially for filtering
                )
                
                # Debug logging
                logger.info(f"HyDE scores: {[score for _, score in hyde_results_with_scores]}")
                
                # Filter by similarity threshold but ensure at least one document
                hyde_results = [
                    doc for doc, score in hyde_results_with_scores 
                    if score <= similarity_threshold
                ]
                
                # If no documents pass threshold, take the best one
                if not hyde_results and hyde_results_with_scores:
                    hyde_results = [hyde_results_with_scores[0][0]]
                
                hyde_results = hyde_results[:k_docs]
                
                # Optional: Also get standard results and combine
                if hybrid_alpha > 0:
                    standard_results_with_scores = self.vectorstore.similarity_search_with_score(
                        query,
                        k=k_docs * 2
                    )
                    
                    standard_results = [
                        doc for doc, score in standard_results_with_scores 
                        if score <= similarity_threshold
                    ]
                    
                    # If no documents pass threshold, take the best one
                    if not standard_results and standard_results_with_scores:
                        standard_results = [standard_results_with_scores[0][0]]
                    
                    standard_results = standard_results[:k_docs]
                    
                    # Combine results with weighting
                    all_docs = []
                    for i in range(max(len(hyde_results), len(standard_results))):
                        if i < len(hyde_results):
                            all_docs.append(hyde_results[i])
                        if i < len(standard_results):
                            all_docs.append(standard_results[i])
                    
                    # Deduplicate while preserving order
                    seen = set()
                    docs = []
                    for doc in all_docs:
                        if doc.page_content not in seen:
                            seen.add(doc.page_content)
                            docs.append(doc)
                    
                    # Trim to k_docs
                    docs = docs[:k_docs]
                else:
                    docs = hyde_results
            else:
                # Standard retrieval
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k_docs * 2
                )
                
                # Filter by similarity threshold but ensure at least one document
                docs = [
                    doc for doc, score in results_with_scores 
                    if score <= similarity_threshold
                ]
                
                # If no documents pass threshold, take the best one
                if not docs and results_with_scores:
                    docs = [results_with_scores[0][0]]
                
                docs = docs[:k_docs]
            
            # Ensure we always return at least the best match if available
            if not docs and method == 'hyde' and hypothetical_doc:
                # Fallback to direct retrieval with relaxed threshold
                docs = self.vectorstore.similarity_search(
                    hypothetical_doc,
                    k=1
                )
            
            return [doc.page_content for doc in docs]
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate the final answer using retrieved context."""
        prompt = f"""Based on the following context, provide a comprehensive answer to the query.
        
        Query: {query}
        Context: {context}
        
        Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."

class HyDERAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the HyDE-based RAG system."""
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
        self.hyde_tools = None
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
            collection_name="hyde_rag_collection",
            persist_directory=self.persist_dir
        )
        
        self.hyde_tools = HyDETools(self.vectorstore, self.llm, self.model_name)
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the HyDE-based RAG system."""
        
        def should_continue(state: HyDEState) -> bool:
            """Determine if the workflow should continue iterating."""
            return (
                state["iteration"] < state["max_iterations"]
                and len(state.get("final_answer", "")) < 100
            )
        
        def hypothetical_node(state: HyDEState) -> HyDEState:
            """Generate hypothetical document for the query."""
            try:
                doc_style = state.get("doc_style", "scientific")
                hypothetical = self.hyde_tools._generate_hypothetical_doc(
                    state["current_query"],
                    doc_style
                )
                state["hypothetical_doc"] = hypothetical
            except Exception as e:
                logger.error(f"Error in hypothetical_node: {str(e)}")
                state["hypothetical_doc"] = state["current_query"]
            return state
        
        def retrieve_node(state: HyDEState) -> HyDEState:
            """Retrieve relevant documents."""
            try:
                docs = self.hyde_tools._retrieve_documents(
                    query=state["current_query"],
                    method=state.get("retrieval_method", "hyde"),
                    hypothetical_doc=state.get("hypothetical_doc"),
                    k_docs=state.get("k_docs", 3),
                    similarity_threshold=state.get("similarity_threshold", 0.7),
                    hybrid_alpha=state.get("hybrid_alpha", 0.5)
                )
                state["retrieved_docs"] = docs
            except Exception as e:
                logger.error(f"Error in retrieve_node: {str(e)}")
                state["retrieved_docs"] = []
            return state
        
        def answer_node(state: HyDEState) -> HyDEState:
            """Generate the final answer."""
            try:
                if state["retrieved_docs"]:
                    context = " ".join(state["retrieved_docs"])
                    answer = self.hyde_tools._generate_answer(
                        query=state["current_query"],
                        context=context
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
        workflow = StateGraph(HyDEState)
        
        # Add nodes
        workflow.add_node("hypothetical", hypothetical_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("answer", answer_node)
        
        # Add edges
        workflow.add_edge("hypothetical", "retrieve")
        workflow.add_edge("retrieve", "answer")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "answer",
            should_continue,
            {
                True: "hypothetical",
                False: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("hypothetical")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(
        self,
        query: str,
        max_iterations: int = 3,
        doc_style: str = "scientific",
        k_docs: int = 3,
        similarity_threshold: float = 0.7,
        hybrid_alpha: float = 0.5,
        retrieval_method: str = "hyde"
    ) -> Dict:
        """Run the HyDE-based RAG process."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        initial_state = {
            "messages": [],
            "current_query": query,
            "hypothetical_doc": None,
            "retrieval_method": retrieval_method,
            "doc_style": doc_style,
            "k_docs": k_docs,
            "similarity_threshold": similarity_threshold,
            "hybrid_alpha": hybrid_alpha,
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
            "hypothetical_doc": final_state["hypothetical_doc"],
            "retrieval_method": final_state["retrieval_method"]
        }