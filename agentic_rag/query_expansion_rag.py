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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

class QueryTransformState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    transformed_queries: Optional[List[str]]
    all_retrieved_docs: Optional[List[str]]
    combined_context: Optional[str]
    final_answer: Optional[str]

class QueryTransformTools:
    def __init__(self, vectorstore: Any, llm: Any):
        self.vectorstore = vectorstore
        self.llm = llm
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="transform_query",
                description="Transforms the original query into multiple variations",
                func=self._transform_query
            ),
            Tool(
                name="multi_retrieve",
                description="Retrieves documents using multiple queries",
                func=self._multi_retrieve
            ),
            Tool(
                name="generate_response",
                description="Generates final response based on combined context",
                func=self._generate_response
            )
        ]
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve documents for UI compatibility."""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [str(doc.page_content) for doc in docs]
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            return []
    
    def _transform_query(self, query: str) -> List[str]:
        """Transform the original query into multiple variations."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Given the original query, generate 3 alternative versions that might improve "
                "retrieval effectiveness. Each version should capture a different aspect or "
                "use different terminology related to the original query. Format each query "
                "on a new line without numbering.\n"
                "Original query: {query}\n"
                "Alternative queries:"
            )
            chain = prompt | self.llm
            response = chain.invoke({"query": query})
            transformed = str(response.content).strip().split('\n')
            # Clean and validate transformed queries
            transformed = [q.strip() for q in transformed if q.strip()][:3]
            if not transformed:
                transformed = [query]  # Fallback to original query
            return transformed
        except Exception as e:
            logger.error(f"Error in query transformation: {str(e)}")
            return [query]
    
    def _multi_retrieve(self, queries: List[str], k: int = 2) -> Dict[str, Any]:
        """Retrieve documents using multiple queries."""
        try:
            all_docs = []
            for query in queries:
                docs = self.vectorstore.similarity_search(query, k=k)
                all_docs.extend([str(doc.page_content) for doc in docs])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc not in seen:
                    seen.add(doc)
                    unique_docs.append(doc)
            
            combined_context = "\n\n".join(unique_docs)
            
            return {
                "all_retrieved_docs": unique_docs,
                "combined_context": combined_context
            }
        except Exception as e:
            logger.error(f"Error in multi retrieve: {str(e)}")
            return {
                "all_retrieved_docs": [],
                "combined_context": ""
            }
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate final response based on combined context."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an AI assistant tasked with answering questions based on the "
                "provided context. The context contains information retrieved using the "
                "original query and its transformed versions.\n\n"
                "Original Question: {query}\n"
                "Context: {context}\n"
                "Answer:"
            )
            chain = prompt | self.llm
            response = chain.invoke({"query": query, "context": context})
            return str(response.content) if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

class QueryTransformRAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Query Transformation RAG system."""
        llm_manager = LLMManager()
        self.llm = llm_manager.initialize_llm(model_provider, model_name)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        
        self.persist_dir = os.path.join(tempfile.gettempdir(), "chroma_query_transform_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        self.vectorstore = None
        self.transform_tools = None
        self.workflow = None
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve documents for UI compatibility."""
        if not self.transform_tools:
            return []
        return self.transform_tools._retrieve_documents(query, k)
    
    def load_documents(self, documents: List[str]):
        """Process and load documents into the vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.create_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name="query_transform_collection",
            persist_directory=self.persist_dir
        )
        
        self.transform_tools = QueryTransformTools(self.vectorstore, self.llm)
        
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the Query Transformation RAG system."""
        
        def transform_node(state: QueryTransformState) -> QueryTransformState:
            try:
                transformed = self.transform_tools._transform_query(state["query"])
                state["transformed_queries"] = [state["query"]] + transformed
            except Exception as e:
                logger.error(f"Error in transform_node: {str(e)}")
                state["transformed_queries"] = [state["query"]]
            return state
        
        def retrieval_node(state: QueryTransformState) -> QueryTransformState:
            try:
                retrieval_results = self.transform_tools._multi_retrieve(
                    state["transformed_queries"]
                )
                state.update(retrieval_results)
            except Exception as e:
                logger.error(f"Error in retrieval_node: {str(e)}")
                state.update({
                    "all_retrieved_docs": [],
                    "combined_context": ""
                })
            return state
        
        def response_node(state: QueryTransformState) -> QueryTransformState:
            try:
                if state.get("combined_context"):
                    response = self.transform_tools._generate_response(
                        query=state["query"],
                        context=state["combined_context"]
                    )
                    state["final_answer"] = str(response)
                else:
                    state["final_answer"] = "No relevant documents found."
            except Exception as e:
                logger.error(f"Error in response_node: {str(e)}")
                state["final_answer"] = f"Error generating response: {str(e)}"
            return state
        
        # Create the graph
        workflow = StateGraph(QueryTransformState)
        
        # Add nodes
        workflow.add_node("transform", transform_node)
        workflow.add_node("retrieval", retrieval_node)
        workflow.add_node("response", response_node)
        
        # Add edges
        workflow.add_edge("transform", "retrieval")
        workflow.add_edge("retrieval", "response")
        workflow.add_edge("response", END)
        
        # Set entry point
        workflow.set_entry_point("transform")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(self, query: str, max_iterations: int = 1, **kwargs) -> Dict[str, Any]:
        """Run the query transformation RAG process."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        initial_state = {
            "messages": [],
            "query": query,
            "transformed_queries": None,
            "all_retrieved_docs": None,
            "combined_context": None,
            "final_answer": None
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        # Format lists as strings for UI display
        transformed_queries_str = "\n".join(final_state["transformed_queries"]) if final_state.get("transformed_queries") else ""
        retrieved_docs_str = "\n---\n".join(final_state["all_retrieved_docs"]) if final_state.get("all_retrieved_docs") else ""
        
        return {
            "original_query": query,
            "transformed_queries": transformed_queries_str,
            "retrieved_docs": retrieved_docs_str,
            "final_answer": final_state["final_answer"],
            "iterations": 1  # Always 1 for query transformation
        }