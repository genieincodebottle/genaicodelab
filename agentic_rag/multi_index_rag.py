import os
import logging
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union
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

class MultiIndexState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    source_docs: Dict[str, List[str]]
    combined_context: Optional[str]
    source_explanation: Optional[str]
    final_answer: Optional[str]

class MultiIndexTools:
    def __init__(self, vector_stores: Dict[str, Any], llm: Any):
        self.vector_stores = vector_stores
        self.llm = llm
        self.tools = self._create_tools()
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve documents for UI compatibility."""
        try:
            all_docs = []
            for source_name, vector_store in self.vector_stores.items():
                docs = vector_store.similarity_search(query, k=k)
                source_docs = [f"Source {source_name}: {str(doc.page_content)}" for doc in docs]
                all_docs.extend(source_docs)
            return all_docs[:k]  # Return top k docs across all sources
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            return []

    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="multi_retrieval",
                description="Retrieves documents from multiple vector stores",
                func=self._multi_retrieval
            ),
            Tool(
                name="generate_explanation",
                description="Generates explanation of source usage",
                func=self._generate_explanation
            ),
            Tool(
                name="generate_response",
                description="Generates final response based on combined context",
                func=self._generate_response
            )
        ]
    
    def _multi_retrieval(self, query: str, k: int = 3) -> Dict[str, List[str]]:
        """Retrieve documents from multiple vector stores."""
        try:
            source_docs = {}
            for source_name, vector_store in self.vector_stores.items():
                docs = vector_store.similarity_search(query, k=k)
                source_docs[source_name] = [str(doc.page_content) for doc in docs]
            return source_docs
        except Exception as e:
            logger.error(f"Error in multi retrieval: {str(e)}")
            return {}
    
    def _generate_explanation(self, query: str, context: str, answer: str) -> str:
        """Generate explanation of how different sources contributed."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "Based on the following context and answer, explain how information "
                "from different sources was used to answer the query. Identify which "
                "aspects of the answer came from which sources.\n\n"
                "Context: {context}\n"
                "Query: {query}\n"
                "Answer: {answer}\n\n"
                "Explanation:"
            )
            chain = prompt | self.llm
            explanation = chain.invoke({
                "context": context,
                "query": query,
                "answer": answer
            })
            return str(explanation.content) if hasattr(explanation, 'content') else str(explanation)
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Unable to generate source explanation due to an error."
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate final response based on combined context."""
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an AI assistant tasked with answering questions based on the "
                "provided context. The context contains information from multiple sources. "
                "Please analyze the context carefully and provide a comprehensive answer.\n\n"
                "Context: {context}\n"
                "Query: {query}\n"
                "Answer:"
            )
            chain = prompt | self.llm
            response = chain.invoke({"context": context, "query": query})
            return str(response.content) if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

class MultiIndexRAG:
    def __init__(self, model_provider: str, model_name: str = None):
        """Initialize the Multi-index RAG system."""
        llm_manager = LLMManager()
        self.llm = llm_manager.initialize_llm(model_provider, model_name)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'trust_remote_code': True}
        )
        
        self.persist_dir = os.path.join(tempfile.gettempdir(), "chroma_multi_index_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        self.vector_stores = {}
        self.multi_tools = None
        self.workflow = None
    
    def _retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve documents for UI compatibility."""
        try:
            all_docs = []
            for source_name, vector_store in self.vector_stores.items():
                docs = vector_store.similarity_search(query, k=k)
                all_docs.extend([f"{source_name}: {str(doc.page_content)}" for doc in docs])
            return all_docs[:k]  # Return top k docs across all sources
        except Exception as e:
            logger.error(f"Error in retrieve_documents: {str(e)}")
            return []
    
    def load_documents(self, documents: Union[List[str], Dict[str, List[str]]]):
        """Process and load documents into multiple vector stores.
        
        Args:
            documents: Either a list of documents or a dictionary mapping source names to lists of documents
        """
        # If input is a list, convert to dictionary with automatic source naming
        if isinstance(documents, list):
            # Split documents into smaller groups (e.g., 5 docs per source)
            docs_per_source = 5
            doc_groups = [documents[i:i + docs_per_source] for i in range(0, len(documents), docs_per_source)]
            documents = {f"source_{i+1}": group for i, group in enumerate(doc_groups)}
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Clear existing vector stores
        self.vector_stores.clear()
        
        # Create vector stores for each source
        for source_name, docs in documents.items():
            chunks = text_splitter.create_documents(docs)
            
            self.vector_stores[source_name] = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=f"multi_index_{source_name}",
                persist_directory=os.path.join(self.persist_dir, source_name)
            )
        
        self.multi_tools = MultiIndexTools(self.vector_stores, self.llm)
        
        self._create_workflow()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for the Multi-index RAG system."""
        
        def retrieval_node(state: MultiIndexState) -> MultiIndexState:
            try:
                source_docs = self.multi_tools._multi_retrieval(state["query"])
                state["source_docs"] = source_docs
                
                # Combine all documents with source labels
                all_docs = []
                for source_name, docs in source_docs.items():
                    all_docs.extend([f"Source {source_name}: {doc}" for doc in docs])
                state["combined_context"] = "\n\n".join(all_docs)
            except Exception as e:
                logger.error(f"Error in retrieval_node: {str(e)}")
                state["source_docs"] = {}
                state["combined_context"] = ""
            return state
        
        def response_node(state: MultiIndexState) -> MultiIndexState:
            try:
                if state.get("combined_context"):
                    response = self.multi_tools._generate_response(
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
        
        def explanation_node(state: MultiIndexState) -> MultiIndexState:
            try:
                if state.get("final_answer") and state.get("combined_context"):
                    explanation = self.multi_tools._generate_explanation(
                        query=state["query"],
                        context=state["combined_context"],
                        answer=state["final_answer"]
                    )
                    state["source_explanation"] = str(explanation)
                else:
                    state["source_explanation"] = "No explanation available."
            except Exception as e:
                logger.error(f"Error in explanation_node: {str(e)}")
                state["source_explanation"] = f"Error generating explanation: {str(e)}"
            return state
        
        # Create the graph
        workflow = StateGraph(MultiIndexState)
        
        # Add nodes
        workflow.add_node("retrieval", retrieval_node)
        workflow.add_node("response", response_node)
        workflow.add_node("explanation", explanation_node)
        
        # Add edges
        workflow.add_edge("retrieval", "response")
        workflow.add_edge("response", "explanation")
        workflow.add_edge("explanation", END)
        
        # Set entry point
        workflow.set_entry_point("retrieval")
        
        # Compile the graph
        self.workflow = workflow.compile()
    
    def run(self, query: str, max_iterations: int = 1, **kwargs) -> Dict[str, Any]:
        """Run the multi-index RAG process."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Please load documents first.")
        
        initial_state = {
            "messages": [],
            "query": query,
            "source_docs": {},
            "combined_context": None,
            "source_explanation": None,
            "final_answer": None
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "source_docs": final_state["source_docs"],
            "source_explanation": final_state["source_explanation"],
            "final_answer": final_state["final_answer"],
            "iterations": 1  # Always 1 for multi-index search
        }