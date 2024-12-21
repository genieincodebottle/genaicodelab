import os
import logging
from typing import Any, Dict, List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import tempfile
from utils.llm_manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

class RAGTools:
    def __init__(self, vectorstore: Any, llm: Any, model_name: Any):
        self.vectorstore = vectorstore
        self.llm = llm
        self.model_name = model_name
        
    def _query_reformulation(self, query: str, context: str) -> str:
        """Creates a more targeted query based on current context."""
        prompt = f"""Given the original query and the current context, please reformulate the query to be more specific and targeted:
        
        Original query: {query}
        Current context: {context}
        
        Reformulated query:"""
        
        response = self.llm.invoke(prompt)
        if self.model_name == "llama2:latest" or self.model_name == "mistral":
            return response
        return response.content

    def _retrieve_documents(self, query: str) -> List[str]:
        """Retrieves relevant documents for the query."""
        docs = self.vectorstore.similarity_search(query, k=4)
        return [doc.page_content for doc in docs]

    def _generate_answer(self, query: str, context: str) -> str:
        """Generates an answer based on query and context."""
        prompt = f"""Please answer the following query based on the given context:
        
        Query: {query}
        Context: {context}
        
        Answer:"""
        
        response = self.llm.invoke(prompt)
        if self.model_name == "llama2:latest" or self.model_name == "mistral":
            return response
        return response.content

class AdaptiveRAG:
    def __init__(self, model_provider: str, model_name: Optional[str] = None):
        """Initialize the Adaptive RAG system."""
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
            collection_name="adaptive_rag_collection",
            persist_directory=self.persist_dir
        )
        
        # Initialize RAG tools
        self.rag_tools = RAGTools(self.vectorstore, self.llm, self.model_name)

    def _is_answer_satisfactory(self, answer: str) -> bool:
        """Check if the generated answer is satisfactory."""
        # Simple length-based heuristic - can be enhanced
        return len(answer) > 100

    def run(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Run the adaptive RAG process with multiple iterations if needed."""
        if not self.vectorstore or not self.rag_tools:
            raise ValueError("System not initialized. Please load documents first.")

        current_query = query
        context = ""
        iteration = 0
        final_answer = None

        try:
            while iteration < max_iterations:
                # Reformulate query after first iteration
                if iteration > 0:
                    current_query = self.rag_tools._query_reformulation(
                        query=query,
                        context=context
                    )

                # Retrieve relevant documents
                docs = self.rag_tools._retrieve_documents(current_query)
                context = " ".join(docs)

                # Generate answer
                answer = self.rag_tools._generate_answer(
                    query=current_query,
                    context=context
                )

                final_answer = answer
                iteration += 1

                # Check if answer is satisfactory
                if self._is_answer_satisfactory(answer):
                    break

            return {
                "final_answer": final_answer,
                "iterations": iteration,
                "final_query": current_query
            }

        except Exception as e:
            logger.error(f"Error in adaptive RAG process: {str(e)}")
            return {
                "final_answer": "Error generating response",
                "iterations": iteration,
                "final_query": current_query
            }