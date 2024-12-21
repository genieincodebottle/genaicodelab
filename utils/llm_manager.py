import os
import logging
from typing import Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

class LLMManager:
    """Singleton class to manage LLM instances."""
    _instance = None
    _llm_instances = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize_llm(cls, 
                      model_provider: str, 
                      model_name: Optional[str] = None) -> Any:
        """
        Initialize an LLM instance based on the provider.
        Returns existing instance if already initialized with same parameters.
        """
        instance_key = f"{model_provider}_{model_name}"
        
        # Return existing instance if available
        if instance_key in cls._llm_instances:
            return cls._llm_instances[instance_key]
        
        # Initialize new LLM instance
        llm = None
        if model_provider == "gemini":
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
            if not model_name:
                raise ValueError("Model Name is required for Gemini")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=TEMPERATURE,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
        elif model_provider == "openai":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            if not model_name:
                raise ValueError("Model Name is required for OpenAI")
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=TEMPERATURE
            )
        elif model_provider == "claude":
            os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
            if not model_name:
                raise ValueError("Model Name is required for Claude")
            llm = ChatAnthropic(
                model_name=model_name,
                temperature=TEMPERATURE
            )
        elif model_provider == "ollama":
            if not model_name:
                raise ValueError("Model Name is required for Ollama")
            llm = OllamaLLM(
                model=model_name,
                temperature=TEMPERATURE
            )
        elif model_provider == "groq":
            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
            if not model_name:
                raise ValueError("Model Name is required for Groq")
            llm = ChatGroq(
                model=model_name,
                temperature=TEMPERATURE
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Store and return the instance
        cls._llm_instances[instance_key] = llm
        return llm

    @classmethod
    def get_llm(cls, 
                model_provider: str, 
                model_name: Optional[str] = None) -> Any:
        """
        Get an existing LLM instance or raise error if not initialized.
        """
        instance_key = f"{model_provider}_{model_name}"
        if instance_key not in cls._llm_instances:
            raise ValueError(f"LLM instance for {model_provider} with model {model_name} not initialized")
        return cls._llm_instances[instance_key]

    @classmethod
    def clear_instances(cls):
        """Clear all stored LLM instances."""
        cls._llm_instances.clear()