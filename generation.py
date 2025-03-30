import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class OllamaGeneration:
    """Class for generating text using Ollama models."""
    
    def __init__(self, model_name="qwen2.5:7b", base_url="http://localhost:11434"):
        """
        Initialize the Ollama generation client.
        
        Args:
            model_name: The name of the Ollama model to use
            base_url: The base URL for the Ollama API server
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        
        # Verify connectivity
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": "Hello", "stream": False}
            )
            if response.status_code != 200:
                logger.warning(f"Ollama generation test failed with status {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server at {self.base_url}: {e}")
            logger.warning("Make sure Ollama is running and the model is pulled.")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated text
        """
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data
            )
            
            if response.status_code != 200:
                logger.error(f"Error in text generation: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}"
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return f"Error generating response: {str(e)}"

    def generate_rag_response(self, question: str, context_docs: list) -> str:
        """
        Generate a response based on retrieved documents.
        
        Args:
            question: The user's question
            context_docs: List of retrieved documents
            
        Returns:
            The generated answer
        """
        # Construct context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # System prompt for RAG
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain enough information to answer the question, state that you cannot answer based on the provided information."""
        
        # User prompt combining question and context
        user_prompt = f"""Question: {question}

Context:
{context}

Based on the provided context, please answer the question concisely and accurately."""
        
        # Generate response
        return self.generate(user_prompt, system_prompt=system_prompt, temperature=0.3)