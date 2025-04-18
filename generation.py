import requests
import json
import logging
from typing import Dict, Any, Optional, List
from langchain.docstore.document import Document

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
        
    def generate_response(self, question: str) -> str:
        """
        Generate an answer to a question using the Ollama model.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        # System prompt for RAG
        system_prompt = """You are a helpful assistant that answers questions."""
        user_prompt = f"""Question: {question}
Your answer must be in one of these formats:
1. 'yes' (for affirmative answers)
2. 'no' (for negative answers)
3. Specific text span from the context (for factual information)

All responses should be in lowercase.

Output only the answer, without explanation."""
        
        raw_response = self.generate(user_prompt, system_prompt=system_prompt, temperature=0.1)
        
        # Process and normalize the response
        normalized_response = self._normalize_response(raw_response)
        
        return normalized_response

    def generate_rag_response(self, question: str, context_docs: List[Document]) -> str:
        """
        Generate a response based on retrieved documents.
        
        Args:
            question: The user's question
            context_docs: List of retrieved documents
            
        Returns:
            The generated answer as YES, NO, or a span from the context
        """
        # Construct context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # System prompt for RAG
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
For yes/no questions, respond only with 'yes' or 'no' in lowercase.
If the answer is a specific piece of information from the context, extract just that information.
If the context doesn't contain enough information to answer the question, respond with 'no answer' in lowercase.
Be precise and concise."""
        
        # User prompt combining question and context
        user_prompt = f"""Question: {question}

Context:
{context}

Based solely on the provided context, answer the question.
Your answer must be in one of these formats:
1. 'yes' (for affirmative answers)
2. 'no' (for negative answers)
3. Specific text span from the context (for factual information)
4. 'no answer' (if the context doesn't provide sufficient information)

All responses should be in lowercase.

Output only the answer, without explanation."""
        
        # Generate response with low temperature for factual consistency
        raw_response = self.generate(user_prompt, system_prompt=system_prompt, temperature=0.1)
        
        # Process and normalize the response
        normalized_response = self._normalize_response(raw_response)
        
        return normalized_response
    
    def _normalize_response(self, response: str) -> str:
        """
        Normalize the model response to ensure it follows the expected format.
        
        Args:
            response: The raw model response
            
        Returns:
            Normalized response (YES, NO, or answer span)
        """
        # Clean up the response
        response = response.strip()
        
        # Convert to lowercase for YES/NO responses
        if response.lower() in ['yes', 'no']:
            return response.lower()
        
        # Check for NO ANSWER variations
        no_answer_phrases = ['no answer', 'cannot answer', 'don\'t know', 'insufficient information']
        if any(phrase in response.lower() for phrase in no_answer_phrases):
            return "no answer"
        
        # Otherwise, return the answer span as is
        return response
