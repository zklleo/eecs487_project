from __future__ import annotations
import logging
import traceback
from typing import List
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class OllamaEmbeddings():
    """Ollama embeddings with fallback to SentenceTransformers."""
    
    def __init__(self, model_name="qwen2.5:14b", base_url="http://localhost:11434"):
        """
        Initialize the Ollama embeddings client with a fallback to SentenceTransformers.
        
        Args:
            model_name: The name of the Ollama model to use
            base_url: The base URL for the Ollama API server
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.use_fallback = False
        
        # Initialize fallback model
        try:
            self.fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized SentenceTransformer fallback model")
        except Exception as e:
            logger.warning(f"Couldn't initialize fallback model: {e}")
            self.fallback_model = None
        
        # Verify connectivity to Ollama
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama API test failed with status {response.status_code}: {response.text}")
                self.use_fallback = True
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server at {self.base_url}: {e}")
            logger.warning("Make sure Ollama is running and the model is pulled.")
            self.use_fallback = True
    
    def embed_query_with_ollama(self, text: str) -> List[float]:
        """
        Generate embeddings using Ollama's API.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            A list of float values representing the embedding
        """
        try:
            # Try the embeddings endpoint first (newer Ollama versions)
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                if embedding:
                    return embedding

            # If embeddings endpoint failed, try the generate endpoint with embedding option
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": text,
                    "options": {
                        "embedding": True
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                if embedding:
                    return embedding
                
            # If we got here, both methods failed
            logger.error(f"Failed to get embeddings from Ollama: {response.status_code} - {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"Error in embed_query_with_ollama: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def embed_query_with_fallback(self, text: str) -> List[float]:
        """
        Generate embeddings using the fallback SentenceTransformer model.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            A list of float values representing the embedding
        """
        if not self.fallback_model:
            # Return a random embedding as last resort
            logger.warning("Using random embeddings as fallback")
            return np.random.randn(384).tolist()  # 384-dim is standard for MiniLM
        
        try:
            embedding = self.fallback_model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error in embed_query_with_fallback: {e}")
            # Return a random embedding as last resort
            return np.random.randn(384).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            A list of float values representing the embedding
        """
        if self.use_fallback:
            return self.embed_query_with_fallback(text)
        
        # Try Ollama first
        embedding = self.embed_query_with_ollama(text)
        
        # If Ollama fails, use fallback
        if embedding is None:
            logger.warning("Ollama embedding failed, switching to fallback")
            self.use_fallback = True
            return self.embed_query_with_fallback(text)
            
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings, one for each text
        """
        if self.use_fallback and self.fallback_model:
            # Use batch processing for fallback model
            try:
                return self.fallback_model.encode(texts).tolist()
            except Exception as e:
                logger.error(f"Error in batch embedding with fallback: {e}")
                # Fall back to individual processing
                return [self.embed_query(text) for text in texts]
        else:
            # Process one by one with Ollama
            return [self.embed_query(text) for text in texts]