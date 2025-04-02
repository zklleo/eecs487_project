"""
Query Decomposition Retrieval for Multi-Hop Question Answering.

This module implements a query decomposition approach inspired by EfficientRAG,
allowing for efficient retrieval without multiple LLM calls.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.docstore.document import Document
from embedding import OllamaEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryDecompositionRetriever:
    """
    Query Decomposition Retrieval for multi-hop questions.
    
    Implements an efficient retrieval strategy based on query decomposition
    and iterative retrieval for complex questions.
    """
    
    def __init__(self, 
                embedding_model=None,
                max_iterations=3,
                top_k_per_iteration=5,
                lambda_param=0.5,
                retrieval_func=None):
        """
        Initialize the QDR system.
        
        Args:
            embedding_model: The embedding model to use for retrieval
            max_iterations: Maximum number of iterations for retrieval
            top_k_per_iteration: Number of documents to retrieve per iteration
            lambda_param: Trade-off between relevance and diversity (0-1) for MMR
            retrieval_func: Optional custom retrieval function to use instead of MMR
        """
        self.embedding_model = embedding_model or OllamaEmbeddings(model_name="qwen2.5:14b")
        self.max_iterations = max_iterations
        self.top_k_per_iteration = top_k_per_iteration
        self.lambda_param = lambda_param
        self.retrieval_func = retrieval_func
        
        logger.info(f"Initialized QueryDecompositionRetriever with {max_iterations} max iterations")
    
    def retrieve(self, query: str, documents: List[Document], top_k: int = 2) -> List[Document]:
        """
        Retrieve documents for a multi-hop query using query decomposition.
        
        Args:
            query: The complex query string
            documents: List of documents to search through
            top_k: Total number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Starting QDR retrieval for query: {query}")
        
        candidate_pool = set()  # Track documents already seen
        retrieved_documents = []
        
        # Initial query
        current_query = query
        iterations = 0
        
        while iterations < self.max_iterations:
            logger.info(f"Iteration {iterations + 1}, Current query: {current_query}")
            
            # Retrieve documents for current query
            current_docs = self._retrieve_for_current_query(current_query, documents, self.top_k_per_iteration)
            
            # Process retrieved documents
            for doc in current_docs:
                doc_id = doc.metadata.get('document_id', '') or doc.metadata.get('title', '')
                
                # Skip if already processed
                if doc_id in candidate_pool:
                    continue
                    
                # Tag document
                tag = self._tag_document(current_query, doc.page_content)
                
                if tag == "continue":
                    # This document is useful for answering
                    retrieved_documents.append(doc)
                    candidate_pool.add(doc_id)
                    
                    # Label tokens and generate next query
                    labeled_tokens = self._label_tokens(current_query, doc.page_content)
                    if labeled_tokens:
                        current_query = self._generate_next_query(current_query, labeled_tokens)
                else:
                    # This document is not helpful, terminate this branch
                    logger.debug(f"Terminating branch for document: {doc_id[:30]}...")
            
            # Stop if we've collected enough documents
            if len(retrieved_documents) >= top_k:
                retrieved_documents = retrieved_documents[:top_k]
                break
                
            iterations += 1
            
            # If no progress was made in this iteration
            if not current_query or current_query == query:
                break
                
        logger.info(f"QDR retrieval completed after {iterations + 1} iterations, found {len(retrieved_documents)} documents")
        
        # If we have a custom retrieval function, use it for final selection from candidate documents
        if self.retrieval_func and len(retrieved_documents) > top_k:
            logger.info(f"Using custom retrieval function for final selection from {len(retrieved_documents)} candidates")
            return self.retrieval_func(query, retrieved_documents, top_k=top_k)
        
        return retrieved_documents[:top_k]
    
    def _retrieve_for_current_query(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """Retrieve documents for the current query using MMR or custom retrieval function."""
        # If we have a custom retrieval function, use it
        if self.retrieval_func:
            return self.retrieval_func(query, documents, top_k=top_k)
        
        # Otherwise use default MMR implementation
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Get document embeddings
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = self.embedding_model.embed_documents(doc_texts)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # MMR selection
        selected_indices = self._mmr_selection(similarities, doc_embeddings, top_k)
        
        # Return selected documents
        return [documents[idx] for idx in selected_indices]
    
    def _mmr_selection(self, similarities: np.ndarray, doc_embeddings: List[List[float]], top_k: int) -> List[int]:
        """
        Perform Maximum Marginal Relevance selection.
        
        Args:
            similarities: Similarity scores between query and documents
            doc_embeddings: Document embeddings
            top_k: Number of documents to select
            
        Returns:
            List of selected document indices
        """
        # Implementation of MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(similarities)))
        
        for _ in range(min(top_k, len(similarities))):
            if not remaining_indices:
                break
                
            # If this is the first document, choose the most similar
            if not selected_indices:
                best_idx = remaining_indices[np.argmax([similarities[idx] for idx in remaining_indices])]
            else:
                # Calculate MMR for each remaining document
                mmr_scores = []
                selected_embeddings = [doc_embeddings[idx] for idx in selected_indices]
                
                for idx in remaining_indices:
                    # Calculate similarity to query
                    sim_query = similarities[idx]
                    
                    # Calculate maximum similarity to already selected documents
                    sim_docs = cosine_similarity([doc_embeddings[idx]], selected_embeddings)[0]
                    max_sim_doc = max(sim_docs) if sim_docs.size > 0 else 0
                    
                    # Calculate MMR score
                    mmr_score = self.lambda_param * sim_query - (1 - self.lambda_param) * max_sim_doc
                    mmr_scores.append(mmr_score)
                
                # Get best MMR score
                best_idx = remaining_indices[np.argmax(mmr_scores)]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _tag_document(self, query: str, document: str) -> str:
        """
        Tag the document as either 'continue' or 'terminate'.
        
        Uses a heuristic approach to determine if a document contains
        useful information for answering the query.
        
        Args:
            query: The current query
            document: The document content
            
        Returns:
            'continue' or 'terminate'
        """
        # Simple heuristic based on term overlap
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        # Check overlap between query and document
        overlap = len(query_terms.intersection(doc_terms))
        
        # Check for potential named entities in document (capitalized words)
        # that might be useful for generating follow-up queries
        potential_entities = set()
        for word in doc_terms:
            if len(word) > 1 and word[0].isupper():
                potential_entities.add(word)
        
        # Calculate a simple relevance score
        relevance_score = overlap + 0.5 * len(potential_entities)
        
        # Use a threshold for tagging
        return "continue" if relevance_score >= 2 else "terminate"
    
    def _label_tokens(self, query: str, document: str) -> str:
        """
        Label important tokens in the document.
        
        Uses a heuristic approach to extract key information from the document
        that might be useful for generating follow-up queries.
        
        Args:
            query: The current query
            document: The document content
            
        Returns:
            String of important tokens
        """
        # Extract sentences relevant to the query
        query_terms = query.lower().split()
        doc_sentences = document.split('.')
        
        extracted_info = []
        
        for sentence in doc_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence is relevant to query
            words = sentence.split()
            if any(term in sentence.lower() for term in query_terms):
                # Extract named entities (capitalized terms) and surrounding context
                for i, word in enumerate(words):
                    if word and len(word) > 1 and word[0].isupper() and word.lower() not in query_terms:
                        # Get surrounding context
                        start = max(0, i-2)
                        end = min(len(words), i+3)
                        context = ' '.join(words[start:end])
                        if context not in extracted_info:
                            extracted_info.append(context)
        
        # Join the extracted information
        return ' '.join(extracted_info)
    
    def _generate_next_query(self, original_query: str, labeled_tokens: str) -> str:
        """
        Generate the next query based on labeled tokens.
        
        Uses a heuristic approach to create a follow-up query using
        information extracted from retrieved documents.
        
        Args:
            original_query: The original query
            labeled_tokens: Important tokens extracted from documents
            
        Returns:
            The next query
        """
        if not labeled_tokens:
            return original_query
            
        # Extract potential entities from labeled tokens
        original_query_words = original_query.lower().split()
        labeled_words = labeled_tokens.split()
        
        # Find entities mentioned in labeled tokens but not in the original query
        new_entities = []
        for word in labeled_words:
            if len(word) > 1 and word[0].isupper() and word.lower() not in original_query_words:
                new_entities.append(word)
        
        # If we found new entities, create a new query
        if new_entities:
            entity_str = " ".join(new_entities)
            
            # Create follow-up query based on entity type
            if 'who' in original_query.lower() or 'person' in original_query.lower():
                next_query = f"Who is {entity_str}?"
            elif 'when' in original_query.lower() or 'date' in original_query.lower():
                next_query = f"When was {entity_str}?"
            elif 'where' in original_query.lower() or 'location' in original_query.lower():
                next_query = f"Where is {entity_str}?"
            else:
                next_query = f"What about {entity_str}?"
                
            return next_query
            
        # If no new entities were found, maintain the original query
        return original_query


class QDRWithHybrid(QueryDecompositionRetriever):
    """Query Decomposition Retrieval with Hybrid retrieval for final document selection."""
    
    def __init__(self, 
                embedding_model=None,
                hybrid_retriever=None,
                max_iterations=3,
                top_k_per_iteration=5,
                lambda_param=0.5):
        """
        Initialize QDR with Hybrid retrieval.
        
        Args:
            embedding_model: The embedding model to use for retrieval
            hybrid_retriever: The hybrid retriever instance
            max_iterations: Maximum number of iterations for retrieval
            top_k_per_iteration: Number of documents to retrieve per iteration
            lambda_param: Trade-off between relevance and diversity (0-1) for MMR
        """
        # If no hybrid_retriever is provided, we'll need the baseline retriever in eval_qdr.py
        self.hybrid_retriever = hybrid_retriever
        
        super().__init__(
            embedding_model=embedding_model,
            max_iterations=max_iterations,
            top_k_per_iteration=top_k_per_iteration,
            lambda_param=lambda_param,
            retrieval_func=None  # We'll handle hybrid retrieval separately
        )
        
        logger.info("Initialized QDR with Hybrid retrieval")
    
    def retrieve(self, query: str, documents: List[Document], top_k: int = 2) -> List[Document]:
        """
        Retrieve documents for a multi-hop query using query decomposition and hybrid selection.
        
        Args:
            query: The complex query string
            documents: List of documents to search through
            top_k: Total number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Starting QDR with Hybrid retrieval for query: {query}")
        
        # First, use QDR to collect candidate documents
        candidate_pool = set()  # Track documents already seen
        candidate_documents = []
        
        # Initial query
        current_query = query
        iterations = 0
        
        while iterations < self.max_iterations:
            logger.info(f"Iteration {iterations + 1}, Current query: {current_query}")
            
            # Retrieve documents for current query using MMR
            current_docs = self._retrieve_for_current_query(current_query, documents, self.top_k_per_iteration)
            
            # Process retrieved documents
            for doc in current_docs:
                doc_id = doc.metadata.get('document_id', '') or doc.metadata.get('title', '')
                
                # Skip if already processed
                if doc_id in candidate_pool:
                    continue
                    
                # Tag document
                tag = self._tag_document(current_query, doc.page_content)
                
                if tag == "continue":
                    # This document is useful for answering
                    candidate_documents.append(doc)
                    candidate_pool.add(doc_id)
                    
                    # Label tokens and generate next query
                    labeled_tokens = self._label_tokens(current_query, doc.page_content)
                    if labeled_tokens:
                        current_query = self._generate_next_query(current_query, labeled_tokens)
                else:
                    # This document is not helpful, terminate this branch
                    logger.debug(f"Terminating branch for document: {doc_id[:30]}...")
            
            iterations += 1
            
            # If no progress was made in this iteration or we have enough candidates
            if not current_query or current_query == query or len(candidate_documents) >= top_k * 3:
                break
                
        logger.info(f"QDR phase completed after {iterations} iterations, found {len(candidate_documents)} candidate documents")
        
        # Now use hybrid retrieval for final selection if we have candidates and a hybrid retriever
        if candidate_documents and len(candidate_documents) > top_k and self.hybrid_retriever:
            logger.info(f"Using hybrid retrieval for final selection from {len(candidate_documents)} candidates")
            final_docs = self.hybrid_retriever.hybrid_retrieval(query, candidate_documents, top_k=top_k)
            return final_docs
        
        # If we don't have enough candidates or no hybrid retriever, return candidates directly (truncated if needed)
        return candidate_documents[:top_k]