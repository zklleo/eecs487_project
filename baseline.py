import re
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from langchain.docstore.document import Document
from embedding import OllamaEmbeddings

class BaselineRetrieval:
    """
    Baseline retrieval methods for HotpotQA.
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize the retrieval system.
        
        Args:
            embedding_model: The embedding model to use
        """
        self.embedding_model = embedding_model or OllamaEmbeddings()
        
    def cosine_similarity_retrieval(self, query: str, documents: List[Document],
                                    top_k: int = 2) -> List[Document]:
        """
        Retrieve documents using cosine similarity.
        
        Args:
            query: The query string
            documents: List of documents to search through
            top_k: Number of documents to retrieve
            
        Returns:
            List of top_k most similar documents
        """
        print(f"Retrieving {top_k} documents for query: {query}")
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Get document embeddings
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = self.embedding_model.embed_documents(doc_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            doc_embeddings
        )[0]
        
        # Get top-k document indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top-k documents
        return [documents[idx] for idx in top_indices]
    
    def mmr_retrieval(self, query: str, documents: List[Document], top_k: int = 2,
                      lambda_param: float = 0.5) -> List[Document]:
        """
        Retrieve documents using Maximal Marginal Relevance (MMR).
        
        Args:
            query: The query string
            documents: List of documents to search through
            top_k: Number of documents to retrieve
            lambda_param: Trade-off between relevance and diversity (0-1)
            
        Returns:
            List of top_k documents selected by MMR
        """
        print(f"Using MMR to retrieve {top_k} documents for query: {query}")
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Get document embeddings
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = self.embedding_model.embed_documents(doc_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            doc_embeddings
        )[0]
        
        # MMR implementation
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        for _ in range(min(top_k, len(documents))):
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
                    mmr_score = lambda_param * sim_query - (1 - lambda_param) * max_sim_doc
                    mmr_scores.append(mmr_score)
                
                # Get best MMR score
                best_idx = remaining_indices[np.argmax(mmr_scores)]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected documents
        return [documents[idx] for idx in selected_indices]
    
    def hybrid_retrieval(self, query: str, documents: List[Document], top_k: int = 2,
                         lambda_param: float = 0.7) -> List[Document]:
        """
        Retrieve documents using a hybrid approach combining dense vector similarity,
        MMR for diversity, and keyword-based matching for query terms.
        
        Args:
            query: The query string.
            documents: List of documents to search through.
            top_k: Number of documents to retrieve.
            lambda_param: Trade-off between relevance and diversity (0-1) for MMR.
            
        Returns:
            List of top_k documents selected by the hybrid retrieval method.
        """
        print(f"Using Hybrid retrieval to retrieve {top_k} documents for query: {query}")
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Get document embeddings
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = self.embedding_model.embed_documents(doc_texts)
        
        # Calculate embedding similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Calculate keyword-based scores (lexical overlap with query terms)
        query_terms = set(re.findall(r"\w+", query.lower()))
        lexical_scores = []
        for doc in documents:
            doc_terms = set(re.findall(r"\w+", doc.page_content.lower()))
            match_count = len(query_terms.intersection(doc_terms))
            lexical_fraction = match_count / len(query_terms) if query_terms else 0
            lexical_scores.append(lexical_fraction)
        lexical_scores = np.array(lexical_scores)
        
        # Combine dense similarity with lexical overlap (simple sum fusion)
        combined_scores = similarities + lexical_scores
        
        # Hybrid MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        for _ in range(min(top_k, len(documents))):
            if not remaining_indices:
                break
            
            if not selected_indices:
                # Select the document with highest combined score first
                best_idx = remaining_indices[np.argmax([combined_scores[idx] for idx in remaining_indices])]
            else:
                # Calculate MMR score for remaining documents using combined query similarity
                mmr_scores = []
                selected_embeddings = [doc_embeddings[idx] for idx in selected_indices]
                
                for idx in remaining_indices:
                    sim_query = combined_scores[idx]
                    sim_docs = cosine_similarity([doc_embeddings[idx]], selected_embeddings)[0]
                    max_sim_doc = max(sim_docs) if sim_docs.size > 0 else 0
                    # MMR formula: balance combined relevance vs. similarity to already selected docs
                    mmr_score = lambda_param * sim_query - (1 - lambda_param) * max_sim_doc
                    mmr_scores.append(mmr_score)
                
                best_idx = remaining_indices[np.argmax(mmr_scores)]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return top-k selected documents
        return [documents[idx] for idx in selected_indices]
