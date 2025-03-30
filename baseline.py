import os
import json
import time
from typing import List, Dict, Any, Set, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.docstore.document import Document
from dataloader import HotpotQALoader
from embedding import OllamaEmbeddings
from generation import OllamaGeneration

# Import the official evaluation functions
from evaluation import run_and_evaluate_methods, save_evaluation_results


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
        
    def cosine_similarity_retrieval(self, query: str, documents: List[Document], top_k: int = 2) -> List[Document]:
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

def run_official_evaluation(examples, max_examples=None):
    """
    Run evaluation with official HotpotQA metrics.
    
    Args:
        examples: List of (question_info, documents) tuples
        max_examples: Maximum number of examples to evaluate
        
    Returns:
        Official evaluation results
    """
    if max_examples:
        examples = examples[:max_examples]
    
    # Initialize retriever
    retriever = BaselineRetrieval()
    
    # Define methods to evaluate
    methods = {
        "cosine_similarity": retriever.cosine_similarity_retrieval,
        "mmr": retriever.mmr_retrieval
    }
    
    # Run and evaluate with official metrics
    print("\n==== Evaluating with Official HotpotQA Metrics ====")
    retrieval_results, official_results = run_and_evaluate_methods(examples, methods)
    
    # Print official results
    print("\n----- Official HotpotQA Metrics Results -----")
    
    for method, metrics in official_results.items():
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  Supporting Facts EM: {metrics['sp_em']:.4f}")
        print(f"  Supporting Facts F1: {metrics['sp_f1']:.4f}")
        print(f"  Supporting Facts Precision: {metrics['sp_precision']:.4f}")
        print(f"  Supporting Facts Recall: {metrics['sp_recall']:.4f}")
    
    # Calculate improvement percentages
    cosine_f1 = official_results.get("cosine_similarity", {}).get("sp_f1", 0)
    mmr_f1 = official_results.get("mmr", {}).get("sp_f1", 0)
    
    if cosine_f1 > 0:
        mmr_improvement = ((mmr_f1 / cosine_f1) - 1) * 100
        print(f"\nImprovements in F1 Score:")
        print(f"  MMR vs Cosine: {mmr_improvement:.2f}%")
    
    return official_results

def main():
    # Load HotpotQA dataset with the correct structure
    loader = HotpotQALoader()
    examples = loader.get_examples_with_contexts('dev', max_samples=50)  # You can adjust sample size
    
    print(f"Loaded {len(examples)} examples")
    
    # Run evaluation
    start_time = time.time()
    official_results = run_official_evaluation(examples)
    end_time = time.time()
    
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_evaluation_results(official_results, 'results/baseline_official_results.json')
    
    print("\nResults saved to results/baseline_official_results.json")

if __name__ == "__main__":
    main()