"""
Evaluation script for Query Decomposition Retrieval (QDR) with HotpotQA.

This script evaluates QDR, including a new QDR+Hybrid retrieval method.
"""

import os
import json
import time
from collections import Counter
from typing import Dict, List, Tuple, Any, Set
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Import existing components
from dataloader import HotpotQALoader
from baseline import BaselineRetrieval
from evaluation import (
    evaluate_retrieval_with_official_metrics, 
    save_evaluation_results, 
    run_and_evaluate_methods
)

# Import QDR and the new QDR+Hybrid class from the qdr.py file
from qdr import QueryDecompositionRetriever, QDRWithHybrid

def run_qdr_evaluation(examples, max_examples=None, output_dir="results/qdr_hybrid"):
    """
    Run evaluation with QDR, Hybrid, and QDR+Hybrid methods using official HotpotQA metrics.
    
    Args:
        examples: List of (question_info, documents) tuples
        max_examples: Maximum number of examples to evaluate
        output_dir: Directory to save results
        
    Returns:
        Official evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if max_examples:
        examples = examples[:max_examples]
    
    # Initialize baseline retriever
    baseline_retriever = BaselineRetrieval()
    
    # Initialize QDR retriever with the same embedding model
    qdr_retriever = QueryDecompositionRetriever(
        embedding_model=baseline_retriever.embedding_model,
        max_iterations=3,
        top_k_per_iteration=5,
        lambda_param=0.5
    )
    
    # Initialize QDR+Hybrid retriever
    qdr_hybrid_retriever = QDRWithHybrid(
        embedding_model=baseline_retriever.embedding_model,
        hybrid_retriever=baseline_retriever,
        max_iterations=3,
        top_k_per_iteration=5,
        lambda_param=0.5
    )
    
    # Define methods to evaluate
    methods = {
        "cosine_similarity": baseline_retriever.cosine_similarity_retrieval,
        "mmr": baseline_retriever.mmr_retrieval,
        # "hybrid": baseline_retriever.hybrid_retrieval,
        "qdr": qdr_retriever.retrieve,
        "qdr_hybrid": qdr_hybrid_retriever.retrieve
    }
    
    # Run and evaluate with official metrics
    print("\n==== Evaluating with Official HotpotQA Metrics (Including QDR+Hybrid) ====")
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
    # Base comparison method
    base_method = "cosine_similarity"
    base_f1 = official_results.get(base_method, {}).get("sp_f1", 0)
    
    if base_f1 > 0:
        print(f"\nImprovements in F1 Score (compared to {base_method}):")
        
        for method in ["mmr", "hybrid", "qdr", "qdr_hybrid"]:
            if method in official_results:
                method_f1 = official_results[method].get("sp_f1", 0)
                improvement = ((method_f1 / base_f1) - 1) * 100
                print(f"  {method.replace('_', ' ').title()} vs {base_method.replace('_', ' ').title()}: {improvement:.2f}%")
        
        # Also compare QDR+Hybrid to QDR and Hybrid
        qdr_f1 = official_results.get("qdr", {}).get("sp_f1", 0)
        hybrid_f1 = official_results.get("hybrid", {}).get("sp_f1", 0)
        qdr_hybrid_f1 = official_results.get("qdr_hybrid", {}).get("sp_f1", 0)
        
        if qdr_f1 > 0:
            qdr_hybrid_vs_qdr = ((qdr_hybrid_f1 / qdr_f1) - 1) * 100
            print(f"  QDR+Hybrid vs QDR: {qdr_hybrid_vs_qdr:.2f}%")
            
        if hybrid_f1 > 0:
            qdr_hybrid_vs_hybrid = ((qdr_hybrid_f1 / hybrid_f1) - 1) * 100
            print(f"  QDR+Hybrid vs Hybrid: {qdr_hybrid_vs_hybrid:.2f}%")
    
    # Save results
    results_path = os.path.join(output_dir, "qdr_hybrid_evaluation_results.json")
    save_evaluation_results(official_results, results_path)
    
    # Create comparison charts
    create_comparison_charts(official_results, output_dir)
    
    return official_results

def create_comparison_charts(evaluation_results, output_dir):
    """
    Create charts comparing different retrieval methods.
    
    Args:
        evaluation_results: Dict of evaluation results
        output_dir: Directory to save charts
    """
    # Extract data
    methods = list(evaluation_results.keys())
    f1_scores = [evaluation_results[method]['sp_f1'] * 100 for method in methods]  # Convert to percentage
    em_scores = [evaluation_results[method]['sp_em'] * 100 for method in methods]  # Convert to percentage
    precision = [evaluation_results[method]['sp_precision'] * 100 for method in methods]
    recall = [evaluation_results[method]['sp_recall'] * 100 for method in methods]
    
    # Create performance chart
    plt.figure(figsize=(14, 6))
    x = range(len(methods))
    width = 0.2
    
    plt.bar([i - width*1.5 for i in x], em_scores, width, label='Exact Match')
    plt.bar([i - width*0.5 for i in x], f1_scores, width, label='F1 Score')
    plt.bar([i + width*0.5 for i in x], precision, width, label='Precision')
    plt.bar([i + width*1.5 for i in x], recall, width, label='Recall')
    
    plt.xlabel('Retrieval Method')
    plt.ylabel('Score (%)')
    plt.title('Retrieval Performance Comparison')
    plt.xticks(x, [m.replace('_', ' ').title() for m in methods], rotation=15)
    plt.legend()
    plt.tight_layout()
    
    performance_chart_path = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(performance_chart_path)
    plt.close()
    
    # Create simpler bar chart comparing just F1 scores
    plt.figure(figsize=(12, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'][:len(methods)]
    plt.bar(range(len(methods)), f1_scores, color=colors)
    plt.xlabel('Retrieval Method')
    plt.ylabel('F1 Score (%)')
    plt.title('F1 Score Comparison')
    plt.xticks(range(len(methods)), [m.replace('_', ' ').title() for m in methods], rotation=15)
    
    # Add value labels
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    f1_chart_path = os.path.join(output_dir, "f1_comparison.png")
    plt.savefig(f1_chart_path)
    plt.close()
    
    # Create a new chart specifically comparing QDR, Hybrid, and QDR+Hybrid
    if all(m in evaluation_results for m in ['hybrid', 'qdr', 'qdr_hybrid']):
        plt.figure(figsize=(10, 5))
        
        methods_subset = ['hybrid', 'qdr', 'qdr_hybrid']
        f1_subset = [evaluation_results[m]['sp_f1'] * 100 for m in methods_subset]
        
        plt.bar(range(len(methods_subset)), f1_subset, color=['tab:green', 'tab:red', 'tab:purple'])
        plt.xlabel('Retrieval Method')
        plt.ylabel('F1 Score (%)')
        plt.title('Comparison of Hybrid, QDR, and QDR+Hybrid')
        plt.xticks(range(len(methods_subset)), [m.replace('_', ' ').title() for m in methods_subset])
        
        # Add value labels
        for i, v in enumerate(f1_subset):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        comparison_chart_path = os.path.join(output_dir, "hybrid_qdr_comparison.png")
        plt.savefig(comparison_chart_path)
        plt.close()

def main():
    # Load HotpotQA dataset with the correct structure
    loader = HotpotQALoader()
    examples = loader.get_examples_with_contexts('dev', max_samples=50)  # You can adjust sample size
    
    print(f"Loaded {len(examples)} examples")
    
    # Run evaluation
    start_time = time.time()
    official_results = run_qdr_evaluation(examples)
    end_time = time.time()
    
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    os.makedirs('results/qdr_hybrid', exist_ok=True)
    save_evaluation_results(official_results, 'results/qdr_hybrid/qdr_hybrid_evaluation_results.json')
    
    print("\nResults saved to results/qdr_hybrid/qdr_hybrid_evaluation_results.json")
    print("Charts saved to results/qdr_hybrid/")

if __name__ == "__main__":
    main()