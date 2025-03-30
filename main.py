from dataloader import HotpotQALoader
from baseline import BaselineRetrieval
from hotpotqa_evaluation import run_and_evaluate_methods, save_evaluation_results
import time
import json
import os

def main():
    # Load dataset
    print("Loading HotpotQA dataset...")
    loader = HotpotQALoader()
    examples = loader.get_examples_with_contexts('dev', max_samples=50)  # You can adjust sample size
    print(f"Loaded {len(examples)} examples")
    
    # Initialize baseline retrieval method
    print("Initializing baseline retrieval methods...")
    baseline = BaselineRetrieval()
    
    # Define only the baseline methods to evaluate
    methods = {
        "cosine_similarity": baseline.cosine_similarity_retrieval,
        "mmr": baseline.mmr_retrieval
    }
    
    # Run and evaluate
    print("\nStarting evaluation...")
    start_time = time.time()
    retrieval_results, evaluation_results = run_and_evaluate_methods(examples, methods)
    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    
    # Save detailed retrieval results
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_retrieval_details.json", "w") as f:
        # Convert sets to lists for JSON serialization
        serializable_results = {}
        for method, results in retrieval_results.items():
            serializable_results[method] = {}
            for qid, titles in results.items():
                serializable_results[method][qid] = list(titles)
        json.dump(serializable_results, f, indent=2)
    
    # Save evaluation results
    save_evaluation_results(evaluation_results, "results/baseline_evaluation.json")
    
    # Print summary
    print("\n===== Evaluation Summary =====")
    
    for method, metrics in evaluation_results.items():
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  Supporting Facts EM: {metrics['sp_em']:.4f}")
        print(f"  Supporting Facts F1: {metrics['sp_f1']:.4f}")
        print(f"  Supporting Facts Precision: {metrics['sp_precision']:.4f}")
        print(f"  Supporting Facts Recall: {metrics['sp_recall']:.4f}")
    
    # Calculate improvement percentages
    cosine_f1 = evaluation_results.get("cosine_similarity", {}).get("sp_f1", 0)
    mmr_f1 = evaluation_results.get("mmr", {}).get("sp_f1", 0)
    
    if cosine_f1 > 0:
        mmr_improvement = ((mmr_f1 / cosine_f1) - 1) * 100
        print(f"\nImprovements in F1 Score:")
        print(f"  MMR vs Cosine: {mmr_improvement:.2f}%")

if __name__ == "__main__":
    main()