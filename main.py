#!/usr/bin/env python
import argparse
import json
import os
import logging
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

from dataloader import HotpotQALoader
from baseline import BaselineRetrieval
from generation import OllamaGeneration
from embedding import OllamaEmbeddings
from efficientRAG_loader import convert_to_documents, load_efficientrag_negsample_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qa_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def normalize_answer(answer: str) -> str:
    """Normalize the answer to yes, no, or the answer span."""
    answer = answer.strip().lower()
    
    # Check for exact yes/no answers (must match exactly)
    if answer == "yes":
        return "yes"
    elif answer == "no":
        return "no"
    # For model output, also check for YES/NO
    elif answer == "YES".lower():
        return "yes"
    elif answer == "NO".lower():
        return "no"
    else:
        # Return the answer span
        return answer

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score
    """
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    
    # Find common tokens
    common = set(prediction_tokens) & set(ground_truth_tokens)
    
    # If no common tokens, F1 = 0
    if not common:
        return 0.0
    
    # Calculate precision, recall, and F1
    precision = len(common) / len(prediction_tokens) if prediction_tokens else 0.0
    recall = len(common) / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate_predictions(predictions: Dict[str, str], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate the predictions against ground truth answers.
    
    Args:
        predictions: Dictionary mapping question IDs to predicted answers
        ground_truth: List of ground truth examples
        
    Returns:
        Dictionary with accuracy metrics
    """
    correct = 0
    total = 0
    f1_scores = []
    
    # Create a mapping from question ID to ground truth answer
    gt_answers = {
        item.get("_id", item.get("id")): item["answer"]
        for item in ground_truth
        if "answer" in item
    }
    
    for qid, pred in predictions.items():
        if qid in gt_answers:
            # Normalize prediction and ground truth
            pred_norm = normalize_answer(pred)
            gt_norm = normalize_answer(gt_answers[qid])
            
            # Calculate F1 score
            f1 = calculate_f1_score(pred_norm, gt_norm)
            f1_scores.append(f1)
            
            # Check for exact match
            exact_match = (pred_norm == gt_norm)
            if exact_match:
                correct += 1
                
            total += 1
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return {
        "exact_match": accuracy,
        "f1_score": avg_f1,
        "correct": correct,
        "total": total
    }

def run_qa_system(args):
    """
    Run the QA system.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Starting QA system with {args.retrieval_method} retrieval method")
    

    # Initialize components
    data_loader = HotpotQALoader(data_dir=args.data_dir)
    embedding_model = OllamaEmbeddings(model_name=args.model_name, base_url=args.ollama_url)
    retriever = BaselineRetrieval(embedding_model=embedding_model)
    generator = OllamaGeneration(model_name=args.model_name, base_url=args.ollama_url)
    
    
    # Load examples
    logger.info(f"Loading {args.split} split data with max {args.max_samples} samples")
    if args.efficientrag_file:
        if "negsample" in args.efficientrag_file.lower():
            examples = load_efficientrag_negsample_dataset(args.efficientrag_file, max_samples=args.max_samples)
    else:
        examples = data_loader.get_examples_with_contexts(split=args.split, max_samples=args.max_samples)

    # Choose retrieval method
    retrieval_methods = {
        "cosine": retriever.cosine_similarity_retrieval,
        "mmr": retriever.mmr_retrieval,
        "hybrid": retriever.hybrid_retrieval,
        "none": None  # Added baseline option
    }

    retrieval_func = retrieval_methods.get(args.retrieval_method)
    
    # Process examples
    predictions = {}
    for i, question_info in enumerate(tqdm(examples, desc="Processing questions")):
        question_id = question_info["id"]
        question = question_info["question"]

        logger.info(f"Processing question {i+1}/{len(examples)}: {question}")

        if retrieval_func:
            if args.efficientrag_file or args.negsample_file:
                # For EfficientRAG dataset, ensure documents are in the correct format
                if "documents" in question_info:
                    documents = convert_to_documents(question_info["documents"])
                else:
                    documents = []
                    logger.warning(f"No documents found for question {question_id}")
            else:
                documents = data_loader.get_corpus() if "documents" not in question_info else question_info["documents"]
            
            # Skip if no documents available
            if not documents:
                logger.warning(f"Skipping question {question_id} due to no documents")
                predictions[question_id] = "No documents available"
                continue
                
            retrieved_docs = retrieval_func(question, documents, top_k=args.top_k)
            answer = generator.generate_rag_response(question, retrieved_docs)

            print("\n" + "="*80)
            print(f"Question {i+1}: {question}")
            print("-"*80)
            print("Retrieved documents:")
            for j, doc in enumerate(retrieved_docs):
                print(f"  Doc {j+1}: {doc.page_content[:100]}...")
            print("-"*80)
            print(f"Model output: {answer}")
        else:
            answer = generator.generate_response(question)

        # Store the prediction
        predictions[question_id] = answer
        
        if "answer" in question_info and question_info["answer"]:
            print(f"Ground truth: {question_info['answer']}")
            normalized_prediction = normalize_answer(answer)
            normalized_truth = normalize_answer(question_info["answer"])
            
            if normalized_prediction == normalized_truth:
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
        print("="*80)
        
        # Also log to file
        logger.info(f"Question: {question}")
        logger.info(f"Predicted answer: {answer}")
        if "answer" in question_info and question_info["answer"]:
            logger.info(f"Ground truth: {question_info['answer']}")
    
    # Save predictions
    output_file = os.path.join(args.output_dir, f"predictions_{args.retrieval_method}_{args.top_k}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Saved predictions to {output_file}")
    
    # Evaluate if we have ground truth answers
    if args.split != "test":
        if args.efficientrag_file or args.negsample_file:
            metrics = evaluate_predictions(predictions, examples)
        else:
            metrics = evaluate_predictions(predictions, data_loader.load_split(args.split, args.max_samples))
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Save metrics
        metrics_file = os.path.join(args.output_dir, f"metrics_{args.retrieval_method}_{args.top_k}.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HotpotQA Question Answering System")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/hotpotqa", 
                        help="Directory containing HotpotQA dataset files")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"],
                        help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to process")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="qwen2.5:7b",
                        help="Ollama model name to use")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434",
                        help="Ollama API URL")
    
    # Retrieval arguments
    parser.add_argument("--retrieval_method", type=str, default="all", 
                        choices=["cosine", "mmr", "hybrid", "all", "none"],
                        help="Document retrieval method to use (or 'all' to compare)")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of documents to retrieve")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    
    parser.add_argument("--efficientrag_file", type=str,
                        help="Optional: Use EfficientRAG filter dataset (jsonl file)")
    
    parser.add_argument("--negsample_file", type=str,
                        help="Optional: Use EfficientRAG negsample dataset (valid.jsonl file)")
    
    args = parser.parse_args()
    
    # If user requested all methods, run them one by one
    if args.retrieval_method == "all":
        logger.info("Running all retrieval methods for comparison")
        
        methods = ["cosine", "mmr", "hybrid"]
        all_metrics = {}
        
        for method in methods:
            print("\n" + "#"*90)
            print(f"RUNNING RETRIEVAL METHOD: {method.upper()}")
            print("#"*90 + "\n")
            
            # Create a copy of args with the current method
            method_args = argparse.Namespace(**vars(args))
            method_args.retrieval_method = method
            
            # Run the QA system with this method
            metrics = run_qa_system(method_args)
            all_metrics[method] = metrics
    else:
        # Run with the specified method
        run_qa_system(args)

if __name__ == "__main__":
    main()
