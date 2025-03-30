import re
import string
import json
import os
from collections import Counter
from typing import Dict, List, Tuple, Any, Set
from tqdm import tqdm

def normalize_answer(s):
    """Normalize answer text for fair comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_supporting_facts(prediction_sp, gold_sp):
    """
    Evaluate supporting facts.
    
    Args:
        prediction_sp: List of predicted supporting facts [(title, sent_id), ...]
        gold_sp: List of gold supporting facts [(title, sent_id), ...]
        
    Returns:
        Tuple of (em, f1, precision, recall)
    """
    cur_sp_pred = set(map(tuple, prediction_sp))
    gold_sp_pred = set(map(tuple, gold_sp))
    tp, fp, fn = 0, 0, 0
    
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    
    return em, f1, prec, recall

def evaluate_retrieval_with_official_metrics(predictions, ground_truths):
    """
    Evaluate retrieval performance using official HotpotQA metrics.
    
    Args:
        predictions: Dict mapping question IDs to lists of retrieved document titles
        ground_truths: List of question info with supporting facts
        
    Returns:
        Dict of evaluation metrics
    """
    metrics = {
        'sp_em': 0, 
        'sp_f1': 0, 
        'sp_precision': 0, 
        'sp_recall': 0
    }
    
    total = 0
    for gt in ground_truths:
        qid = gt['id']
        
        if qid not in predictions:
            continue
            
        # Get predicted supporting facts (we only have titles, not sentence IDs)
        # Convert to expected format: [(title, 0), (title, 0), ...]
        pred_titles = predictions[qid]
        pred_sp = [(title, 0) for title in pred_titles]
        
        # Get gold supporting facts
        gold_sp = gt.get('supporting_facts', [])
        
        # Evaluate
        em, f1, precision, recall = evaluate_supporting_facts(pred_sp, gold_sp)
        
        metrics['sp_em'] += em
        metrics['sp_f1'] += f1
        metrics['sp_precision'] += precision
        metrics['sp_recall'] += recall
        
        total += 1
    
    # Calculate averages
    if total > 0:
        for k in metrics:
            metrics[k] /= total
    
    return metrics

def run_and_evaluate_methods(examples, methods_dict):
    """
    Run retrieval methods and evaluate them using official metrics.
    
    Args:
        examples: List of (question_info, documents) tuples
        methods_dict: Dict mapping method names to retrieval functions
        
    Returns:
        Tuple of (retrieval_results, evaluation_results)
    """
    # Prepare ground truth data
    ground_truth_data = [question_info for question_info, _ in examples]
    
    # Run retrieval methods
    retrieval_results = {}
    
    for method_name, retrieval_func in methods_dict.items():
        print(f"\nRunning {method_name} retrieval...")
        method_results = {}
        
        for question_info, documents in tqdm(examples, desc=f"Running {method_name}"):
            qid = question_info['id']
            query = question_info['question']
            
            # Run retrieval
            retrieved_docs = retrieval_func(query, documents, top_k=2)
            
            # Store retrieved document titles
            retrieved_titles = [doc.metadata['title'] for doc in retrieved_docs]
            method_results[qid] = retrieved_titles
        
        retrieval_results[method_name] = method_results
    
    # Evaluate retrieval methods
    evaluation_results = {}
    for method_name, method_results in retrieval_results.items():
        print(f"\nEvaluating {method_name} with official metrics...")
        metrics = evaluate_retrieval_with_official_metrics(method_results, ground_truth_data)
        evaluation_results[method_name] = metrics
        
        # Print results
        print(f"{method_name} results:")
        print(f"  Supporting Facts EM: {metrics['sp_em']:.4f}")
        print(f"  Supporting Facts F1: {metrics['sp_f1']:.4f}")
        print(f"  Supporting Facts Precision: {metrics['sp_precision']:.4f}")
        print(f"  Supporting Facts Recall: {metrics['sp_recall']:.4f}")
    
    return retrieval_results, evaluation_results

def save_evaluation_results(evaluation_results, output_path="results/official_evaluation.json"):
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Saved evaluation results to {output_path}")