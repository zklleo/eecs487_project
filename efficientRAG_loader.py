import json
from langchain.docstore.document import Document

def convert_to_documents(doc_texts):
    """
    Convert raw text documents to Document objects for the retriever.
    
    Args:
        doc_texts: List of document texts or text string
        
    Returns:
        List of Document objects
    """
    if isinstance(doc_texts, str):
        # Handle single string case
        return [Document(page_content=doc_texts)]
    elif isinstance(doc_texts, list):
        # Handle list of strings
        return [Document(page_content=text) for text in doc_texts]
    else:
        # Return empty list for invalid input
        return []

def load_efficientrag_negsample_dataset(filepath: str, max_samples: int = None):
    """
    Load and process the EfficientRAG negsample dataset (valid.jsonl format).
    
    Args:
        filepath: Path to the negsample jsonl file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of processed examples with questions, answers and documents
    """
    examples = []

    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            entry = json.loads(line)
            
            # Extract main question and answer
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            question_id = entry.get("id", f"negsample_{i}")
            
            # Extract documents from decomposed questions
            documents = []
            if "decomposed_questions" in entry:
                for _, sub_q_data in entry["decomposed_questions"].items():
                    # Add positive paragraphs
                    if "positive_paragraph" in sub_q_data:
                        documents.append(sub_q_data["positive_paragraph"])
                    
                    # Add negative paragraphs (for contrast learning)
                    if "negative_paragraph" in sub_q_data:
                        documents.append(sub_q_data["negative_paragraph"])
            
            examples.append({
                "id": question_id,
                "question": question,
                "answer": answer,
                "documents": documents
            })

    return examples