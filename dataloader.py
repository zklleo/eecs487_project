import json
import os
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document

class HotpotQALoader:
    """
    Loader for the HotpotQA dataset.
    """
    
    def __init__(self, data_dir: str = 'data/hotpotqa'):
        """
        Initialize the HotpotQA loader.
        
        Args:
            data_dir: Directory containing the HotpotQA dataset files.
        """
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'hotpot_train.json')
        self.dev_path = os.path.join(data_dir, 'hotpot_dev.json')
        self.test_path = os.path.join(data_dir, 'hotpot_test.json')
    
    def load_split(self, split: str = 'dev', max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Load a specific split of the dataset.
        
        Args:
            split: The split to load ('train', 'dev', or 'test').
            max_samples: Maximum number of samples to load.
            
        Returns:
            A list of processed examples.
        """
        if split == 'train':
            path = self.train_path
        elif split == 'dev':
            path = self.dev_path
        elif split == 'test':
            path = self.test_path
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'test'.")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}. Did you download it?")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
            
        return data
    
    def get_question_answer_pairs(self, split: str = 'dev', max_samples: int = None) -> List[Dict[str, str]]:
        """
        Extract just the question-answer pairs from the dataset.
        
        Args:
            split: The split to load ('train', 'dev', or 'test').
            max_samples: Maximum number of samples to load.
            
        Returns:
            A list of question-answer pairs.
        """
        data = self.load_split(split, max_samples)
        qa_pairs = []
        
        for item in data:
            qa_pair = {
                'question': item['question'],
                'answer': item['answer'] if 'answer' in item else None,
                'id': item['_id']
            }
            qa_pairs.append(qa_pair)
            
        return qa_pairs
    
    def process_single_example(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Document]]:
        """
        Process a single example into a question and its associated documents.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            A tuple of (question_info, documents)
        """
        # Extract question information
        question_info = {
            'id': example['_id'],
            'question': example['question'],
            'answer': example.get('answer'),
            'supporting_facts': example.get('supporting_facts', [])
        }
        
        # Process the context paragraphs for this specific question
        documents = []
        for title, paragraphs in example['context']:
            # Join the sentences into a single paragraph
            full_paragraph = ' '.join(paragraphs)
            
            if full_paragraph.strip():  # Skip empty paragraphs
                doc = Document(
                    page_content=full_paragraph,
                    metadata={
                        'title': title,
                        'document_id': title,
                        'question_id': example['_id']
                    }
                )
                documents.append(doc)
        
        return question_info, documents
    
    def get_examples_with_contexts(self, split: str = 'dev', max_samples: int = None) -> List[Tuple[Dict[str, Any], List[Document]]]:
        """
        Get examples with their associated contexts.
        
        Args:
            split: The split to load ('train', 'dev', or 'test').
            max_samples: Maximum number of samples to load.
            
        Returns:
            A list of (question_info, documents) tuples
        """
        data = self.load_split(split, max_samples)
        examples = []
        
        for example in data:
            question_info, documents = self.process_single_example(example)
            examples.append((question_info, documents))
        
        return examples