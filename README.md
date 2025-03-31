# HotpotQA Retrieval Project

This project implements and evaluates different retrieval algorithms for the HotpotQA dataset, focusing on multi-hop question answering.

---

## Setup

### 1. Install Dependencies

```bash
pip install langchain langchain_community numpy scikit-learn tqdm sentence-transformers
```

### 2. Download HotpotQA Dataset

```bash
python main.py
```

### 3. Install and Run Ollama

- Download Ollama from [ollama.ai](https://ollama.ai)
- Pull the Qwen2.5 model:

```bash
ollama pull qwen2.5:7b
```
- Keep Ollama running in the background.

---

## Running Evaluations

### Evaluate Baseline Retrieval Methods

```bash
python baseline.py
```

### Evaluate Custom Retrieval Method

```bash
python custom_retrieval.py
```

---

## Project Structure

- `main.py`: Downloads the HotpotQA dataset.
- `dataloader.py`: Handles loading and processing HotpotQA data.
- `embedding.py`: Implements embedding functionality using Ollama with SentenceTransformers fallback.
- `generation.py`: Provides text generation capabilities for query decomposition.
- `evaluation.py`: Contains the official HotpotQA evaluation metrics.
- `baseline.py`: Implements and evaluates baseline retrieval methods.
- `custom_retrieval.py`: Implements our custom retrieval approach.
