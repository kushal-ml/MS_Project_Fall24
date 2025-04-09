import json
import nltk
import random
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from query_medical_db import query_medical_knowledge
from typing import List, Dict
import openai
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Connect to existing index
index_name = "medical-textbook-embeddings"
index = pc.Index(index_name)

# Load questions, ground truth answers, and options from JSONL
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append((item['question'], item['answer'], item['options']))  # Extract question, answer & options
    return data

# Evaluation Metrics
def compute_metrics(reference, generated):
    # Handle case where generated answer is None or empty
    if not generated:
        return {
            "BLEU": 0.0,
            "ROUGE-1": 0.0,
            "ROUGE-2": 0.0,
            "ROUGE-L": 0.0,
            "BERTScore-F1": 0.0
        }
    
    from nltk.translate.bleu_score import SmoothingFunction  
    smoother = SmoothingFunction().method1  
    bleu = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoother)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(reference, generated)
    P, R, F1 = score([generated], [reference], lang="en", verbose=False)
    
    return {
        "BLEU": bleu,
        "ROUGE-1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].fmeasure,
        "BERTScore-F1": F1.item()
    }

def calculate_average_metrics(results):
    """Calculate average scores across all evaluated questions"""
    metrics_sum = {
        "BLEU": 0.0,
        "ROUGE-1": 0.0,
        "ROUGE-2": 0.0,
        "ROUGE-L": 0.0,
        "BERTScore-F1": 0.0
    }
    
    for result in results:
        for metric, value in result["metrics"].items():
            metrics_sum[metric] += value
    
    num_questions = len(results)
    return {metric: value/num_questions for metric, value in metrics_sum.items()}

# Add new functions for retrieval metrics
def calculate_precision_at_k(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
    """
    Calculate Precision@k
    retrieved_docs: List of retrieved documents with their metadata
    relevant_docs: List of relevant document IDs
    k: Number of top documents to consider
    """
    if not retrieved_docs or k == 0:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in top_k if doc['id'] in relevant_docs)
    return relevant_retrieved / k

def calculate_recall_at_k(retrieved_docs: List[Dict], relevant_docs: List[str], k: int) -> float:
    """
    Calculate Recall@k
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in top_k if doc['id'] in relevant_docs)
    return relevant_retrieved / len(relevant_docs)

def calculate_mrr(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc['id'] in relevant_docs:
            return 1.0 / rank
    return 0.0

# Modify the evaluate_rag function to include retrieval metrics
def evaluate_rag(jsonl_path):
    dataset = load_jsonl(jsonl_path)
    results = []
    
    # Initialize retrieval metrics
    retrieval_metrics = {
        'precision@1': [],
        'precision@3': [],
        'recall@1': [],
        'recall@3': [],
        'mrr': []
    }
    
    print(f"Evaluating {len(dataset)} questions... Please wait...")
    
    for question, ground_truth, options in dataset:
        formatted_options = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"""Question: {question}

Available options:
{formatted_options}

Based on the retrieved medical knowledge, select the most appropriate answer option (A, B, C, or D) and explain your reasoning."""
        
        # Get retrieved documents and their metadata
        query_embedding = openai_embeddings.embed_query(prompt)
        retrieved_results = index.query(
            vector=query_embedding,
            top_k=5,  # Retrieve top 5 for evaluation
            include_metadata=True
        )
        
        # Format retrieved documents for metric calculation
        retrieved_docs = [{
            'id': match.id,
            'score': match.score,
            'text': match.metadata.get('text', '')
        } for match in retrieved_results.matches]
        
        # Determine relevant documents (you'll need to define this based on your data)
        # For example, documents containing the ground truth answer
        relevant_docs = [doc['id'] for doc in retrieved_docs 
                        if ground_truth.lower() in doc['text'].lower()]
        
        # Calculate retrieval metrics
        retrieval_metrics['precision@1'].append(
            calculate_precision_at_k(retrieved_docs, relevant_docs, k=1))
        retrieval_metrics['precision@3'].append(
            calculate_precision_at_k(retrieved_docs, relevant_docs, k=3))
        retrieval_metrics['recall@1'].append(
            calculate_recall_at_k(retrieved_docs, relevant_docs, k=1))
        retrieval_metrics['recall@3'].append(
            calculate_recall_at_k(retrieved_docs, relevant_docs, k=3))
        retrieval_metrics['mrr'].append(
            calculate_mrr(retrieved_docs, relevant_docs))
        
        # Generate answer and calculate existing metrics
        generated_answer = query_medical_knowledge(prompt, options)
        
        if not generated_answer:
            generated_answer = "No answer found"
        
        metrics = compute_metrics(ground_truth, generated_answer)
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "options": options,
            "generated_answer": generated_answer,
            "metrics": metrics,
            "retrieval_metrics": {
                'precision@1': retrieval_metrics['precision@1'][-1],
                'precision@3': retrieval_metrics['precision@3'][-1],
                'recall@1': retrieval_metrics['recall@1'][-1],
                'recall@3': retrieval_metrics['recall@3'][-1],
                'mrr': retrieval_metrics['mrr'][-1]
            }
        })
    
    # Calculate average metrics
    avg_metrics = calculate_average_metrics(results)
    avg_retrieval_metrics = {
        metric: np.mean(values) 
        for metric, values in retrieval_metrics.items()
    }
    
    # Print results
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print("\nAverage Scores Across All Questions:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nRetrieval Quality Metrics:")
    for metric, value in avg_retrieval_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Display 3 random examples with retrieval metrics
    print("\n" + "="*50)
    print("SAMPLE EVALUATIONS (3 Random Questions)")
    print("="*50)
    
    sample_results = random.sample(results, min(3, len(results)))
    for i, result in enumerate(sample_results, 1):
        print(f"\nExample {i}:")
        print(f"Question: {result['question']}")
        print("\nOptions:")
        for key, value in result['options'].items():
            print(f"{key}: {value}")
        print(f"\nGenerated Answer: {result['generated_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Metrics: {result['metrics']}")
        print(f"Retrieval Metrics: {result['retrieval_metrics']}")
        print("-" * 80)
    
    return results, avg_retrieval_metrics

# Run the evaluation
if __name__ == "__main__":
    jsonl_path = r"E:\MS CS LU\MS Project\Week 14 & 15\medqa_english\data_clean\data_clean\questions\US\4_options\phrases_no_exclude_test.jsonl"
    evaluate_rag(jsonl_path)
