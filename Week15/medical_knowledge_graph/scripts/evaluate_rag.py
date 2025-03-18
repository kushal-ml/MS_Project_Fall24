import json
import nltk
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from query_medical_db import query_medical_knowledge

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
    
    bleu = sentence_bleu([reference.split()], generated.split())
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

# Run Evaluation
def evaluate_rag(jsonl_path):
    dataset = load_jsonl(jsonl_path)
    results = []
    
    for question, ground_truth, options in dataset:
        print(f"\nEvaluating Question: {question}")
        print(f"Options: {options}")
        
        # Format options for better LLM understanding
        formatted_options = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"""Question: {question}

Available options:
{formatted_options}

Based on the retrieved medical knowledge, select the most appropriate answer option (A, B, C, or D) and explain your reasoning."""
        
        # Pass both the prompt and options to query_medical_knowledge
        generated_answer = query_medical_knowledge(prompt, options)
        
        if not generated_answer:
            print("Warning: No answer generated for this question")
            generated_answer = "No answer found"
        
        metrics = compute_metrics(ground_truth, generated_answer)
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "options": options,
            "generated_answer": generated_answer,
            "metrics": metrics
        })
        
        print(f"Generated Answer: {generated_answer}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Metrics: {metrics}")
    
    return results

# Run the evaluation
if __name__ == "__main__":
    jsonl_path = r"E:\MS CS LU\MS Project\Week 14 & 15\medqa_english\data_clean\data_clean\questions\US\4_options\phrases_no_exclude_test.jsonl"
    evaluate_rag(jsonl_path)
