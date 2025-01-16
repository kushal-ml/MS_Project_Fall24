import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import time

# Step 1: Initialize Embedding Model and ChromaDB
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_data",
    chroma_db_impl="duckdb+parquet"
))
collection_name = "english_medqa_embeddings"
collection = chroma_client.get_or_create_collection(name=collection_name)

def load_and_preprocess_dataset(source="US", split=None):
    """
    Load and preprocess English MedQA dataset with 4 options
    """
    try:
        base_path = os.path.join('data_clean', 'data_clean', 'questions')
        
        if split:
            file_path = os.path.join(base_path, source, '4_options', f'phrases_no_exclude_{split}.jsonl')
        else:
            file_path = os.path.join(base_path, source, '4_options', f'phrases_no_exclude_train.jsonl')
            
        print(f"Attempting to load file: {file_path}")

        # Read JSONL file
        records = []
        total_records = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_records += 1
                entry = json.loads(line.strip())
                
                # Check if options is a dictionary with 4 items
                if isinstance(entry.get('options'), dict) and len(entry['options']) == 4:
                    options_text = '\n'.join(f"{k}: {v}" for k, v in entry['options'].items())
                    content = f"Question: {entry['question']}\nOptions: {options_text}\nAnswer: {entry['answer']}"
                    
                    records.append({
                        'content': content,
                        'meta_info': entry.get('meta_info', 'N/A'),
                        'question': entry['question'],
                        'answer': entry['answer']
                    })

        print(f"\nData Analysis:")
        print(f"Total records: {total_records}")
        print(f"Records with 4 options: {len(records)}")
        
        data = pd.DataFrame(records)
        print(f"Final processed records: {len(data)}")
        return data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def process_in_batches(data, batch_size=1000):
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        
        batch_data = data.iloc[start_idx:end_idx]
        
        # Process batch
        contents = batch_data['content'].tolist()
        print(f"\nGenerating embeddings for batch {batch_idx + 1}/{total_batches} ({len(contents)} items)...")
        
        # Generate embeddings and convert to list
        embeddings = embedding_model.encode(contents, convert_to_tensor=False)
        embeddings_list = embeddings.tolist()
        
        metadata = [{"meta_info": str(batch_data.iloc[i]['meta_info']),
                    "question": str(batch_data.iloc[i]['question']),
                    "answer": str(batch_data.iloc[i]['answer'])} 
                   for i in range(len(batch_data))]
        
        # Add to ChromaDB
        print(f"Adding batch {batch_idx + 1} to ChromaDB...")
        collection.add(
            embeddings=embeddings_list,
            documents=contents,
            metadatas=metadata,
            ids=[str(i) for i in range(start_idx, end_idx)]
        )
        
        print(f"Batch {batch_idx + 1} completed. Processed {end_idx}/{len(data)} items")
        time.sleep(1)  # Small delay between batches

def query_database(question, top_k=5):
    # Generate embedding for the question and convert to list
    question_embedding = embedding_model.encode([question], convert_to_tensor=False).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    
    print(f"\nQuery: {question}")
    print("\nTop Results:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\n{i}. Question: {meta['question']}")
        print(f"   Answer: {meta['answer']}")
        print(f"   Meta Info: {meta['meta_info']}")

if __name__ == "__main__":
    try:
        # Load and preprocess dataset
        print("Loading and preprocessing dataset...")
        data = load_and_preprocess_dataset(source="US", split="train")
        print(f"Processed {len(data)} items")
        
        # Process in batches
        print("\nStarting batch processing...")
        process_in_batches(data, batch_size=1000)
        
        print("\nAll data processed successfully!")
        
        # Test retrievals
        print("\nTesting retrievals...")
        test_queries = [
            "Which coenzyme contains vitamin pantothenic acid?",
            "What is the maximum UV absorption peak of nucleic acids?",
            "Explain the basic principles of biochemistry."
        ]
        
        for query in test_queries:
            query_database(query)
            
    except Exception as e:
        print(f"Error in main execution: {e}")
