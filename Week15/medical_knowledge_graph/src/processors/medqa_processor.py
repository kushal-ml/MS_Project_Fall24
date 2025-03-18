import os
import json
import pandas as pd
from tqdm import tqdm
import time
import torch
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
import re
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from transformers import AutoTokenizer
from pinecone import Pinecone, ServerlessSpec

# Check GPU and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # Best model for medical data
    openai_api_key=OPENAI_API_KEY
)

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# Create or get index
index_name = "medical-textbook-embeddings"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        metric='cosine',
        dimension=3072,  # Dimension for text-embedding-3-large
        spec=ServerlessSpec(
            cloud='aws',  # or 'gcp' based on your environment
            region='us-east-1'  # adjust based on your region
        )
    )

index = pc.Index(index_name)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Smaller chunks for GPU memory
    chunk_overlap=200,
    separators=[". ", "? ", "! ", ";"]  # Modified separators for better sentence splitting
)

# Load spaCy model with GPU support
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Clean text by removing unwanted elements"""
    # Remove figure and table references
    text = re.sub(r'\b(Fig|Table|Figure)\s+\d+[\.\d]*\b', '', text)
    
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with single space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,;?!-]', '', text)
    
    return text.strip()

def load_and_preprocess_textbook():
    """Load and preprocess medical textbook data"""
    try:
        base_path = os.path.join('data', 'raw', 'processed', 'medqa', 'data_clean',  'textbooks','en')
        textbook_files = [f for f in os.listdir(base_path) if f.endswith('.txt')]
        print(f"\nFound {len(textbook_files)} medical textbook files")
        
        records = []
        
        for textbook_file in tqdm(textbook_files, desc="Processing textbooks"):
            file_path = os.path.join(base_path, textbook_file)
            textbook_name = textbook_file[:-4]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Clean the text
                text = clean_text(text)
                
                # Split into chunks
                chunks = text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    # Clean each chunk individually
                    clean_chunk = clean_text(chunk)
                    
                    # Only add non-empty chunks with sufficient content
                    if len(clean_chunk.strip()) > 50:  # Minimum length threshold
                        records.append({
                            'text': clean_chunk,
                            'metadata': {
                                'source': textbook_name,
                                'chunk_id': f"{textbook_name}_chunk_{i}"
                            }
                        })
        
        return records
        
    except Exception as e:
        print(f"Error loading medical textbook data: {e}")
        raise

def process_batch(batch_records):
    """Process a batch and create fresh embeddings"""
    try:
        texts = [record['text'] for record in batch_records]
        metadatas = []
        ids = []
        
        for record in batch_records:
            metadata = record['metadata']
            metadata['text'] = record['text']  # Include text in metadata
            metadatas.append(metadata)
            ids.append(metadata['chunk_id'])
        
        # Create embeddings for the entire batch
        embeddings = openai_embeddings.embed_documents(texts)
        
        # Prepare vectors for batch upsert
        vectors = [
            (id, emb, meta) 
            for id, emb, meta in zip(ids, embeddings, metadatas)
        ]
        
        # Batch upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"\rProcessed batch of {len(vectors)} vectors", end="")
        
        time.sleep(0.1)  # Rate limiting for API
            
    except Exception as e:
        print(f"\nError in batch processing: {e}")
        torch.cuda.empty_cache()
        raise

def query_database(query_text, top_k=5):
    """Query medical knowledge base"""
    try:
        # Get query embedding
        query_embedding = openai_embeddings.embed_query(query_text)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"\nMedical Query: {query_text}")
        print("\nRelevant Medical Information:")
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. Relevance Score: {match.score:.4f}")
            print(f"Text: {match.metadata.get('text', 'No text available')[:300]}...")
            print(f"Source: {match.metadata.get('source', '')}")
            print(f"Chunk ID: {match.metadata.get('chunk_id', '')}")
            
    except Exception as e:
        print(f"Error querying database: {e}")
        torch.cuda.empty_cache()
        raise

if __name__ == "__main__":
    try:
        # Monitor GPU memory
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        print("Loading and preprocessing medical textbook data...")
        records = load_and_preprocess_textbook()
        
        print(f"\nProcessing {len(records)} medical text chunks...")
        batch_size = 64  # Smaller batch size for GPU
        total_processed = 0
        
        for i in tqdm(range(0, len(records), batch_size), desc="Processing batches"):
            if i % 100 == 0:  # Clear GPU cache periodically
                torch.cuda.empty_cache()
            batch = records[i:i + batch_size]
            process_batch(batch)
            total_processed += len(batch)
            
            if i % 10 == 0:  # Print progress every 10 batches
                print(f"\nTotal vectors processed: {total_processed}/{len(records)}")
        
        print("\nProcessing complete!")
        
        # Test the embeddings
        test_queries = [
            "What are the pathophysiological mechanisms of heart failure?",
            "Explain the mechanism of action of beta-blockers in hypertension",
            "What are the diagnostic criteria for Type 2 Diabetes Mellitus?",
            "Describe the inflammatory cascade in autoimmune diseases",
            "What are the key features of Alzheimer's disease pathology?"
        ]
        
        print("\nTesting medical knowledge retrieval...")
        for query in test_queries:
            query_database(query)
            time.sleep(1)  # Rate limiting for API
            
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")