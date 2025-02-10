import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from tqdm import tqdm
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re  # Import the regular expressions module
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Embedding Model and ChromaDB
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Initialize ChromaDB Client with the new configuration
chroma_client = Client(Settings(
    persist_directory="chroma_data",
    # You may need to specify other settings based on the new configuration
))
collection_name = "english_textbook_embeddings"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def generate_summary(text):
    sentences = text.split('. ')  # Split into sentences
    summary = []
    
    for sentence in sentences:
        doc = nlp(sentence)
        # Extract keywords: NOUN, VERB, ADJ
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        summary.append(' '.join(keywords[:6]) + '...')  # Take meaningful words

    return ' '.join(summary)

def load_and_preprocess_textbook():
    """
    Load and preprocess English textbook data
    """
    try:
        base_path = os.path.join('data_clean', 'data_clean', 'textbooks', 'en')
        
        # Get list of all textbook files
        textbook_files = [f for f in os.listdir(base_path) if f.endswith('.txt')]
        print(f"\nFound {len(textbook_files)} textbook files")
        
        records = []
        total_records = 0
        
        # Process each textbook file
        for textbook_file in textbook_files:
            print(f"\nProcessing {textbook_file}...")
            file_path = os.path.join(base_path, textbook_file)
            
            # Extract textbook name from filename (remove .txt extension)
            textbook_name = textbook_file[:-4]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the entire text file
                text = f.read()
                
                # Remove unnecessary references like "Fig 1.1" or "Table 1.1"
                text = re.sub(r'\b(Fig|Table) \d+\.\d+\b', '', text)  # Adjust regex as needed
                
                # Split the text into sentences using the text splitter
                sentences = text_splitter.split_text(text)
                
                # Add each sentence as a record
                for sentence in sentences:
                    total_records += 1
                    # Generate a meaningful summary for the section
                    section_summary = generate_summary(sentence)  # Use the new summary function
                    records.append({
                        'content': sentence,
                        'meta_info': {
                            'source': textbook_name,
                            'section': section_summary  # Store the summary in section
                        }
                    })
            
            print(f"Processed {len(sentences)} sentences from {textbook_name}")

        print(f"\nData Analysis:")
        print(f"Total textbook sentences: {total_records}")
        
        data = pd.DataFrame(records)
        print(f"Final processed records: {len(data)}")
        return data
        
    except Exception as e:
        print(f"Error loading textbook data: {e}")
        raise

def process_batch(batch_data):
    """
    Process a single batch of data to generate embeddings.
    """
    contents = batch_data['content'].tolist()
    embeddings = embedding_model.encode(contents, convert_to_tensor=False)
    embeddings_list = embeddings.tolist()
    
    # Ensure metadata matches the number of contents
    metadata = [batch_data.iloc[i]['meta_info'] for i in range(len(batch_data))]
    
    # Generate ids based on the current batch
    start_idx = batch_data.index[0]  # Get the starting index of the batch
    end_idx = batch_data.index[-1]    # Get the ending index of the batch
    ids = [str(i) for i in range(start_idx, end_idx + 1)]  # Adjust to include end_idx

    # Debugging output
    print(f"Length of ids: {len(ids)}")
    print(f"Length of embeddings: {len(embeddings_list)}")
    print(f"Length of documents: {len(contents)}")
    print(f"Length of metadatas: {len(metadata)}")

    return embeddings_list, contents, metadata, ids

def process_in_batches(data, batch_size=1000):
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            batch_data = data.iloc[start_idx:end_idx]
            
            # Submit the batch processing to the executor
            futures.append(executor.submit(process_batch, batch_data))
        
        for future in as_completed(futures):
            embeddings_list, contents, metadata, ids = future.result()
            
            # Add to ChromaDB
            print(f"Adding batch to ChromaDB...")
            collection.add(
                embeddings=embeddings_list,
                documents=contents,
                metadatas=metadata,
                ids=ids
            )
            
            print(f"Batch completed. Processed {end_idx}/{len(data)} items")
            time.sleep(1)  # Small delay between batches

def query_database(query_text, top_k=5):
    # Generate embedding for the query and convert to list
    query_embedding = embedding_model.encode([query_text], convert_to_tensor=False).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    print(f"\nQuery: {query_text}")
    print("\nTop Results:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\n{i}. Text: {doc}")
        print(f"   Source: {meta['source']}")
        print(f"   Section: {meta['section']}")

if __name__ == "__main__":
    try:
        # Load and preprocess textbook data
        print("Loading and preprocessing textbook data...")
        data = load_and_preprocess_textbook()
        print(f"Processed {len(data)} items")
        
        # Process in batches
        print("\nStarting batch processing...")
        process_in_batches(data, batch_size=1000)
        
        print("\nAll data processed successfully!")
        
        # Test retrievals
        print("\nTesting retrievals...")
        test_queries = [
            "What are the symptoms of a heart attack?",
            "How does DNA replication work?",
            "How does the immune system work?",
            "What is the role of the heart in the circulatory system?",
            "What is the function of the kidneys in the urinary system?",
            "What is the structure of the human brain?",
            "What is the function of the lungs in the respiratory system?",
            "What is the role of the liver in the digestive system?",
            "What is the function of the pancreas in the digestive system?",
            "What is the role of the spleen in the immune system?",
            "What is the function of the stomach in the digestive system?",
            "What is the role of the small intestine in the digestive system?",
            "What is the function of the large intestine in the digestive system?",
            "What is the role of the kidneys in the urinary system?",
            "What is the function of the lungs in the respiratory system?",
            "What is the role of the liver in the digestive system?",
            "What is the function of the pancreas in the digestive system?",
            "What is the role of the spleen in the immune system?",
            "What is the function of the stomach in the digestive system?",
            "What is the role of the small intestine in the digestive system?",
            "What is the function of the large intestine in the digestive system?",
        ]
        
        for query in test_queries:
            query_database(query)
            
    except Exception as e:
        print(f"Error in main execution: {e}")