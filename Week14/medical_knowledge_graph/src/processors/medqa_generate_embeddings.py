import os
import re
import time
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define paths
BASE_PATH = os.path.join('Week14', 'medical_knowledge_graph', 'data', 'raw', 'processed', 'data_clean', 'textbooks', 'en')
CHROMA_PATH = os.path.join('Week14', 'medical_knowledge_graph', 'chroma_data')

class MedicalTextProcessor:
    def __init__(self):
        try:
            # Initialize models and tools
            print("Initializing models and tools...")
            self.embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
            self.nlp = spacy.load("en_core_web_sm")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            # Initialize ChromaDB
            print("Initializing ChromaDB...")
            self.chroma_client = Client(Settings(
                persist_directory=CHROMA_PATH,
                is_persistent=True,
                anonymized_telemetry=False
            ))
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.create_collection(
                    name="medical_textbook_embeddings",
                    metadata={"description": "Medical textbook embeddings with section metadata"}
                )
                print("Created new collection: medical_textbook_embeddings")
            except Exception as e:
                print(f"Collection might exist, getting existing collection: {e}")
                self.collection = self.chroma_client.get_collection(name="medical_textbook_embeddings")
                
        except Exception as e:
            print(f"Error initializing MedicalTextProcessor: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the text"""
        try:
            # Remove figure and table references
            text = re.sub(r'\b(Fig\.|Figure|Table)\s+\d+(\.\d+)?', '', text)
            
            # Remove multiple spaces and newlines
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep necessary punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,;:()\-\'"]', '', text)
            
            return text.strip()
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return text  # Return original text if preprocessing fails

    def extract_section_info(self, text: str) -> Dict[str, str]:
        """Extract key medical terms and concepts from text"""
        try:
            doc = self.nlp(text[:1000000])  # Limit text length to prevent memory issues
            
            # Extract all named entities (not just medical)
            entities = [ent.text for ent in doc.ents]
            
            # Extract key noun phrases
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Get main topics using noun and verb phrases
            main_topics = [token.text for token in doc if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 3]
            
            return {
                "medical_entities": list(set(entities))[:5],
                "key_concepts": list(set(noun_phrases))[:5],
                "main_topics": list(set(main_topics))[:5]
            }
        except Exception as e:
            print(f"Error in section info extraction: {e}")
            return {"medical_entities": [], "key_concepts": [], "main_topics": []}

    def process_batch(self, texts: List[str], source: str) -> tuple:
        """Process a batch of texts"""
        try:
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(processed_texts, convert_to_tensor=False)
            
            # Generate metadata for each text
            metadata = []
            for text in processed_texts:
                section_info = self.extract_section_info(text)
                metadata.append({
                    "source": source,
                    "medical_entities": "; ".join(section_info["medical_entities"]),  # Convert list to string
                    "key_concepts": "; ".join(section_info["key_concepts"]),         # Convert list to string
                    "main_topics": "; ".join(section_info["main_topics"]),          # Convert list to string
                    "length": len(text),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return embeddings.tolist(), processed_texts, metadata
        except Exception as e:
            print(f"Error in batch processing: {e}")
            raise

    def process_textbook(self, file_path: str):
        """Process a single textbook file"""
        try:
            textbook_name = os.path.basename(file_path)[:-4]
            print(f"\nProcessing {textbook_name}...")
            
            # Try different encodings
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            print(f"Split into {len(chunks)} chunks")
            
            # Process in batches
            batch_size = 50
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
                batch_chunks = chunks[i:i + batch_size]
                embeddings, texts, metadata = self.process_batch(batch_chunks, textbook_name)
                
                # Generate unique IDs for this batch
                timestamp = int(time.time() * 1000)
                ids = [f"{textbook_name}_{i + j}_{timestamp}" for j in range(len(batch_chunks))]
                
                # Add to ChromaDB with retry mechanism
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        self.collection.add(
                            embeddings=embeddings,
                            documents=texts,
                            metadatas=metadata,
                            ids=ids
                        )
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            print(f"Failed to add batch after {max_retries} retries: {e}")
                            raise
                        time.sleep(1)  # Wait before retry
                
                time.sleep(0.1)  # Prevent overwhelming the database
            
            return len(chunks)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return 0

    def verify_embeddings(self):
        """Verify the stored embeddings"""
        try:
            count = self.collection.count()
            print(f"\nVerification Results:")
            print(f"Total embeddings stored: {count}")
            
            if count > 0:
                sample = self.collection.peek(limit=1)
                print("\nSample Document:")
                print(f"Text: {sample['documents'][0][:200]}...")
                print(f"Metadata: {sample['metadatas'][0]}")
                return True
            return False
        except Exception as e:
            print(f"Error verifying embeddings: {e}")
            return False

def main():
    try:
        # Verify paths exist
        if not os.path.exists(BASE_PATH):
            raise FileNotFoundError(f"Textbook directory not found: {BASE_PATH}")
        
        # Create ChromaDB directory if it doesn't exist
        os.makedirs(CHROMA_PATH, exist_ok=True)
        print(f"Using ChromaDB directory: {os.path.abspath(CHROMA_PATH)}")
        
        # Initialize processor
        processor = MedicalTextProcessor()
        
        # Process all textbook files
        total_chunks = 0
        textbook_files = [f for f in os.listdir(BASE_PATH) if f.endswith('.txt')]
        
        if not textbook_files:
            raise FileNotFoundError(f"No .txt files found in {BASE_PATH}")
        
        print(f"\nFound {len(textbook_files)} textbook files")
        
        for textbook_file in textbook_files:
            file_path = os.path.join(BASE_PATH, textbook_file)
            chunks_processed = processor.process_textbook(file_path)
            total_chunks += chunks_processed
            
            # Verify after each file
            print(f"\nVerifying after processing {textbook_file}...")
            processor.verify_embeddings()
        
        print(f"\nProcessing complete!")
        print(f"Total chunks processed: {total_chunks}")
        
        # Final verification
        if processor.verify_embeddings():
            print("\nAll embeddings verified successfully!")
        else:
            print("\nWarning: Final verification failed!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()