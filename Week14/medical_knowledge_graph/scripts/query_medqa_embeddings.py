import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings

class MedicalKnowledgeBase:
    def __init__(self):
        # Define paths
        self.chroma_path = os.path.join('Week14', 'medical_knowledge_graph', 'chroma_data')
        
        # Initialize models
        print("Initializing models...")
        self.embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
        # Initialize ChromaDB
        print("Connecting to ChromaDB...")
        self.client = Client(Settings(
            persist_directory=self.chroma_path,
            is_persistent=True
        ))
        
        # Get the collection
        try:
            self.collection = self.client.get_collection(name="medical_textbook_embeddings")
            print(f"Connected to collection. Total documents: {self.collection.count()}")
        except Exception as e:
            raise Exception(f"Error connecting to collection: {e}")

    def search(self, query: str, n_results: int = 5, source: str = None) -> List[Dict]:
        """
        Search the medical knowledge base
        Args:
            query: The search query
            n_results: Number of results to return
            source: Optional textbook name to search within
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare search parameters
            search_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add source filter if specified
            if source:
                search_params["where"] = {"source": source}
            
            # Perform search
            results = self.collection.query(**search_params)
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                # Calculate cosine similarity score (0 to 1 scale)
                distance = results['distances'][0][i]
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity score
                
                # Parse metadata strings back into lists
                metadata = results['metadatas'][0][i].copy()
                if isinstance(metadata.get('medical_entities'), str):
                    metadata['medical_entities'] = metadata['medical_entities'].split('; ')
                if isinstance(metadata.get('key_concepts'), str):
                    metadata['key_concepts'] = metadata['key_concepts'].split('; ')
                
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': metadata,
                    'similarity_score': similarity_score
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def list_sources(self) -> List[str]:
        """List all available textbook sources"""
        try:
            # Get all unique sources
            results = self.collection.get()
            sources = set(meta['source'] for meta in results['metadatas'])
            return sorted(list(sources))
        except Exception as e:
            print(f"Error listing sources: {e}")
            return []

def main():
    try:
        # Initialize the knowledge base
        kb = MedicalKnowledgeBase()
        
        # List available sources
        sources = kb.list_sources()
        print("\nAvailable textbooks:")
        for source in sources:
            print(f"- {source}")
        
        # Interactive search loop
        while True:
            print("\nEnter your medical query (or 'quit' to exit):")
            query = input().strip()
            
            if query.lower() == 'quit':
                break
            
            print("\nSearching all textbooks or specific one? (Enter textbook name or press Enter for all):")
            source = input().strip()
            source = source if source else None
            
            # Perform search
            results = kb.search(query, n_results=3, source=source)
            
            # Display results
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Source: {result['metadata']['source']}")
                print(f"Text: {result['text'][:300]}...")
                print(f"Medical Entities: {', '.join(result['metadata']['medical_entities'])}")
                print(f"Key Concepts: {', '.join(result['metadata']['key_concepts'])}")
                print(f"Relevance Score: {result['similarity_score']:.2f}")
                print("-" * 80)
                
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()