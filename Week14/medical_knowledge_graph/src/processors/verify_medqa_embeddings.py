import os
from chromadb import Client
from chromadb.config import Settings

# Define the ChromaDB path
CHROMA_PATH = os.path.join('Week14', 'medical_knowledge_graph', 'chroma_data')

def verify_embeddings():
    try:
        # Initialize ChromaDB client
        print(f"\nConnecting to ChromaDB at: {os.path.abspath(CHROMA_PATH)}")
        client = Client(Settings(
            persist_directory=CHROMA_PATH,
            is_persistent=True
        ))
        
        # List all collections
        print("\nAvailable collections:")
        collections = client.list_collections()
        for collection in collections:
            print(f"- {collection.name}")
        
        if not collections:
            print("No collections found in the database!")
            return False
            
        # Try to get the first collection if it exists
        collection = collections[0]
        print(f"\nUsing collection: {collection.name}")
        
        # Get collection stats
        count = collection.count()
        print(f"\nVerification Results:")
        print(f"Number of embeddings stored: {count}")
        
        # Try retrieving a sample
        if count > 0:
            print("\nRetrieving sample document...")
            sample = collection.peek(limit=1)
            print(f"Sample document: {sample['documents'][0][:100]}...")
            print(f"Sample metadata: {sample['metadatas'][0]}")
            return True
        else:
            print("Collection is empty!")
            return False
            
    except Exception as e:
        print(f"Error during verification: {e}")
        print("\nPossible issues:")
        print("1. ChromaDB directory doesn't exist")
        print("2. No collections found")
        print("3. Permissions issue with the directory")
        print(f"4. Database file exists but might be corrupted")
        
        # Print the contents of the ChromaDB directory
        print("\nChromaDB directory contents:")
        try:
            for root, dirs, files in os.walk(CHROMA_PATH):
                print(f"\nDirectory: {root}")
                print("Files:", files)
        except Exception as e:
            print(f"Error listing directory contents: {e}")
        return False

if __name__ == "__main__":
    verify_embeddings()