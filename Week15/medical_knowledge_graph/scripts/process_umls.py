import sys
from pathlib import Path
import os

# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
import logging
from src.processors.umls_processor import UMLSProcessor
from src.config.constants import IMPORTANT_SEMANTIC_TYPE
from tenacity import RetryError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize Neo4j connection
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Check database connection
        result = graph.query("CALL dbms.components() YIELD name, versions, edition")
        logger.info("Successfully connected to Neo4j database")
        
        # Get current counts
        node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        logger.info(f"Current database stats - Nodes: {node_count}, Relationships: {rel_count}")

        # Define file paths
        base_path = Path(root_dir) / "data" / "raw" / "umls"
        files = {
            'mrconso': base_path / "MRCONSO.RRF",
            'mrrel': base_path / "MRREL.RRF",
            'mrdef': base_path / "MRDEF.RRF",
            'mrsty': base_path / "MRSTY.RRF",
            # 'mrhier': base_path / "MRHIER.RRF"
        }
        
        # Check if files exist
        for name, path in files.items():
            if not path.exists():
                logger.error(f"Required file {name} not found at {path}")
                return

        # Initialize UMLS processor and load data
        processor = UMLSProcessor(
            graph,
            max_retries=3,
            min_wait=1,
            max_wait=10
        )
        analysis = processor.diagnose_missing_relationships(files['mrrel'])
        logger.info("Starting UMLS data processing...")
        
        # Process new additions and capture statistics
        # print("\nStarting to process new additions...")
        # stats = processor.process_new_additions()

        # # Print processing results
        # print("\nProcessing Summary:")
        # print(f"✓ Added {stats['new_concepts']:,} new concepts")
        # print(f"✓ Added {stats['new_relationships']:,} new relationships")
        # print(f"✓ Added {stats['new_semantic_types']:,} new semantic type mappings")
        # print(f"✓ Total processing time: {stats['processing_time']:.1f} seconds")
        
        # Create indexes first
        processor.create_indexes()
        processor.create_vector_index()
 
        
        # Process the dataset
        processor.process_dataset(files)
        
        # Log completion of data loading
        node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        logger.info(f"UMLS data loaded - Nodes: {node_count}, Relationships: {rel_count}")
        
        # Print current mapping
        print("Current Semantic Type Mapping:")
        processor.print_semantic_type_mapping()

        # Check current state
        print("\nCurrent State of Semantic Type Nodes:")
        processor.check_semantic_type_nodes()

        print("\nSemantic Types:")
        for type_id, name in processor.important_semantic_types.items():
            print(f"{type_id}: {name}")

        

        # Add labels to existing concepts
        processor.add_concept_labels()

        # Create clinical relationships
        # processor.create_clinical_relationships()

        
        # Verify final state
        print("\nFinal State of Semantic Type Nodes:")
        processor.check_semantic_type_nodes()
                
        # Create missing concepts and relationships
        deleted_count = processor.cleanup_duplicate_relationships()
        print(f"Cleaned up {deleted_count} duplicate relationships")
        
        results = processor.create_missing_concepts_and_relationships(
            mrrel_file=files['mrrel'],
            mrconso_file=files['mrconso'],
            mrsty_file=files['mrsty']
        )
        
        logger.info(f"""
        Processing completed:
        - Created concepts: {results['concepts_created']}
        - Created relationships: {results['relationships_created']}
        """)
        
        # Add embeddings to existing concepts
        # processor.add_embeddings_to_existing_concepts(batch_size=200) 
        
        results = processor.vector_similarity_search("diabetes mellitus", limit=10)
        for result in results:
            print(f"{result['term']} (CUI: {result['cui']}) - Score: {result['score']}")
        
        results = processor.hybrid_search("heart attack symptoms", keyword_limit=5, vector_limit=10)
        for result in results:
            print(f"{result['term']} (CUI: {result['cui']}) - Score: {result['score']} - Source: {result['source']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        logger.info("UMLS processing script completed")

if __name__ == "__main__":
    main()