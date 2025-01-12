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
            'mrsty': base_path / "MRSTY.RRF"
        }
        
        # Check if files exist
        for name, path in files.items():
            if not path.exists():
                logger.error(f"Required file {name} not found at {path}")
                return

        # Initialize UMLS processor and load data
        processor = UMLSProcessor(graph)
        logger.info("Starting UMLS data processing...")
        
        # Create indexes first
        processor.create_indexes()
        
        # Process the dataset
        processor.process_dataset(files)
        
        # Log completion of data loading
        node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        logger.info(f"UMLS data loaded - Nodes: {node_count}, Relationships: {rel_count}")
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        logger.info("UMLS processing script completed")

if __name__ == "__main__":
    main()