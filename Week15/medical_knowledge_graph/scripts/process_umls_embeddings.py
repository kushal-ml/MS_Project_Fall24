"""
Fixed UMLS Processor for Medical Knowledge Graph
"""

from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import time
import logging
import argparse

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

# Import processor
from src.processors.umls_processor_embeddings import UMLSProcessorEmbeddings
from src.config.constants import (CONCEPT_TIERS, RELATIONSHIP_TIERS, RELATION_TYPE_MAPPING,
                                 NEO4J_FREE_TIER_LIMITS)

# Langchain
from langchain_community.graphs import Neo4jGraph

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fixed_medical_kg_construction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build USMLE knowledge graph from UMLS data')
    parser.add_argument('--target_step', default='ALL', choices=['STEP1', 'STEP2', 'STEP3', 'ALL'], 
                        help='Target USMLE step to optimize for')
    parser.add_argument('--data_dir', required=True, help='Directory containing UMLS RRF files')
    args = parser.parse_args()
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Connect to Neo4j
        logger.info("Connecting to Neo4j database...")
        kg = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
        
        # Check database connection
        result = kg.query("CALL dbms.components() YIELD name, versions, edition")
        logger.info(f"Connected to Neo4j {result[0]['name']} {result[0]['versions']} {result[0]['edition']}")
        
        # Define file paths
        base_path = Path(args.data_dir)
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
        
        # Initialize processor
        processor = UMLSProcessorEmbeddings(
            graph=kg,
            max_retries=3,
            min_wait=1,
            max_wait=10,
            target_step=args.target_step
        )
        # create vector index
        processor.create_vector_index() 
        
        # Process MRSTY first to load semantic types
        logger.info("Loading semantic types from MRSTY...")
        processor._load_semantic_types(str(files['mrsty']))
        
        # Create indexes
        logger.info("Creating indexes...")
        processor.create_indexes()
        
        
        # Process MRCONSO
        logger.info("Processing concepts from MRCONSO...")
        concepts_processed = processor._process_mrconso_parallel(str(files['mrconso']))
        logger.info(f"Processed {concepts_processed} concepts")
        
        # Load processed concepts
        logger.info("Loading processed concepts...")
        processor._load_processed_concepts()
        
        # Process semantic types
        logger.info("Processing semantic types...")
        semantic_types_processed = processor.process_mrsty(str(files['mrsty']))
        logger.info(f"Processed {semantic_types_processed} semantic types")

        # Process definitions
        logger.info("Processing definitions...")
        definitions_processed = processor._process_mrdef_parallel(str(files['mrdef']))
        logger.info(f"Processed {definitions_processed} definitions")
        
        # Add labels
        logger.info("Adding concept labels...")
        labeled_concepts = processor.add_concept_labels()
        logger.info(f"Added labels to {labeled_concepts} concepts")
        
        
        # Process relationships
        logger.info("Processing relationships...")
        relationships_processed = processor.process_mrrel(str(files['mrrel']))
        logger.info(f"Processed {relationships_processed} relationships")
        

        # Add embeddings
        logger.info("Adding embeddings...")
        embeddings_added = processor.add_embeddings_to_existing_concepts()
        logger.info(f"Added embeddings to {embeddings_added} concepts")
        #Print statistics
        processor.print_stats()
        
        # Log completion
        node_count = kg.query("MATCH (n) RETURN count(n) as count")[0]['count']
        rel_count = kg.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        logger.info(f"Final database stats - Nodes: {node_count}, Relationships: {rel_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        logger.info("Fixed UMLS processing completed")

if __name__ == "__main__":
    main()