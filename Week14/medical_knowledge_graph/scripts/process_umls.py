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
from src.processors.question_processor import QuestionProcessor
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_files_exist(file_paths):
    """Check if all required files exist"""
    for name, path in file_paths.items():
        if not path.exists():
            logger.error(f"Required file {name} not found at {path}")
            return False
    return True

def check_database_connection(graph):
    """Verify database connection and get current stats"""
    try:
        # Check connection
        result = graph.query("CALL dbms.components() YIELD name, versions, edition")
        logger.info("Successfully connected to Neo4j database")
        
        # Get current counts
        node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        
        logger.info(f"Current database stats - Nodes: {node_count}, Relationships: {rel_count}")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

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
        if not check_database_connection(graph):
            logger.error("Failed to connect to database. Exiting...")
            return
        
        # Define file paths
        base_path = Path(root_dir) / "data" / "raw" / "umls"
        files = {
            'mrconso': base_path / "MRCONSO.RRF",
            'mrrel': base_path / "MRREL.RRF",
            'mrdef': base_path / "MRDEF.RRF",
            'mrsty': base_path / "MRSTY.RRF"
        }
        
        # Check if files exist
        if not check_files_exist(files):
            logger.error("Required files missing. Please check the data directory.")
            return
        
        # Initialize processor
        processor = UMLSProcessor(graph)
        question_processor = QuestionProcessor(graph)  # Add this line
        
        while True:
            # Get user input
            print("\nEnter your medical question (or 'quit' to exit): ", end='')
            question = input().strip()
            
            if question.lower() == 'quit':
                break
                
            try:
                # Use question_processor instead of processor
                results = question_processor.process_medical_question(question)
                
                print(f"\nResults for: {question}")
                print("-" * 50)
                
                if not results['concepts']:
                    print("No relevant medical concepts found.")
                else:
                    for concept in results['concepts']:
                        print(f"\nTerm: {concept['term']}")
                        if concept['definitions']:
                            print("\nDefinitions:")
                            for definition in concept['definitions']:
                                print(f"- {definition}")
                        if concept['relationships']:
                            print("\nRelated concepts:")
                            for rel in concept['relationships']:
                                print(f"- {rel['type']}: {rel['related_term']}")
                        print()
                        
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print("Sorry, there was an error processing your question. Please try again.")
                
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        # Log completion
        logger.info("UMLS processing script completed")

if __name__ == "__main__":
    main()