# scripts/process_question.py
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processors.question_processor import QuestionProcessor  # Change this import
from src.utils.database import DatabaseConnection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database_exists(graph):
    """Check if database already has UMLS data"""
    try:
        cypher = "MATCH (c:Concept) RETURN count(c) as count"
        result = graph.query(cypher)
        concept_count = result[0]['count']
        logger.info(f"Found {concept_count} existing concepts in database")
        return concept_count > 0
    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
        return False

def verify_database_schema():
    try:
        # Initialize database connection
        db = DatabaseConnection()
        graph = db.get_connection()
        
        # Check node labels
        label_query = """
        CALL db.labels()
        YIELD label
        RETURN collect(label) as labels
        """
        labels = graph.query(label_query)
        logger.info(f"Available labels in database: {labels[0]['labels']}")
        
        # Check relationship types
        rel_query = """
        CALL db.relationshipTypes()
        YIELD relationshipType
        RETURN collect(relationshipType) as types
        """
        relationships = graph.query(rel_query)
        logger.info(f"Available relationship types: {relationships[0]['types']}")
        
        # Sample some data
        sample_query = """
        MATCH (c:Concept)
        WHERE c.term IS NOT NULL
        RETURN c.term as term, c.cui as cui
        LIMIT 5
        """
        samples = graph.query(sample_query)
        logger.info("Sample concepts:")
        for sample in samples:
            logger.info(f"Term: {sample['term']}, CUI: {sample['cui']}")
            
    except Exception as e:
        logger.error(f"Error verifying database schema: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize database connection
        db = DatabaseConnection()
        graph = db.get_connection()
        
        # Check if database already has data
        database_exists = check_database_exists(graph)
        
        # Use QuestionProcessor instead of UMLSProcessor
        processor = QuestionProcessor(graph)
        
        verify_database_schema()
        
        while True:
            question = input("\nEnter your medical question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            try:
                results = processor.process_medical_question(question)
                print(f"\nResults for: {results['question']}")
                print("-" * 50)
                
                if not results['concepts']:
                    print("No relevant medical concepts found.")
                    
                for concept in results['concepts']:
                    try:
                        print(f"\nConcept: {concept.get('term', 'Unknown term')}")
                        
                        print("\nDefinitions:")
                        definitions = concept.get('definitions', [])
                        if definitions:
                            for definition in definitions:
                                print(f"- {definition}")
                        else:
                            print("No definitions found")
                            
                        print("\nRelated Concepts:")
                        relationships = concept.get('relationships', [])
                        if relationships:
                            for rel in relationships:
                                print(f"- {rel.get('type', 'Unknown type')}: {rel.get('related_term', 'Unknown term')}")
                        else:
                            print("No relationships found")
                        print("-" * 50)
                    except Exception as e:
                        logger.error(f"Error displaying concept: {str(e)}")
                        continue
            
            except Exception as e:
                  logger.error(f"Error processing question: {str(e)}")
                  print("Sorry, there was an error processing your question. Please try again.")
                
    except Exception as e:
        logger.error(f"Error initializing processors: {str(e)}")
        raise

if __name__ == "__main__":
    main()