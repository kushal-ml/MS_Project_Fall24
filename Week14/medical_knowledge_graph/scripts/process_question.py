import sys
from pathlib import Path
import os

# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
import logging
from src.processors.question_processor import QuestionProcessor

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
        
        # Verify database has data
        node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        if node_count == 0:
            logger.error("Database is empty. Please run process_umls.py first to load the data.")
            return
            
        # Initialize question processor
        question_processor = QuestionProcessor(graph)
        
        # Process questions
        while True:
            print("\nEnter your medical question (or 'quit' to exit): ", end='')
            question = input().strip()
            
            if question.lower() == 'quit':
                break
                
            try:
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
        logger.info("Question processing script completed")

if __name__ == "__main__":
    main()