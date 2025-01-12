import logging
logger = logging.getLogger(__name__)
from src.processors.umls_processor import UMLSProcessor
from src.utils.database import DatabaseConnection
from src.processors.base_processor import BaseProcessor
from src.utils.term_extractor import MedicalTermExtractor

logger = logging.getLogger(__name__)

class QuestionProcessor(BaseProcessor):
    def __init__(self, graph):
        super().__init__(graph)
        self.term_extractor = MedicalTermExtractor()
        self.umls_processor = UMLSProcessor(graph)  # Create instance to use helper methods

    def create_indexes(self):
        """
        Implement required abstract method.
        Question processor doesn't need its own indexes,
        so we can just pass
        """
        pass

    def process_dataset(self, file_path: str):
        """
        Implement required abstract method.
        Question processor doesn't process datasets,
        so we can just pass
        """
        pass

    def process_medical_question(self, question: str) -> dict:
        """Process a medical question using term extraction"""
        try:
            # Extract key terms (improved version)
            stop_words = {'what', 'is', 'are', 'the', 'and', 'or', 'to', 'in', 'of', 'for'}
            terms = [word.lower().strip('?.,!') for word in question.split() 
                    if len(word) > 2 and word.lower() not in stop_words]
            
            # Also try compound terms
            compound_terms = []
            for i in range(len(terms) - 1):
                compound_terms.append(' '.join(terms[i:i+2]))
            
            all_terms = terms + compound_terms
            logger.info(f"Processing terms: {all_terms}")
            
            # Add this diagnostic query first
            diagnostic_cypher = """
            MATCH (c:Concept) 
            RETURN count(c) as count
            """
            count_result = self.graph.query(diagnostic_cypher)
            logger.info(f"Total concepts in database: {count_result[0]['count']}")
            
            # Test query for a specific term
            test_cypher = """
            MATCH (c:Concept)
            WHERE c.term IS NOT NULL
            RETURN c.term as term
            LIMIT 5
            """
            test_results = self.graph.query(test_cypher)
            logger.info(f"Sample terms in database: {[r['term'] for r in test_results]}")
            
            # Get concepts for each term
            results = []
            for term in all_terms:
                # Modified query for debugging
                cypher = """
                MATCH (c:Concept)
                WHERE c.term IS NOT NULL
                AND (
                    toLower(c.term) CONTAINS toLower($term)
                    OR toLower($term) CONTAINS toLower(c.term)
                    OR any(word IN split(toLower(c.term), ' ') WHERE word CONTAINS toLower($term))
                )
                RETURN DISTINCT c.term as term, c.cui as cui, c.domain as domain
                LIMIT 5
                """
                logger.info(f"Searching for term: {term}")
                concepts = self.graph.query(cypher, {'term': term})
                logger.info(f"Found {len(concepts)} matches for term '{term}'")
                
                for concept in concepts:
                    # Get semantic types for the concept
                    semantic_types = self.umls_processor.get_semantic_types_for_concept(concept['cui'])
            
                    concept_data = {
                        'term': concept['term'],
                        'cui': concept['cui'],
                        'semantic_types': semantic_types,
                        'definitions': [{'d': {'text': d['d']['text']}} for d in self.umls_processor._get_definitions(concept['cui'])],
                        'relationships': self.umls_processor._get_top_relationships(concept['cui'])
                    }
                    if concept_data['definitions']:
                        results.append(concept_data)

            # Remove duplicates based on CUI
            seen_cuis = set()
            unique_results = []
            for r in results:
                if r['cui'] not in seen_cuis:
                    seen_cuis.add(r['cui'])
                    unique_results.append(r)
            
            logger.info(f"Found {len(unique_results)} relevant concepts")

            print(f"\nResults for: {question}")
            print("-" * 50)

            if not unique_results:
                print("No relevant medical concepts found.")
            else:
                for concept in unique_results:
                    print(f"\nTerm: {concept['term']}")
                    
                    # Print semantic types
                    if concept['semantic_types']:
                        print("\nSemantic Types:")
                        for st in concept['semantic_types']:
                            print(f"- {st['semantic_type']}")
                    
                    if concept['definitions']:
                        print("\nDefinitions:")
                        for definition in concept['definitions']:
                            print(f"- {definition['d']['text']}")  # Updated to access definition text correctly
                    
                    if concept['relationships']:
                        print("\nRelated concepts:")
                        for rel in concept['relationships']:
                            print(f"- {rel['type']}: {rel['related_term']}")
                    print()

            return {
                'question': question,
                'concepts': unique_results
            }
            
        except Exception as e:
            logger.error(f"Error processing medical question: {str(e)}")
            raise

