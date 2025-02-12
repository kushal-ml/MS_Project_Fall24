# import sys
# from pathlib import Path
# import os

# # Add the project root directory to Python path
# root_dir = str(Path(__file__).parent.parent)
# sys.path.append(root_dir)

# from dotenv import load_dotenv
# from langchain_community.graphs import Neo4jGraph
# import logging
# from src.processors.question_processor import QuestionProcessor

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def main():
#     try:
#         # Load environment variables
#         load_dotenv()
        
#         # Initialize Neo4j connection
#         graph = Neo4jGraph(
#             url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
#             username=os.getenv("NEO4J_USERNAME", "neo4j"),
#             password=os.getenv("NEO4J_PASSWORD")
#         )
        
#         # Check database connection
#         result = graph.query("CALL dbms.components() YIELD name, versions, edition")
#         logger.info("Successfully connected to Neo4j database")
        
#         # Verify database has data
#         node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
#         if node_count == 0:
#             logger.error("Database is empty. Please run process_umls.py first to load the data.")
#             return
            
#         # Initialize question processor
#         question_processor = QuestionProcessor(graph)
        
#         # Process questions
#         while True:
#             print("\nEnter your medical question (or 'quit' to exit): ", end='')
#             question = input().strip()
            
#             if question.lower() == 'quit':
#                 break
                
#             try:
#                 results = question_processor.process_medical_question(question)
                
#                 print(f"\nResults for: {question}")
#                 print("-" * 50)
                
#                 if not results['concepts']:
#                     print("No relevant medical concepts found.")
#                 else:
#                     for concept in results['concepts']:
#                         print(f"\nTerm: {concept['term']}")
#                         if concept['definitions']:
#                             print("\nDefinitions:")
#                             for definition in concept['definitions']:
#                                 print(f"- {definition}")
#                         if concept['relationships']:
#                             print("\nRelated concepts:")
#                             for rel in concept['relationships']:
#                                 print(f"- {rel['type']}: {rel['related_term']}")
#                         print()
                        
#             except Exception as e:
#                 logger.error(f"Error processing question: {str(e)}")
#                 print("Sorry, there was an error processing your question. Please try again.")
                
#     except Exception as e:
#         logger.error(f"Fatal error: {str(e)}")
#         raise
#     finally:
#         logger.info("Question processing script completed")

# if __name__ == "__main__":
#     main()

import sys
from pathlib import Path
import os
from typing import Dict, List, Any

# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from openai import OpenAI  # Use OpenAI's official client instead
import logging
from src.processors.umls_processor import UMLSProcessor
from src.processors.question_processor import QuestionProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalQuestionProcessor:
    def __init__(self, graph: Neo4jGraph, llm_function):
        self.graph = graph
        self.llm = llm_function
        logger.info("Medical Question Processor initialized")
        
        # Define UMLS schema and question types
        self.umls_schema = """
        You are an expert in medical knowledge graphs, specifically the UMLS (Unified Medical Language System) knowledge graph.

        UMLS Knowledge Graph Schema:

        1. Node Types (Labels):
        - Disease: Represents medical conditions and disorders
        - Drug: Represents medications and therapeutic substances
        - Symptom: Represents clinical findings and manifestations
        - Anatomy: Represents anatomical structures and locations
        - SemanticType: Represents semantic classifications of medical concepts
        - Definition: Represents concept definitions
   

        2. Key Relationships:
        a) Disease-Related:
        - Drug
        - DISEASE_HAS_FINDING -> Symptom
        - DISEASE_MAY_HAVE_FINDING -> Symptom
        - HAS_CAUSATIVE_AGENT
        - OCCURS_IN -> Anatomy

        b) Drug-Related:
        - HAS_MECHANISM_OF_ACTION
        - CHEMICAL_OR_DRUG_HAS_MECHANISM_OF_ACTION
        - HAS_INGREDIENT
        - HAS_PRECISE_INGREDIENT
        - CONTRAINDICATED_WITH_DISEASE -> Disease

        c) Symptom-Related:
        - IS_FINDING_OF_DISEASE -> Disease
        - ASSOCIATED_FINDING_OF
        - MAY_BE_FINDING_OF_DISEASE
        - MANIFESTATION_OF

        d) Anatomical:
        - HAS_LOCATION
        - LOCATION_OF
        - PART_OF
        - DRAINS_INTO

        e) General Clinical:
        - ASSOCIATED_WITH
        - CAUSES
        - OCCURS_BEFORE
        - REGULATES
        - POSITIVELY_REGULATES
        - NEGATIVELY_REGULATES

        Properties:
        - cui: Concept Unique Identifier
        - term: Preferred term
        - source: Source vocabulary
        - domain: Medical domain
        - created_at: Timestamp
        """
        
        self.question_types = {
            "diagnosis": {
                "pattern": "symptoms -> disease",
                "key_relationships": [
                    "IS_FINDING_OF_DISEASE",
                    "DISEASE_HAS_FINDING",
                    "DISEASE_MAY_HAVE_FINDING",
                    "MAY_BE_FINDING_OF_DISEASE",
                    "ASSOCIATED_FINDING_OF",
                    "MANIFESTATION_OF",
                    "CLINICAL_COURSE_OF"
                ],
                "example_query": """Example: To find diseases related to symptoms:
                    // For multiple symptoms
                    MATCH (d:Disease)-[r]-(s:Symptom)
                    WHERE s.term IN ['Fever', 'Cough']
                    AND type(r) IN ['IS_FINDING_OF_DISEASE', 'DISEASE_HAS_FINDING', 
                                   'DISEASE_MAY_HAVE_FINDING', 'MAY_BE_FINDING_OF_DISEASE']
                    WITH d, collect(DISTINCT s.term) as symptoms, count(DISTINCT s) as symptomCount
                    WHERE symptomCount >= 2
                    RETURN d.term AS Disease, 
                           symptomCount AS MatchingSymptoms,
                           symptoms AS SymptomList
                    ORDER BY symptomCount DESC
                    LIMIT 5"""
            },
            
            "treatment": {
                "pattern": "disease -> treatment",
                "key_relationships": [
                    "MAY_BE_TREATED_BY",
                    "MAY_TREAT",
                    "MAY_PREVENT",
                    "HAS_INGREDIENT",
                    "HAS_PRECISE_INGREDIENT",
                    "CONTRAINDICATED_WITH_DISEASE"
                ],
                "example_query": """Example: To find treatments for a disease:
                    // Find treatments and their mechanisms
                    MATCH (d:Disease)-[r1]-(drug:Drug)
                    WHERE d.term = 'Type 2 Diabetes'
                    AND type(r1) IN ['MAY_BE_TREATED_BY', 'MAY_TREAT']
                    OPTIONAL MATCH (drug)-[r2]->(m)
                    WHERE type(r2) = 'HAS_MECHANISM_OF_ACTION'
                    WITH drug, collect(DISTINCT m.term) as mechanisms
                    OPTIONAL MATCH (drug)-[r3]->(contra:Disease)
                    WHERE type(r3) = 'CONTRAINDICATED_WITH_DISEASE'
                    RETURN drug.term AS Treatment, 
                           mechanisms AS Mechanisms,
                           collect(DISTINCT contra.term) as Contraindications"""
            },
            
            "mechanism": {
                "pattern": "drug -> mechanism",
                "key_relationships": [
                    "HAS_MECHANISM_OF_ACTION",
                    "CHEMICAL_OR_DRUG_HAS_MECHANISM_OF_ACTION",
                    "MECHANISM_OF_ACTION_OF",
                    "CHEMICAL_OR_DRUG_AFFECTS_GENE_PRODUCT",
                    "REGULATES",
                    "POSITIVELY_REGULATES",
                    "NEGATIVELY_REGULATES"
                ],
                "example_query": """Example: To find mechanism of action for a drug:
                    // Find direct mechanisms and regulatory effects
                    MATCH (drug:Drug)-[r]->(m)
                    WHERE drug.term = 'Metformin'
                    AND type(r) IN ['HAS_MECHANISM_OF_ACTION', 
                                  'CHEMICAL_OR_DRUG_HAS_MECHANISM_OF_ACTION',
                                  'REGULATES', 'POSITIVELY_REGULATES', 'NEGATIVELY_REGULATES']
                    RETURN drug.term AS Drug,
                           collect(DISTINCT {
                               mechanism: m.term,
                               type: type(r),
                               effect: CASE 
                                   WHEN type(r) = 'POSITIVELY_REGULATES' THEN 'increases'
                                   WHEN type(r) = 'NEGATIVELY_REGULATES' THEN 'decreases'
                                   ELSE 'affects'
                               END
                           }) as Mechanisms"""
            },
            
            "anatomical": {
                "pattern": "anatomy -> structure/location",
                "key_relationships": [
                    "HAS_LOCATION",
                    "LOCATION_OF",
                    "IS_LOCATION_OF_ANATOMIC_STRUCTURE",
                    "IS_LOCATION_OF_BIOLOGICAL_PROCESS",
                    "PART_OF",
                    "DRAINS_INTO",
                    "OCCURS_IN"
                ],
                "example_query": """Example: To find anatomical relationships:
                    // Find structural and functional relationships
                    MATCH (a:Anatomy)-[r]-(related:Anatomy)
                    WHERE a.term = 'Heart'
                    AND type(r) IN ['PART_OF', 'HAS_LOCATION', 'DRAINS_INTO']
                    WITH a, related, r
                    OPTIONAL MATCH (related)-[r2]->(process)
                    WHERE type(r2) = 'IS_LOCATION_OF_BIOLOGICAL_PROCESS'
                    RETURN a.term AS Structure,
                           collect(DISTINCT {
                               related_structure: related.term,
                               relationship: type(r),
                               processes: collect(DISTINCT process.term)
                           }) as AnatomicalRelations"""
            },
            
            "clinical_course": {
                "pattern": "disease -> progression",
                "key_relationships": [
                    "HAS_COURSE",
                    "DEVELOPS_INTO",
                    "CLINICAL_COURSE_OF",
                    "CAUSE_OF",
                    "HAS_CAUSATIVE_AGENT",
                    "OCCURS_BEFORE"
                ],
                "example_query": """Example: To find disease progression:
                    // Find disease progression and complications
                    MATCH (d:Disease)-[r]->(outcome)
                    WHERE d.term = 'Type 2 Diabetes'
                    AND type(r) IN ['HAS_COURSE', 'DEVELOPS_INTO', 'CLINICAL_COURSE_OF']
                    WITH d, outcome, r
                    OPTIONAL MATCH (outcome)-[r2]->(complication:Disease)
                    WHERE type(r2) = 'CAUSE_OF'
                    RETURN d.term AS Disease,
                           collect(DISTINCT {
                               stage: outcome.term,
                               progression_type: type(r),
                               complications: collect(DISTINCT complication.term)
                           }) as DiseaseProgression
                    ORDER BY outcome.term"""
            },
            
            "drug_interaction": {
                "pattern": "drug -> drug/disease interactions",
                "key_relationships": [
                    "CONTRAINDICATED_WITH_DISEASE",
                    "CHEMICAL_OR_DRUG_AFFECTS_GENE_PRODUCT",
                    "HAS_MECHANISM_OF_ACTION",
                    "CONTRAINDICATED_MECHANISM_OF_ACTION_OF",
                    "HAS_INGREDIENT",
                    "HAS_PRECISE_INGREDIENT",
                    "MAY_TREAT",
                    "MAY_PREVENT"
                ],
                "example_query": """Example: To find drug interactions:
                    // Find drug interactions and contraindications
                    MATCH (drug1:Drug)
                    WHERE drug1.term = 'Warfarin'
                    OPTIONAL MATCH (drug1)-[r1]->(d:Disease)
                    WHERE type(r1) = 'CONTRAINDICATED_WITH_DISEASE'
                    WITH drug1, collect(DISTINCT d.term) as contraindications
                    OPTIONAL MATCH (drug1)-[r2]->(g)
                    WHERE type(r2) = 'CHEMICAL_OR_DRUG_AFFECTS_GENE_PRODUCT'
                    RETURN drug1.term AS Drug,
                           contraindications AS Contraindications,
                           collect(DISTINCT {
                               target: g.term,
                               interaction_type: type(r2)
                           }) as GeneInteractions"""
            },
            
            "gene_disease": {
                "pattern": "gene -> disease associations",
                "key_relationships": [
                    "GENE_ASSOCIATED_WITH_DISEASE",
                    "CHEMICAL_OR_DRUG_AFFECTS_GENE_PRODUCT",
                    "ASSOCIATED_WITH",
                    "REGULATES",
                    "POSITIVELY_REGULATES",
                    "NEGATIVELY_REGULATES"
                ],
                "example_query": """Example: To find gene-disease associations:
                    // Find genetic associations and regulatory relationships
                    MATCH (g)-[r]->(d:Disease)
                    WHERE d.term = 'Cystic Fibrosis'
                    AND type(r) IN ['GENE_ASSOCIATED_WITH_DISEASE', 'ASSOCIATED_WITH']
                    WITH g, d, r
                    OPTIONAL MATCH (g)-[r2]->(regulated)
                    WHERE type(r2) IN ['REGULATES', 'POSITIVELY_REGULATES', 'NEGATIVELY_REGULATES']
                    RETURN d.term AS Disease,
                           collect(DISTINCT {
                               gene: g.term,
                               association_type: type(r),
                               regulated_elements: collect(DISTINCT {
                                   element: regulated.term,
                                   regulation_type: type(r2)
                               })
                           }) as GeneticFactors"""
            }
        }

    def process_medical_question(self, question: str) -> Dict[str, Any]:
        """Process a medical question using Neo4j knowledge graph"""
        try:
            # Extract key terms using LLM
            key_terms = self._extract_key_terms(question)
            logger.info(f"Extracted terms: {key_terms}")
            
            # Get concept information from Neo4j
            concepts = self._get_concepts(key_terms)
            logger.info(f"Found {len(concepts)} concepts")
            
            # Get relationships between concepts
            relationships = self._get_relationships(concepts)
            logger.info(f"Found {len(relationships)} relationships")
            
            # Generate answer using only Neo4j data
            answer = self._generate_answer(question, concepts, relationships)
            
            return {
                'answer': answer,
                'concepts': concepts,
                'relationships': relationships
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {'error': str(e)}

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key medical terms from the question using LLM"""
        prompt = f"""
        Extract key medical terms from this question:
        {question}
        Return only the medical terms as a comma-separated list.
        """
        response = self.llm(prompt)
        return [term.strip() for term in response.split(',')]

    def _get_concepts(self, terms: List[str]) -> List[Dict]:
        """Get concept information from Neo4j with correct property names"""
        try:
            concepts = []
            for term in terms:
                # Updated query with correct property names
                query = """
                MATCH (c)
                WHERE (c:Disease OR c:Drug OR c:Symptom OR c:Anatomy OR c:Concept)
                AND (
                    toLower(c.term) CONTAINS toLower($term)
                    OR any(syn IN c.synonyms WHERE toLower(syn) CONTAINS toLower($term))
                )
                OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d:Definition)
                OPTIONAL MATCH (c)-[:HAS_SEMANTIC_TYPE]->(s:SemanticType)
                RETURN 
                    c.cui as cui,
                    c.term as name,
                    d.text as definition,
                    labels(c) as types,
                    s.semantic_type as semantic_type,
                    c.source as source
                LIMIT 5
                """
                results = self.graph.query(query, {'term': term})
                concepts.extend(results)
                
            logger.info(f"Found {len(concepts)} concepts for terms: {terms}")
            return concepts
            
        except Exception as e:
            logger.error(f"Error getting concepts from Neo4j: {str(e)}")
            return []

    def _get_relationships(self, concepts: List[Dict]) -> List[Dict]:
        """Get relationships between concepts with correct property names"""
        try:
            relationships = []
            cuis = [concept['cui'] for concept in concepts if 'cui' in concept]
            
            if not cuis:
                return []
                
            # Updated relationship query with semantic type relationship
            query = """
            MATCH (c1)-[r]-(c2)
            WHERE c1.cui IN $cuis
            AND type(r) IN [
                'ASSOCIATED_WITH', 'CAUSE_OF', 'CHEMICAL_OR_DRUG_AFFECTS_GENE_PRODUCT',
                'DISEASE_HAS_FINDING', 'DISEASE_MAY_HAVE_FINDING',
                'HAS_MECHANISM_OF_ACTION', 'MAY_TREAT', 'MAY_PREVENT',
                'IS_FINDING_OF_DISEASE', 'MANIFESTATION_OF'
            ]
            OPTIONAL MATCH (c1)-[:HAS_SEMANTIC_TYPE]->(s1:SemanticType)
            OPTIONAL MATCH (c2)-[:HAS_SEMANTIC_TYPE]->(s2:SemanticType)
            RETURN 
                type(r) as relationship_type,
                c1.cui as source_cui,
                c1.term as source_name,
                c2.cui as target_cui,
                c2.term as target_name,
                s1.semantic_type as source_semantic_type,
                s2.semantic_type as target_semantic_type,
                r.source as relationship_source
            ORDER BY type(r)
            LIMIT 20
            """
            results = self.graph.query(query, {'cuis': cuis})
            relationships.extend(results)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships from Neo4j: {str(e)}")
            return []

    def _generate_answer(self, question: str, concepts: List[Dict], relationships: List[Dict]) -> str:
        """Generate answer using knowledge graph data to enhance LLM response"""
        try:
            # Format relevant knowledge graph data as context
            context = "Medical Knowledge Graph Context:\n\n"
            
            # Add relevant concepts with definitions
            context += "Related Medical Concepts:\n"
            for concept in concepts[:5]:
                context += f"\n• {concept.get('name')} (CUI: {concept.get('cui')})"
                if concept.get('definition'):
                    context += f"\n  Definition: {concept.get('definition')}"
                if concept.get('semantic_type'):
                    context += f"\n  Type: {concept.get('semantic_type')}"
            
            # Add relevant relationships
            context += "\n\nKnown Medical Relationships:\n"
            seen = set()
            for rel in relationships[:10]:
                rel_key = f"{rel.get('source_name')} -> {rel.get('relationship_type')} -> {rel.get('target_name')}"
                if rel_key not in seen:
                    context += f"\n• {rel_key}"
                    seen.add(rel_key)

            prompt = f"""You are a medical expert with access to a medical knowledge graph. 
            Use the following knowledge graph data to help inform your answer, but you can also 
            use your medical knowledge to provide a complete response.

            Question: {question}

            {context}

            Please provide a comprehensive answer that:
            1. Uses the knowledge graph data where relevant
            2. Cites specific concepts (with CUIs) when referencing graph data
            3. Adds medical context and explanations as needed
            4. Clearly distinguishes between graph-based and general medical knowledge

            Answer the question in a clear, professional manner."""
            
            return self.llm(prompt)
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Error: Unable to generate answer due to data processing limitations."

def main():
    try:
        load_dotenv()
        
        # Initialize connections
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize processor
        processor = MedicalQuestionProcessor(
            graph=graph,
            llm_function=lambda x: client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": x}]
            ).choices[0].message.content
        )
        
        while True:
            print("\nEnter your medical question (or 'quit' to exit): ")
            question = input().strip()
            
            if question.lower() == 'quit':
                break
                
            print("\nProcessing question...")
            result = processor.process_medical_question(question)
            
            if 'error' in result:
                print(f"\nError: {result['error']}")
            else:
                # Display answer in a structured format
                print("\n" + "="*50)
                print("MEDICAL QUESTION ANALYSIS")
                print("="*50)
                
                print("\nQUESTION:")
                print("-"*50)
                print(question)
                
                print("\nANSWER:")
                print("-"*50)
                answer_parts = result['answer'].split('\n')
                for part in answer_parts:
                    print(part)
                
                print("\nRELEVANT CONCEPTS:")
                print("-"*50)
                for concept in result['concepts'][:5]:  # Limit to top 5 concepts
                    print(f"\nConcept: {concept.get('name')}")
                    print(f"CUI: {concept.get('cui')}")
                    if concept.get('definition'):
                        print(f"Definition: {concept.get('definition')[:100]}...")
                    print(f"Type: {', '.join(concept.get('types', []))}")
                
                print("\nKEY RELATIONSHIPS:")
                print("-"*50)
                seen_relationships = set()
                count = 0
                for rel in result['relationships']:
                    if count >= 10:  # Limit to top 10 relationships
                        break
                    rel_key = f"{rel.get('source_name')} -> {rel.get('relationship_type')} -> {rel.get('target_name')}"
                    if rel_key not in seen_relationships:
                        print(rel_key)
                        seen_relationships.add(rel_key)
                        count += 1
                
                print("\n" + "="*50)
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        print("\nSession ended")

if __name__ == "__main__":
    main()