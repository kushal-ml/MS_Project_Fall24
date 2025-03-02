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
import json

# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
# from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from openai import OpenAI  # Use OpenAI's official client instead
import logging
from src.processors.umls_processor import UMLSProcessor
from src.processors.question_processor import QuestionProcessor
from notebooks.query_breakdown import PICOFormatter
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
        #self.pico_formatter = PICOFormatter(llm=llm_function)
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
        - text: Definition
        - semantic_type: Semantic type
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
            # Extract key terms using PICO format
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

    def _extract_key_terms(self, question: str) -> List[Dict]:
        """Extract key medical terms and categorize them by node types"""
        prompt = f"""
        Extract and categorize medical terms from this question based on these node types:
        - Anatomy: anatomical structures and locations
        - Disease: medical conditions and disorders
        - Drug: medications and therapeutic substances
        - Procedure: medical procedures and interventions
        - Symptom: clinical findings and manifestations
        - ClinicalScenario: patient context and clinical situations
        
        Question: {question}
        
        Return ONLY a JSON list with format:
        [
            {{
                "term": "term_name",
                "type": "node_type",
                "priority": 1-3 (1=highest)
            }}
        ]
        
        Rules:
        1. Assign priority 1 to main conditions/drugs in question
        2. Priority 2 to symptoms/findings
        3. Priority 3 to contextual information
        4. Only include relevant medical terms
        5. Categorize each term into exactly one node type
        6. Return ONLY the JSON array, no markdown formatting or backticks
        """
        
        response = self.llm(prompt)
        try:
            # Clean the response by removing any markdown formatting or backticks
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.split('```')[1]
            if cleaned_response.lower().startswith('json'):
                cleaned_response = cleaned_response[4:]
            cleaned_response = cleaned_response.strip()
            
            terms = json.loads(cleaned_response)
            logger.info(f"Extracted terms: {terms}")
            return terms
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            logger.error(f"JSON decode error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {str(e)}")
            return []

    def _get_concepts(self, terms: List[Dict]) -> List[Dict]:
        """Get concept information from Neo4j based on node types"""
        try:
            concepts = []
            
            # Define label mapping for different term types
            label_mapping = {
                'Disease': ['Disease'],
                'Drug': ['Drug', 'Chemical'],
                'Symptom': ['Symptom', 'Finding'],
                'Anatomy': ['Anatomy'],
                'Procedure': ['Procedure'],
                'ClinicalScenario': ['ClinicalScenario']
            }
            
            # Process terms by priority
            for priority in [1, 2, 3]:
                priority_terms = [t for t in terms if t.get('priority', 3) == priority]
                
                for term_info in priority_terms:
                    term = term_info['term']
                    term_type = term_info['type']
                    
                    # Get appropriate labels for the term type
                    labels = label_mapping.get(term_type, ['Concept'])
                    
                    query = """
                    MATCH (c)
                    WHERE any(label IN $labels WHERE label IN labels(c))
                    AND (
                        toLower(c.term) CONTAINS toLower($term)
                        OR any(syn IN c.synonyms WHERE toLower(syn) CONTAINS toLower($term))
                    )
                    WITH c
                    OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d:Definition)
                    WITH c, d
                    ORDER BY 
                        CASE 
                            WHEN toLower(c.term) = toLower($term) THEN 0
                            WHEN toLower(c.term) STARTS WITH toLower($term) THEN 1
                            WHEN toLower(c.term) CONTAINS toLower($term) THEN 2
                            ELSE 3 
                        END,
                        CASE WHEN d.source = 'UMLS' THEN 0 ELSE 1 END
                    LIMIT 2
                    RETURN DISTINCT
                        c.cui as cui,
                        c.term as term,
                        d.text as definition,
                        labels(c) as types,
                        $term_type as node_type,
                        $priority as priority
                    """
                    
                    results = self.graph.query(
                        query,
                        {
                            'term': term,
                            'labels': labels,
                            'term_type': term_type,
                            'priority': priority
                        }
                    )
                    
                    if results:
                        concepts.extend(results)
            
            logger.info(f"Found {len(concepts)} concepts")
            return concepts
            
        except Exception as e:
            logger.error(f"Error getting concepts from Neo4j: {str(e)}")
            return []

    def _get_relationships(self, concepts: List[Dict]) -> List[Dict]:
        """Get relationships between concepts based on node types"""
        try:
            if not concepts:
                return []
                
            # Get CUIs from found concepts
            cuis = [concept['cui'] for concept in concepts if 'cui' in concept]
            
            # First, get RELATES_TO relationships for all concepts
            relates_query = """
            MATCH (c1)-[r:RELATES_TO]-(c2)
            WHERE c1.cui IN $cuis
            RETURN DISTINCT
                'RELATES_TO' as relationship_type,
                c1.cui as source_cui,
                c1.term as source_name,
                c2.cui as target_cui,
                c2.term as target_name,
                labels(c1)[0] as source_type,
                labels(c2)[0] as target_type
            """
            
            related_concepts = self.graph.query(relates_query, {'cuis': cuis})
            
            # Add CUIs from related concepts to include them in relationship search
            for rel in related_concepts:
                if rel['target_cui'] not in cuis:
                    cuis.append(rel['target_cui'])
            
            # Define relationship types as before
            relationship_types = {
                'Disease': [
                    'MAY_TREAT', 'MAY_PREVENT', 'DISEASE_HAS_FINDING',
                    'HAS_CAUSATIVE_AGENT', 'OCCURS_IN', 'CAUSES',
                    'ASSOCIATED_WITH', 'DEVELOPS_INTO', 'HAS_COURSE',
                    'CLINICAL_COURSE_OF', 'CONTRAINDICATED_WITH_DISEASE'
                ],
                'Drug': [
                    'MAY_TREAT', 'MAY_PREVENT', 'HAS_MECHANISM_OF_ACTION',
                    'CHEMICAL_OR_DRUG_HAS_MECHANISM_OF_ACTION',
                    'CONTRAINDICATED_WITH_DISEASE', 'HAS_INGREDIENT',
                    'HAS_PRECISE_INGREDIENT', 'REGULATES', 
                    'POSITIVELY_REGULATES', 'NEGATIVELY_REGULATES'
                ],
                'Symptom': [
                    'IS_FINDING_OF_DISEASE', 'ASSOCIATED_FINDING_OF',
                    'MANIFESTATION_OF', 'DISEASE_HAS_FINDING',
                    'DISEASE_MAY_HAVE_FINDING', 'MAY_BE_FINDING_OF_DISEASE'
                ],
                'Anatomy': [
                    'HAS_LOCATION', 'LOCATION_OF', 'PART_OF',
                    'DRAINS_INTO', 'IS_LOCATION_OF_ANATOMIC_STRUCTURE',
                    'IS_LOCATION_OF_BIOLOGICAL_PROCESS', 'OCCURS_IN'
                ],
                'SemanticType': [
                    'IS_A', 'ASSOCIATED_WITH', 'AFFECTS',
                    'INTERACTS_WITH', 'PROCESS_OF'
                ]
            }
            
            # Get all relevant relationship types
            all_rel_types = []
            for concept in concepts:
                node_type = concept.get('node_type')
                if node_type in relationship_types:
                    all_rel_types.extend(relationship_types[node_type])
            
            all_rel_types = list(set(all_rel_types))
            
            # Query for relationships between extracted concepts only
            direct_query = """
            MATCH (c1)-[r]-(c2)
            WHERE c1.cui IN $cuis AND c2.cui IN $cuis
            AND type(r) IN $rel_types
            RETURN DISTINCT
                type(r) as relationship_type,
                c1.cui as source_cui,
                c1.term as source_name,
                c2.cui as target_cui,
                c2.term as target_name,
                labels(c1)[0] as source_type,
                labels(c2)[0] as target_type
            """
            
            direct_relationships = self.graph.query(
                direct_query,
                {
                    'cuis': cuis,
                    'rel_types': all_rel_types
                }
            )
            
            # Combine direct relationships with RELATES_TO relationships
            all_relationships = direct_relationships + related_concepts
            
            return all_relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            return []

    def _generate_answer(self, question: str, concepts: List[Dict], relationships: List[Dict]) -> str:
        """Generate answer using knowledge graph data to enhance LLM response"""
        try:
            # Format relevant knowledge graph data as context
            context = "Medical Knowledge Graph Data:\n\n"
            
            # Add relevant concepts with definitions
            context += "Relevant Medical Concepts:\n"
            for concept in concepts:
                context += f"\n• {concept.get('term')} (CUI: {concept.get('cui')})"
                if concept.get('definition'):
                    context += f"\n  Definition: {concept.get('definition')}"
                if concept.get('types'):
                    context += f"\n  Type: {', '.join(concept.get('types'))}"
                
                # Add related terms right under each concept
                related_terms = [r for r in relationships 
                               if r.get('relationship_type') == 'RELATES_TO' 
                               and r.get('source_cui') == concept.get('cui')]
                if related_terms:
                    context += "\n  Related terms: " + ", ".join(r.get('target_name') for r in related_terms)
            
            # Filter out RELATES_TO relationships for the main relationships section
            direct_relationships = [r for r in relationships if r.get('relationship_type') != 'RELATES_TO']
            
            # Add treatment relationships
            context += "\n\nTreatment Relationships:\n"
            treatment_rels = [r for r in direct_relationships if r.get('relationship_type') in 
                            ['MAY_TREAT', 'MAY_BE_TREATED_BY', 'MAY_PREVENT']]
            for rel in treatment_rels:
                context += f"\n• {rel.get('source_name')} -> {rel.get('relationship_type')} -> {rel.get('target_name')}"

            # Add mechanism relationships
            context += "\n\nMechanism Relationships:\n"
            mechanism_rels = [r for r in direct_relationships if r.get('relationship_type') in 
                            ['HAS_MECHANISM_OF_ACTION', 'CHEMICAL_OR_DRUG_HAS_MECHANISM_OF_ACTION']]
            for rel in mechanism_rels:
                context += f"\n• {rel.get('source_name')} -> {rel.get('relationship_type')} -> {rel.get('target_name')}"

            prompt = f"""You are a medical expert. Answer this question using ONLY the provided knowledge graph data. 
            Do not use external medical knowledge.

            Question: {question}

            {context}

            Provide a comprehensive answer that:
            1. Pick the correct answer from the list of choices provided
            2. Addresses the question directly
            3. Uses only the relationships and concepts shown above
            4. Cites specific concepts using their CUIs
            5. Explains any treatment recommendations using the available mechanism data
            6. States if any crucial information is missing from the knowledge graph

            Answer in a clear, professional manner."""
        
            logger.info(f"\n\nPrompt: {prompt}\n\n")
            return self.llm(prompt)
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Error: Unable to generate answer due to data processing limitations."

    def _validate_response(self, response, evidence_list):
        """Validate that response only uses provided evidence"""
        used_cuis = set()
        for evidence in evidence_list:
            used_cuis.add(evidence['source_cui'])
            used_cuis.add(evidence['target_cui'])
        
        # Add checks for any external medical terms or claims
        # that aren't supported by the evidence
        return {
            'is_valid': True/False,
            'unsupported_claims': [],
            'missing_evidence': []
        }

    def _format_evidence(self, relationship):
        """Format relationship evidence with source attribution"""
        evidence = {
            'statement': f"{relationship['source_name']} {relationship['relationship_type'].lower().replace('_', ' ')} {relationship['target_name']}",
            'source': relationship['relationship_source'],
            'confidence': relationship.get('relationship_confidence'),
            'year': relationship.get('relationship_year'),
            'source_cui': relationship['source_cui'],
            'target_cui': relationship['target_cui']
        }
        return evidence

    def _generate_response(self, relationships, question):
        """Generate response with evidence tracking"""
        evidence_list = []
        for rel in relationships:
            evidence = self._format_evidence(rel)
            evidence_list.append(evidence)
        
        response = {
            'answer': self._construct_answer(evidence_list, question),
            'evidence': evidence_list
        }
        return response

    def _construct_answer(self, evidence_list, question):
        """Construct answer using only knowledge graph evidence"""
        prompt = f"""
        Based ONLY on the following evidence from the knowledge graph, answer the question.
        Do not include any external knowledge not present in these relationships.
        If you cannot answer the question completely with the given evidence, explicitly state what information is missing.

        Question: {question}

        Available Evidence:
        {json.dumps(evidence_list, indent=2)}

        Format your response as:
        1. Answer (using only the above evidence)
        2. Missing Information (what additional knowledge graph relationships would help answer the question more completely)
        3. Evidence Used (list the specific relationships used)
        """
        # Call LLM with the prompt
        return self.llm(prompt)

def main():
    try:
        load_dotenv()
        
        # Initialize Neo4j connection with correct protocol
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j"  # Specify the database name
        )
        
        try:
            # Test the connection
            result = graph.query("RETURN 1 as test")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize processor
        processor = MedicalQuestionProcessor(
            graph=graph,
            llm_function=lambda x: client.chat.completions.create(
                model="gpt-4o",
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
                print("\n" + "="*50)
                print("MEDICAL QUESTION ANALYSIS")
                print("="*50)
                
                print("\nQUESTION:")
                print("-"*50)
                print(question)
                
                print("\nANSWER:")
                print("-"*50)
                print(result['answer'])
                
                print("\nRELEVANT CONCEPTS:")
                print("-"*50)
                # Filter duplicate concepts and show only unique ones
                seen_cuis = set()
                for concept in result['concepts']:
                    if concept.get('cui') not in seen_cuis:
                        print(f"\nConcept: {concept.get('term')}")
                        print(f"CUI: {concept.get('cui')}")
                        if concept.get('definition'):
                            print(f"Definition: {concept.get('definition')[:100]}...")
                        print(f"Type: {', '.join(concept.get('types', []))}")
                        seen_cuis.add(concept.get('cui'))
                
                print("\nKEY RELATIONSHIPS:")
                print("-"*50)
                # Filter relationships to show only treatment and direct disease relationships
                relevant_relationship_types = {
                    'MAY_TREAT',
                    'MAY_BE_TREATED_BY',
                    'CONTRAINDICATED_WITH_DISEASE',
                    'HAS_MECHANISM_OF_ACTION',
                    'CHEMICAL_OR_DRUG_HAS_MECHANISM_OF_ACTION'
                }
                
                # Get all unique terms mentioned in the question
                question_terms = {concept['term'].lower() for concept in result['concepts']}
                
                seen_relationships = set()
                for rel in result['relationships']:
                    # Only show relationships that:
                    # 1. Are of relevant types
                    # 2. Involve concepts mentioned in the question
                    # 3. Haven't been shown before
                    rel_type = rel.get('relationship_type')
                    source_name = rel.get('source_name', '').lower()
                    target_name = rel.get('target_name', '').lower()
                    
                    if (rel_type in relevant_relationship_types and 
                        (source_name in question_terms or target_name in question_terms)):
                        rel_key = f"{rel.get('source_name')} -> {rel_type} -> {rel.get('target_name')}"
                        if rel_key not in seen_relationships:
                            print(rel_key)
                            seen_relationships.add(rel_key)
                
                print("\n" + "="*50)
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        print("\nSession ended")

if __name__ == "__main__":
    main()