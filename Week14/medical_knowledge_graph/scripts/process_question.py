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
            logger.info("Generated answer from LLM")
            
            # Return properly formatted response with answer as string
            return {
                'answer': str(answer),  # Ensure answer is treated as string
                'key_terms': [term['term'] for term in key_terms] if key_terms else [],
                'concepts': concepts if concepts else [],
                'relationships': relationships if relationships else []
            }
            
        except Exception as e:
            logger.error(f"Error in UMLS processing: {str(e)}")
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
                logger.info("No concepts provided to find relationships")
                return []
                
            # Get CUIs from found concepts
            cuis = [concept['cui'] for concept in concepts if 'cui' in concept]
            logger.info(f"Searching relationships for CUIs: {cuis}")
            
            # More comprehensive relationship query
            query = """
            // Direct relationships (outgoing)
            MATCH (c1)-[r1]->(c2)
            WHERE c1.cui IN $cuis
            WITH COLLECT({
                relationship_type: type(r1),
                direction: 'outgoing',
                source_cui: c1.cui,
                source_name: c1.term,
                target_cui: c2.cui,
                target_name: CASE 
                    WHEN type(r1) = 'HAS_SEMANTIC_TYPE' THEN c2.semantic_type
                    WHEN type(r1) = 'HAS_DEFINITION' THEN c2.term
                    ELSE c2.term
                END,
                source_labels: labels(c1),
                target_labels: labels(c2)
            }) as outgoing

            // Direct relationships (incoming)
            MATCH (c2)-[r2]->(c1)
            WHERE c1.cui IN $cuis
            WITH outgoing, COLLECT({
                relationship_type: type(r2),
                direction: 'incoming',
                source_cui: c2.cui,
                source_name: c2.term,
                target_cui: c1.cui,
                target_name: c1.term,
                source_labels: labels(c2),
                target_labels: labels(c1)
            }) as incoming

            // Bidirectional relationships between found concepts
            MATCH (c1)-[r3]-(c2)
            WHERE c1.cui IN $cuis AND c2.cui IN $cuis AND c1.cui <> c2.cui
            WITH outgoing, incoming, COLLECT({
                relationship_type: type(r3),
                direction: 'bidirectional',
                source_cui: c1.cui,
                source_name: c1.term,
                target_cui: c2.cui,
                target_name: c2.term,
                source_labels: labels(c1),
                target_labels: labels(c2)
            }) as between

            // Two-hop relationships (outgoing-outgoing)
            MATCH (c1)-[r4]->(intermediate1)-[r5]->(c2)
            WHERE c1.cui IN $cuis AND c2.cui IN $cuis 
            AND c1.cui <> c2.cui
            AND NOT (c1)-[]-(c2)
            WITH outgoing, incoming, between, COLLECT({
                relationship_type: type(r4) + '_via_' + type(r5),
                direction: 'outgoing_chain',
                source_cui: c1.cui,
                source_name: c1.term,
                target_cui: c2.cui,
                target_name: c2.term,
                intermediate_name: intermediate1.term,
                source_labels: labels(c1),
                target_labels: labels(c2)
            }) as twohop_out

            // Two-hop relationships (incoming-incoming)
            MATCH (c2)<-[r6]-(intermediate2)<-[r7]-(c1)
            WHERE c1.cui IN $cuis AND c2.cui IN $cuis 
            AND c1.cui <> c2.cui
            AND NOT (c1)-[]-(c2)
            WITH outgoing, incoming, between, twohop_out, COLLECT({
                relationship_type: type(r7) + '_via_' + type(r6),
                direction: 'incoming_chain',
                source_cui: c1.cui,
                source_name: c1.term,
                target_cui: c2.cui,
                target_name: c2.term,
                intermediate_name: intermediate2.term,
                source_labels: labels(c1),
                target_labels: labels(c2)
            }) as twohop_in

            // Mixed direction two-hop relationships
            MATCH (c1)-[r8]-(intermediate3)-[r9]-(c2)
            WHERE c1.cui IN $cuis AND c2.cui IN $cuis 
            AND c1.cui <> c2.cui
            AND NOT (c1)-[]-(c2)
            WITH outgoing, incoming, between, twohop_out, twohop_in, COLLECT({
                relationship_type: type(r8) + '_via_' + type(r9),
                direction: 'mixed',
                source_cui: c1.cui,
                source_name: c1.term,
                target_cui: c2.cui,
                target_name: c2.term,
                intermediate_name: intermediate3.term,
                source_labels: labels(c1),
                target_labels: labels(c2)
            }) as mixed_hops

            // Combine all relationships
            UNWIND outgoing + incoming + between + twohop_out + twohop_in + mixed_hops as result
            RETURN DISTINCT 
                   result.relationship_type as relationship_type,
                   result.direction as direction,
                   result.source_cui as source_cui,
                   result.source_name as source_name,
                   result.target_cui as target_cui,
                   result.target_name as target_name,
                   result.source_labels as source_labels,
                   result.target_labels as target_labels,
                   result.intermediate_name as intermediate_name
            """
            
            try:
                relationships = self.graph.query(query, {'cuis': cuis})
                logger.info(f"Found {len(relationships)} total relationships")
                
                # Log relationship details for debugging
                for rel in relationships:
                    source = rel.get('source_name', 'Unknown')
                    rel_type = rel.get('relationship_type', 'Unknown')
                    target = rel.get('target_name', 'Unknown')
                    intermediate = rel.get('intermediate_name')
                    
                    if intermediate:
                        logger.info(f"Relationship: {source} --[{rel_type}]--> via {intermediate} --> {target}")
                    else:
                        logger.info(f"Relationship: {source} --[{rel_type}]--> {target}")
                
                return relationships
                
            except Exception as e:
                logger.error(f"Neo4j query error: {str(e)}")
                logger.error("Query failed, trying fallback query...")
                return []
            
        except Exception as e:
            logger.error(f"Error in _get_relationships: {str(e)}")
            logger.error(f"Stack trace: ", exc_info=True)
            return []

    def _generate_answer(self, question: str, concepts: List[Dict], relationships: List[Dict]) -> str:
        """Generate an answer using LLM with concepts and relationships"""
        try:
            # Format concepts for display
            context = "MEDICAL QUESTION ANALYSIS\n"
            context += "=" * 80 + "\n\n"
            
            # Add the question
            context += "QUESTION:\n"
            context += "-" * 50 + "\n"
            context += f"{question}\n\n"
            
            # Add extracted terms
            context += "EXTRACTED TERMS:\n"
            context += "-" * 50 + "\n"
            for concept in concepts:
                term_type = concept.get('term_type', 'Unknown')
                context += f"• {concept.get('name', 'Unknown Term')} ({term_type})\n"
            context += "\n"
            
            # Add concepts with definitions
            context += "RETRIEVED CONCEPTS:\n"
            context += "-" * 50 + "\n\n"
            
            # Track seen CUIs to avoid duplicates
            seen_cuis = set()
            
            for concept in concepts:
                cui = concept.get('cui')
                if cui and cui not in seen_cuis:
                    seen_cuis.add(cui)
                    
                    # Add concept name and CUI
                    context += f"• {concept.get('name', 'Unknown')} (CUI: {cui})\n"
                    
                    # Add definition if available
                    definition = concept.get('definition', '')
                    if definition:
                        context += f"  Definition: {definition}\n"
                    
                    context += "\n"
            
            # Add relationships section
            context += "RELEVANT RELATIONSHIPS:\n"
            context += "-" * 50 + "\n"
            
            # Group relationships by type
            relationship_groups = {}
            for rel in relationships:
                rel_type = rel.get('relationship_type', 'Unknown')
                if rel_type not in relationship_groups:
                    relationship_groups[rel_type] = []
                relationship_groups[rel_type].append(rel)
            
            # Display relationships by group
            for rel_type, rels in sorted(relationship_groups.items()):
                if rel_type in ['HAS_SEMANTIC_TYPE', 'HAS_DEFINITION']:
                    continue  # Skip semantic type and definition relationships
                
                context += f"\n• {rel_type} Relationships:\n"
                for rel in rels[:5]:  # Limit to 5 relationships per type to avoid overwhelming
                    source = rel.get('source_name', 'Unknown')
                    target = rel.get('target_name', 'Unknown')
                    direction = rel.get('direction', 'outgoing')
                    
                    if 'intermediate_node' in rel:
                        intermediate = rel.get('intermediate_node', {}).get('name', 'Unknown')
                        context += f"  - {source} → {intermediate} → {target} ({direction})\n"
                    else:
                        context += f"  - {source} → {target} ({direction})\n"
                
                if len(rels) > 5:
                    context += f"  - ... and {len(rels) - 5} more {rel_type} relationships\n"
            
            context += "\n"
            
            # Add analysis and answer section
            context += "ANALYSIS AND ANSWER:\n"
            context += "-" * 50 + "\n"
            
            # Create the prompt for the LLM
            prompt = f"""You are a medical expert analyzing a question using a knowledge graph. 
            Focus on the definitions and relationships provided to give a precise answer.

            Question: {question}

            {context}

            Instructions for answering:
            1. Start by identifying the key concepts relevant to the question
            2. Use the detailed definitions provided to understand each concept thoroughly
            3. Examine ALL relationships between concepts, including:
               a) Disease-Related:
                  - Disease-Drug relationships
                  - Disease-Symptom findings
                  - Causative agents
                  - Anatomical locations
               b) Drug-Related:
                  - Mechanisms of action
                  - Chemical/drug ingredients
                  - Contraindications
               c) Symptom-Related:
                  - Disease findings
                  - Associated findings
                  - Manifestations
               d) Anatomical:
                  - Locations and structures
                  - Part-whole relationships
                  - Drainage pathways
               e) General Clinical:
                  - Associations
                  - Causal relationships
                  - Temporal sequences
                  - Regulatory effects
               f) Semantic and Definitional:
                  - Semantic types
                  - Definitions
                  - Synonyms and related terms

            4. If this is a multiple choice question:
               a) Directly state which answer option is correct
               b) Explain why this option is correct using concepts and relationships
               c) Briefly explain why other options are incorrect

            5. Structure your response with these sections:
               - RELEVANT RELATIONSHIPS: List key relationships that inform your answer
               - KEY CONCEPTS USED: Identify the most important concepts and their definitions
               - DEDUCTION PROCESS: Show your reasoning step by step
               - FINAL ANSWER: State your conclusion clearly
               - Include your confidence level and any limitations in the knowledge graph

            Remember to cite specific concepts by their CUI when explaining your reasoning.
            """
            
            # Generate answer using LLM
            answer = self.llm(prompt)
            
            # Combine the context and answer
            full_response = context + answer
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

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
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical expert analyzing questions using a knowledge graph."},
                    {"role": "user", "content": x}
                ],
                temperature=0.0
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
                        
                        # Add synonyms
                        synonyms = [r.get('target_name') for r in result['relationships'] 
                                  if r.get('relationship_type') == 'RELATED_TO' 
                                  and r.get('source_cui') == concept.get('cui')
                                  and r.get('target_name') != concept.get('term')]
                        if synonyms:
                            print(f"Synonyms: {', '.join(synonyms)}")
                        seen_cuis.add(concept.get('cui'))
                
                print("\nCLINICAL RELATIONSHIPS:")
                print("-"*50)
                
                # Group relationships by type (excluding RELATES_TO)
                rel_by_type = {}
                for rel in result['relationships']:
                    if not rel or not isinstance(rel, dict):
                        continue
                        
                    rel_type = rel.get('relationship_type')
                    if rel_type == 'RELATED_TO':  # Skip synonyms as they're shown with concepts
                        continue
                        
                    source = rel.get('source_name', '')
                    target = rel.get('target_name', '')
                    intermediate = rel.get('intermediate_name')
                    
                    if rel_type and source:
                        if rel_type not in rel_by_type:
                            rel_by_type[rel_type] = set()
                            
                        # Format the relationship based on its type
                        if rel_type == 'HAS_SEMANTIC_TYPE':
                            rel_str = f"{source} has semantic type: {target}"
                        elif rel_type == 'HAS_DEFINITION':
                            rel_str = f"{source} definition: {target}"
                        elif intermediate:
                            rel_str = f"{source} -> {intermediate} -> {target}"
                        else:
                            rel_str = f"{source} -> {target}"
                            
                        rel_by_type[rel_type].add(rel_str)
                
                # Display relationships grouped by type
                for rel_type, rels in sorted(rel_by_type.items()):
                    print(f"\n{rel_type}:")
                    for rel in sorted(rels):
                        print(f"• {rel}")
                
                print("\n" + "="*50)
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        print("\nSession ended")

if __name__ == "__main__":
    main()