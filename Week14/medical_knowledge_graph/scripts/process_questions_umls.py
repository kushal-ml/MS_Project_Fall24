import sys
from pathlib import Path
import os
from typing import Dict, List, Any
import json

# Add project root to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from openai import OpenAI
import logging
from src.config.constants import (
    USMLE_DOMAINS, CONCEPT_TIERS, RELATIONSHIP_TIERS,
    SEMANTIC_TYPE_TO_LABEL, STEP1_PRIORITY, STEP2_PRIORITY, STEP3_PRIORITY
)
from src.processors.umls_processor_embeddings import UMLSProcessorEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class USMLEQuestionProcessor:
    def __init__(self, graph: Neo4jGraph, llm_function, target_step="STEP2"):
        self.graph = graph
        self.llm = llm_function
        self.target_step = target_step
        self.processor = UMLSProcessorEmbeddings(graph, target_step=target_step)
        
        logger.info(f"Initialized USMLE Question Processor for {target_step}")
        
        # Step-specific configuration
        self.concept_priorities = self._get_step_priorities()
        self.relationship_weights = self._get_relationship_weights()

    def _get_step_priorities(self) -> Dict:
        """Get concept priorities based on target USMLE step"""
        if self.target_step == "STEP1":
            return STEP1_PRIORITY
        elif self.target_step == "STEP2":
            return STEP2_PRIORITY
        elif self.target_step == "STEP3":
            return STEP3_PRIORITY
        return {}

    def _get_relationship_weights(self) -> Dict:
        """Get relationship weights based on tiers"""
        return {
            rel: 1.0 - (0.2 * i) 
            for i, tier in enumerate(RELATIONSHIP_TIERS.values())
            for rel in tier
        }

    def process_question(self, question: str) -> Dict[str, Any]:
        """Process a USMLE-style question using the knowledge graph"""
        try:
            # Step 1: Extract key clinical entities with embeddings
            key_terms = self._extract_key_terms(question)
            logger.info(f"Extracted terms: {key_terms}")
            
            # Step 2: Get enhanced concepts with vector search
            concepts = self._get_enhanced_concepts(key_terms)
            logger.info(f"Found {len(concepts)} concepts")
            
            # Step 3: Get clinical relationships with tier filtering
            relationships = self._get_clinical_relationships(concepts)
            logger.info(f"Found {len(relationships)} relationships")
            
            # Step 4: Generate evidence-based answer
            answer = self._generate_clinical_answer(question, concepts, relationships)
            
            return {
                'answer': answer,
                'concepts': concepts,
                'relationships': relationships,
                'step': self.target_step
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {'error': str(e)}

    def _extract_key_terms(self, question: str) -> List[Dict]:
        """Extract medical terms with clinical context awareness"""
        prompt = f"""
        Analyze this USMLE {self.target_step} question and extract key medical terms:
        {question}
        
        Return JSON list with:
        [{
            "term": "extracted_term",
            "type": "Disease|Drug|Symptom|Procedure|Anatomy",
            "context": "patient_population|pathophysiology|diagnostic_clue|treatment_consideration"
        }]
        """
        
        response = self.llm(prompt)
        try:
            terms = json.loads(response.strip().replace('```json\n', '').replace('\n```', ''))
            return terms
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response")
            return []

    def _get_enhanced_concepts(self, terms: List[Dict]) -> List[Dict]:
        """Get concepts using vector similarity search with clinical context"""
        concepts = []
        
        for term_info in terms:
            # Vector similarity search with clinical context
            vector_results = self.processor.vector_similarity_search(
                term_info['term'], 
                limit=3,
                score_threshold=0.65
            )
            
            if vector_results:
                best_match = max(vector_results, key=lambda x: x['score'])
                concept = self._enrich_concept_data(best_match['cui'])
                concept['match_score'] = best_match['score']
                concept['query_term'] = term_info['term']
                concepts.append(concept)
        
        return concepts

    def _enrich_concept_data(self, cui: str) -> Dict:
        """Enrich concept data with relationships and definitions"""
        cypher = """
        MATCH (c:Concept {cui: $cui})
        OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d:Definition)
        OPTIONAL MATCH (c)-[r]->(related:Concept)
        WHERE r.tier IN ['tier_1', 'tier_2']
        RETURN 
            c.cui as cui,
            c.term as term,
            c.semantic_type as semantic_type,
            c.tier as tier,
            collect(DISTINCT d.text) as definitions,
            collect(DISTINCT {
                type: type(r),
                target_cui: related.cui,
                target_term: related.term,
                target_type: related.semantic_type
            }) as relationships
        """
        result = self.graph.query(cypher, {'cui': cui})
        return dict(result[0]) if result else {}

    def _get_clinical_relationships(self, concepts: List[Dict]) -> List[Dict]:
        """Get clinically relevant relationships between concepts"""
        if not concepts:
            return []
            
        cuis = [c['cui'] for c in concepts]
        
        cypher = """
        MATCH (c1:Concept)-[r]->(c2:Concept)
        WHERE c1.cui IN $cuis AND c2.cui IN $cuis
        AND r.tier IN ['tier_1', 'tier_2']
        RETURN 
            type(r) as relationship_type,
            c1.cui as source_cui,
            c1.term as source_term,
            c2.cui as target_cui,
            c2.term as target_term,
            r.tier as tier,
            r.source as source
        ORDER BY r.tier
        """
        results = self.graph.query(cypher, {'cuis': cuis})
        return [dict(record) for record in results]

    def _generate_clinical_answer(self, question: str, concepts: List[Dict], relationships: List[Dict]) -> str:
        """Generate evidence-based clinical answer using KG data"""
        context = self._build_clinical_context(concepts, relationships)
        
        prompt = f"""
        You are a medical expert answering a USMLE {self.target_step} question. 
        Use ONLY the provided clinical context. If unsure, state what's missing.
        
        Question: {question}
        
        Clinical Context:
        {json.dumps(context, indent=2)}
        
        Answer Structure:
        1. Final diagnosis/diagnostic approach
        2. Key pathophysiological mechanisms
        3. First-line management
        4. Critical differential diagnoses
        5. High-yield associations
        
        Format: Clear, concise bullet points with KG references (CUIs)
        """
        
        return self.llm(prompt)

    def _build_clinical_context(self, concepts: List[Dict], relationships: List[Dict]) -> Dict:
        """Build structured clinical context from KG data"""
        return {
            'concepts': [{
                'cui': c['cui'],
                'term': c['term'],
                'type': c['semantic_type'],
                'tier': c['tier'],
                'definitions': c.get('definitions', []),
                'key_relationships': [r for r in c.get('relationships', []) 
                                    if r['type'] in RELATIONSHIP_TIERS['tier_1']]
            } for c in concepts],
            'relationships': [{
                'type': r['relationship_type'],
                'source': r['source_term'],
                'target': r['target_term'],
                'tier': r['tier']
            } for r in relationships if r['tier'] == 'tier_1']
        }

def main():
    try:
        load_dotenv()
        
        # Initialize Neo4j connection
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j"
        )
        
        # Verify connection
        graph.query("CALL dbms.components()")
        logger.info("Connected to Neo4j database")
        
        # Check data exists
        node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        if node_count == 0:
            logger.error("Database empty! Run UMLS processor first.")
            return
            
        # Initialize OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize processor for Step 2 by default
        processor = USMLEQuestionProcessor(
            graph=graph,
            llm_function=lambda x: client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": x}],
                temperature=0.3
            ).choices[0].message.content,
            target_step="STEP2"
        )
        
        # Interactive question processing
        while True:
            print("\nEnter USMLE-style question (or 'exit'): ")
            question = input().strip()
            
            if question.lower() in ['exit', 'quit']:
                break
                
            print("\nProcessing...")
            result = processor.process_question(question)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue
                
            # Display results
            print("\n=== CLINICAL ANSWER ===")
            print(result['answer'])
            
            print("\nKey Concepts:")
            for concept in result['concepts']:
                print(f"- {concept['term']} ({concept['semantic_type']}) [CUI: {concept['cui']}]")
                if concept.get('definitions'):
                    print(f"  Definition: {concept['definitions'][0][:100]}...")
            
            print("\nCritical Relationships:")
            for rel in result['relationships']:
                if rel['tier'] == 'tier_1':
                    print(f"- {rel['source_term']} → {rel['relationship_type']} → {rel['target_term']}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        print("\nSession ended.")

if __name__ == "__main__":
    main()