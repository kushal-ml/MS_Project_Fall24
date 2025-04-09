import sys
from pathlib import Path
import os
from typing import Dict, List, Any
import json
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from openai import OpenAI
import logging

# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class USMLEQuestionProcessor:
    def __init__(self, graph: Neo4jGraph, llm_function):
        self.graph = graph
        self.llm = llm_function
        logger.info("USMLE Question Processor initialized")

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
            logger.info("Generated answer from LLM")
            
            # Ensure 'concepts' and 'relationships' are never None
            return {
                'answer': str(answer),
                'key_terms': [term['term'] for term in key_terms] if key_terms else [],
                'concepts': concepts if concepts else [],  # Ensure concepts is a list
                'relationships': relationships if relationships else []  # Ensure relationships is a list
            }
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            return {'error': str(e)}

    def _extract_key_terms(self, question: str) -> List[Dict]:
        """Extract key medical terms and categorize them by node types"""
        prompt = f"""
        Extract medical terms from this question and categorize them:
        - Disease: Medical conditions/disorders
        - Drug: Medications/substances
        - Symptom: Clinical findings
        - Anatomy: Body structures
        - Procedure: Medical procedures
        - ClinicalScenario: Patient context
        
        Question: {question}
        
        Return JSON list with:
        [
            {{
                "term": "term",
                "type": "node_type",
                "priority": 1-3
            }}
        ]
        Return ONLY the JSON array.
        """
        
        response = self.llm(prompt)
        try:
            cleaned_response = response.strip().replace('```json', '').replace('```', '')
            return json.loads(cleaned_response)
        except Exception as e:
            logger.error(f"Error parsing terms: {str(e)}")
            return []
    
    def _get_concepts(self, terms: List[Dict]) -> List[Dict]:
        """Get concepts using keyword search"""
        try:
            concepts = []
            for term_info in terms:
                term = term_info['term']
                query = """
                MATCH (c:Concept)
                WHERE toLower(c.term) CONTAINS toLower($term)
                OR EXISTS {
                    MATCH (c)-[:SAME_AS]-(syn:Concept)
                    WHERE toLower(syn.term) CONTAINS toLower($term)
                }
                OR EXISTS {
                    MATCH (c)-[:TRADENAME_OF]-(tradename:Concept)
                    WHERE toLower(tradename.term) CONTAINS toLower($term)
                }
                WITH c
                OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d:Definition)
                RETURN DISTINCT
                    c.cui AS cui,
                    c.term AS term,
                    d.text AS definition,
                    labels(c) AS types
                
                """
                results = self.graph.query(query, {'term': term})
                # Filter out invalid results
                valid_results = []
                for r in results:
                    if not r:
                        logger.warning(f"Skipping None result for term: {term}")
                        continue
                    if not isinstance(r, dict):
                        logger.warning(f"Skipping invalid result (not a dictionary) for term: {term}")
                        continue
                    if 'cui' not in r or 'term' not in r or 'types' not in r:
                        logger.warning(f"Skipping invalid result (missing required keys) for term: {term}")
                        continue
                    valid_results.append(r)
                concepts.extend(valid_results)
            return concepts
        except Exception as e:
            logger.error(f"Concept search error: {str(e)}")
            return []
    def _get_relationships(self, concepts: List[Dict]) -> List[Dict]:
        """Get relationships between found concepts"""
        try:
            if not concepts:
                return []
                
            cuis = [c.get('cui') for c in concepts if c and c.get('cui')]
            if not cuis:
                return []
                
            query = """
            MATCH (c1:Concept)-[r]->(c2:Concept)
            WHERE c1.cui IN $cuis OR c2.cui IN $cuis
                RETURN DISTINCT
                type(r) AS relationship_type,
                c1.cui AS source_cui,
                c1.term AS source_name,
                c2.cui AS target_cui,
                c2.term AS target_name,
                labels(c1) AS source_labels,
                labels(c2) AS target_labels
            """
            results = self.graph.query(query, {'cuis': cuis})
            # Filter out invalid results
            valid_results = [r for r in results if r and 'source_name' in r and 'relationship_type' in r and 'target_name' in r]
            return valid_results
                except Exception as e:
            logger.error(f"Relationship search error: {str(e)}")
            return []
    def _generate_answer(self, question: str, concepts: List[Dict], relationships: List[Dict]) -> str:
        """Generate answer using retrieved concepts and relationships"""
        context = f"Question: {question}\n\nRelevant Concepts:\n"
        
        # Build concept context
        seen = set()
        for concept in concepts:
            # Skip if concept is None or missing required keys
            if not concept:
                logger.warning("Skipping None concept.")
                continue
            
            if not isinstance(concept, dict):
                logger.warning(f"Skipping invalid concept (not a dictionary): {concept}")
                continue
            
            if 'cui' not in concept or 'term' not in concept or 'types' not in concept:
                logger.warning(f"Skipping invalid concept (missing required keys): {concept}")
                continue
            
            if concept['cui'] not in seen:
                context += f"- {concept['term']} ({', '.join(concept['types'])}"
                if concept.get('definition'):
                    context += f": {concept['definition'][:100]}..."
                context += ")\n"
                seen.add(concept['cui'])
        
        # Build relationship context
        context += "\nKey Relationships:\n"
        for rel in relationships:
            # Skip if relationship is None or missing required keys
            if not rel or 'source_name' not in rel or 'relationship_type' not in rel or 'target_name' not in rel:
                logger.warning(f"Skipping invalid relationship: {rel}")
                continue
            
            context += (f"- {rel['source_name']} --[{rel['relationship_type']}]--> "
                        f"{rel['target_name']}\n")
        
        # Generate answer
        prompt = f"""
        As a medical expert, answer this question using ONLY the provided context.
        Be concise and specific. If unsure, state the limitations.
        
        Context:
        {context}
        
        Question: {question}
        
        Please provide a detailed answer using ONLY the provided evidence following this format:
        1. REASONING PROCESS:
        - Initial Understanding: [Break down the question and what it's asking]
        - Key Findings: [List the relevant facts from both knowledge sources]
        - Chain of Thought: [Explain step-by-step how these facts lead to the answer. Do not look for the answer in the evidence, but rather use the evidence to reason to the answer.]
        
        2. EVIDENCE USED:
        - Knowledge Graph: [Cite specific concepts and relationships used]
        
        3. DIFFERENTIAL REASONING:
        - Why the correct answer is right
        - Why other choices are wrong (if information available)
        
        4. CONFIDENCE AND LIMITATIONS:
        - State confidence level in answer
        - Note any missing information that would have helped

        5. MY KNOWLEDGE AND ASSUMPTIONS:
        - State any assumptions you made in your reasoning process
        - State any knowledge you had used in your reasoning process, that was not in the evidence provided
        - Also, state if you could have answered the question with high confidence, without the evidence provided and explain why. If not, explain how the evidence helped you better arrive at the answer.

        Remember:
        - Only use provided evidence
        - Clearly cite sources for each claim
        - Be explicit about any assumptions
        - If evidence is insufficient, say so
        """
        
        return self.llm(prompt)

    def _display_results(self, result: Dict):
        if not isinstance(result, dict):
            logger.error("Result is not a dictionary")
            return

        # Display concepts section
        concepts = result.get('concepts', [])
        if concepts:
            print("\n" + "="*50)
            print("SUPPORTING CONCEPTS FROM KNOWLEDGE GRAPH")
            print("="*50)
            
            # Track seen concepts to avoid duplicates
            seen_concepts = set()
            displayed_count = 0
            
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue
                    
                # Break if we've displayed 10 concepts
                if displayed_count >= 5:
                    remaining = len(concepts) - displayed_count
                    if remaining > 0:
                        print(f"\n... and {remaining} more concepts")
                    break
                    
                # Skip if we've seen this concept already
                concept_id = (concept.get('term', ''), concept.get('definition', ''))
                if concept_id in seen_concepts:
                    continue
                seen_concepts.add(concept_id)
                
                # Format the definition
                definition = concept.get('definition', 'No definition available')
                if definition and len(definition) > 100:
                    definition = definition[:97] + "..."
                
                # Remove HTML tags if present
                if definition:
                    definition = definition.replace('<p>', '').replace('</p>', '')
                    definition = definition.replace('<h3>', '').replace('</h3>', '')
                    definition = definition.replace('<a href="', '').replace('">', ' ').replace('</a>', '')
                
                # Print formatted concept
                print(f"\n�� {concept.get('term', 'Unknown term')}")
                print(f"   └─ {definition}")
                displayed_count += 1
            
            # Display relationships section
            relationships = result.get('relationships', [])
            if relationships:
                print("\n" + "="*50)
                print("RELATIONSHIPS FOUND")
                print("="*50)
                
                try:
                    # Group relationships by type
        rel_by_type = {}
        for rel in relationships:
                        if not isinstance(rel, dict):
                            continue
            rel_type = rel.get('relationship_type', 'Unknown')
            if rel_type not in rel_by_type:
                rel_by_type[rel_type] = []
            rel_by_type[rel_type].append(rel)
        
                    if rel_by_type:
                        # Sort relationship types by number of relationships and take top 5
                        sorted_rel_types = sorted(
                            rel_by_type.items(), 
                            key=lambda x: len(x[1]), 
                            reverse=True
                        )[:5]
                        
                        # Display top 5 relationship types
                        for rel_type, rels in sorted_rel_types:
                            print(f"\n🔄 {rel_type} ({len(rels)} total):")
                            for rel in rels[:5]:  # Show up to 5 relationships per type
                                source = rel.get('source_name', 'Unknown')
                                target = rel.get('target_name', 'Unknown')
                                print(f"   └─ {source} → {target}")
            if len(rels) > 5:
                                print(f"   └─ ... and {len(rels) - 5} more")
                        
                        # Show how many relationship types were not displayed
                        remaining_types = len(rel_by_type) - 5
                        if remaining_types > 0:
                            print(f"\n... and {remaining_types} more relationship types")
                    else:
                        print("\nNo valid relationships found")
                except Exception as e:
                    logger.error(f"Error processing relationships: {str(e)}")
                    print("\nError processing relationships")
        else:
                print("\nNo relationships found")
            
            print("\n" + "="*50)
        else:
            print("\nNo concepts found in the knowledge graph")

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
        
        # Initialize OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        processor = USMLEQuestionProcessor(
            graph=graph,
            llm_function=lambda x: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": x}],
                temperature=0.0
            ).choices[0].message.content
        )
        
        while True:
            print("\nEnter medical question (or 'quit'):")
            question = input().strip()
            if question.lower() == 'quit':
                break
                
            result = processor.process_medical_question(question)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("\n=== ANSWER ===")
                print(result['answer'])
            
            processor._display_results(result)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        print("\nSession ended")

if __name__ == "__main__":
    main()















