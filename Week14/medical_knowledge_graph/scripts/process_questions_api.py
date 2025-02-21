import sys
from pathlib import Path
import os
from typing import Dict, List, Any
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import json
# Get the absolute path to the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.processors.question_processor import QuestionProcessor
from src.processors.umls_api_processor import UMLSAPIProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalQuestionProcessorAPI:
    def __init__(self, llm_function, umls_api_key: str):
        self.llm = llm_function
        self.umls_api = UMLSAPIProcessor(api_key=umls_api_key)
        
        # Define medical knowledge schema
        self.medical_schema = """
        Medical Knowledge Schema:
        1. Concept Types:
        - Diseases and Conditions
        - Symptoms and Findings
        - Treatments and Procedures
        - Medications
        - Anatomical Structures
        - Laboratory Tests
        
        2. Key Relationships:
        - has_symptom
        - treats
        - caused_by
        - located_in
        - part_of
        - associated_with
        """
        
    async def process_medical_question(self, question: str) -> Dict[str, Any]:
        """Process a medical question using UMLS API"""
        try:
            # 1. Extract key medical terms from question using LLM
            key_terms = await self._extract_key_terms(question)
            logger.info(f"Extracted terms: {key_terms}")
            
            # 2. Search UMLS for each term
            concept_info = {}
            for term in key_terms:
                results = await self.umls_api.search_and_get_info(term)
                if results:
                    concept_info[term] = results
            
            # 3. Get relationships between concepts
            relationships = await self._get_concept_relationships(concept_info)
            
            # 4. Generate answer using LLM
            answer = await self._generate_answer(question, concept_info, relationships)
            
            return {
                'question': question,
                'key_terms': key_terms,
                'concepts': concept_info,
                'relationships': relationships,
                'answer': answer
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {'error': str(e)}

    async def _extract_key_terms(self, question: str) -> List[Dict]:
        """Extract key medical terms from the question using LLM"""
        prompt = f"""
        Extract and categorize medical terms from this question based on these node types:
        - Disease: medical conditions and disorders
        - Drug: medications and therapeutic substances
        - Procedure: medical procedures and interventions
        - Symptom: clinical findings, lab values, and manifestations
        - Anatomy: anatomical structures and locations
        
        Question: {question}
        
        Return ONLY a JSON array of objects with this format:
        [
            {{"term": "term_name", "type": "node_type", "priority": 1-3}}
        ]
        
        Rules:
        1. Priority 1 for main conditions/diseases
        2. Priority 2 for symptoms/findings
        3. Priority 3 for procedures/context
        4. Only include relevant medical terms
        5. Do not include any markdown formatting or backticks
        """
        
        try:
            response = self.llm(prompt)
            
            # Clean the response - remove backticks and "json" text
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            
            # Parse the cleaned JSON
            terms = json.loads(cleaned_response)
            logger.info(f"Successfully extracted {len(terms)} terms")
            
            # Extract just the term names for UMLS search
            term_names = [term['term'] for term in terms]
            return term_names
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {response}")
            logger.error(f"JSON Error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in term extraction: {str(e)}")
            return []

    async def _get_concept_relationships(self, concept_info: Dict) -> List[Dict]:
        """Get relationships between identified concepts"""
        relationships = []
        
        # Get CUIs from concept info
        concept_cuis = {}  # Store CUI to concept name mapping
        for term, results in concept_info.items():
            for result in results:
                if 'basic_info' in result:
                    cui = result['basic_info'].get('ui')
                    name = result['basic_info'].get('name')
                    if cui and name:
                        concept_cuis[cui] = name

        # Get relationships for each CUI
        for cui, name in concept_cuis.items():
            rels = await self.umls_api.get_concept_relationships(cui)
            for rel in rels:
                if rel.get('relatedId') in concept_cuis:  # Only include relationships between found concepts
                    relationships.append({
                        'sourceUi': cui,
                        'sourceName': name,
                        'relationLabel': rel.get('relationLabel'),
                        'relatedId': rel.get('relatedId'),
                        'relatedName': concept_cuis[rel.get('relatedId')]
                    })
        
        return relationships

    def _select_best_definition(self, definitions: List[Dict]) -> str:
        """
        Select the best English definition from pre-filtered English definitions.
        """
        if not definitions:
            return "No definition available."

        # Print all available definitions
        print("\nAvailable English Definitions:")
        for i, d in enumerate(definitions, 1):
            print(f"\nDefinition {i}:")
            print(f"Value: {d.get('value', '')}")
            print(f"Source: {d.get('rootSource', 'Unknown')}")

        # Prioritize sources in this order
        preferred_sources = ['NCI', 'MSH', 'SNOMEDCT_US', 'MTH', 'CSP']
        
        # Try to find definition from preferred sources
        for source in preferred_sources:
            for definition in definitions:
                if definition.get('rootSource') == source:
                    selected_def = definition.get('value', '')
                    print(f"\nSelected Definition (from {source}):")
                    print(selected_def)
                    return selected_def

        # If no preferred source found, return the first available definition
        selected_def = definitions[0].get('value', '')
        print("\nSelected Definition (first available):")
        print(selected_def)
        return selected_def

    async def _generate_answer(self, question: str, concepts: Dict, relationships: List) -> str:
        """Generate answer using both UMLS data and LLM knowledge"""
        try:
            # Format UMLS data with more detail
            umls_data = "UMLS Knowledge Base:\n\n"
            
            # Add concepts with best definitions
            umls_data += "Concepts and Definitions:\n"
            for term, results in concepts.items():
                for result in results:
                    if 'basic_info' in result:
                        cui = result['basic_info'].get('ui', '')
                        name = result['basic_info'].get('name', '')
                        umls_data += f"\n• {name} (CUI: {cui})"
                        
                        # Add the best definition
                        if result.get('definitions'):
                            best_definition = self._select_best_definition(result['definitions'])
                            if best_definition:
                                umls_data += f"\n  Definition: {best_definition}"
                        
                        # Add semantic types
                        if result.get('semantic_types'):
                            sem_types = [st.get('name', '') for st in result['semantic_types']]
                            umls_data += f"\n  Semantic Types: {', '.join(sem_types)}"
            
            # Add relationships with more detail
            umls_data += "\n\nRelationships between Concepts:\n"
            seen_relationships = set()
            for rel in relationships:
                if 'relationLabel' in rel and 'relatedId' in rel:
                    source_cui = rel.get('sourceUi', '')
                    target_cui = rel.get('relatedId', '')
                    rel_type = rel.get('relationLabel', '')
                    source_name = rel.get('sourceName', '')
                    target_name = rel.get('relatedName', '')
                    
                    rel_key = f"{source_cui}:{rel_type}:{target_cui}"
                    if rel_key not in seen_relationships:
                        umls_data += f"\n• {source_name} (CUI: {source_cui}) -> {rel_type} -> {target_name} (CUI: {target_cui})"
                        seen_relationships.add(rel_key)

            prompt = f"""
            Answer this medical question using ONLY the provided UMLS data. Do not use any of your previous medical knowledge.
            
            Question: {question}

            {umls_data}

            Provide your answer in this format:
            1. Answer and Explanation:
                - State the correct option
                - Explain using ONLY the UMLS data provided above
                - Cite specific CUIs and relationships
            
            2. UMLS Evidence Used:
                - List all relevant concepts with their CUIs
                - List all relevant relationships
                - Explicitly state what information comes from UMLS
            
            3. Missing Information:
                - State what additional UMLS data would be helpful
            """
            
            return self.llm(prompt)
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Error: Unable to generate answer due to data processing limitations."

    def _format_concepts(self, concepts: Dict) -> str:
        """Format concepts for LLM prompt"""
        formatted = []
        for term, results in concepts.items():
            for result in results:
                if 'basic_info' in result and 'definitions' in result:
                    basic = result['basic_info']
                    defs = result['definitions']
                    formatted.append(f"Term: {basic.get('name', '')}")
                    if defs:
                        formatted.append(f"Definition: {defs[0].get('value', '')}")
        return "\n".join(formatted)

    def _format_relationships(self, relationships: List) -> str:
        """Format relationships for LLM prompt"""
        formatted = []
        for rel in relationships:
            if 'relationship_type' in rel and 'related_concept' in rel:
                rel_type = rel['relationship_type']
                related = rel['related_concept']
                formatted.append(f"{rel_type}: {related.get('name', '')}")
        return "\n".join(formatted)

async def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        umls_api_key = os.getenv("API_KEY")
        # Initialize processor
        processor = MedicalQuestionProcessorAPI(
            llm_function=lambda x: client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": x}]
            ).choices[0].message.content,
            umls_api_key=umls_api_key
        )
        

        # Process questions
        while True:
            print("\nEnter your medical question (or 'quit' to exit): ")
            question = input().strip()
            
            if question.lower() == 'quit':
                break
                
            print("\nProcessing question...")
            result = await processor.process_medical_question(question)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("\nAnswer:")
                print("-" * 50)
                print(result['answer'])
                
                print("\nKey Terms Used:")
                print("-" * 50)
                print(", ".join(result['key_terms']))
                
                print("\nRelevant Concepts and Definitions:")
                print("-" * 50)
                for term, concepts in result['concepts'].items():
                    for concept in concepts:
                        if 'basic_info' in concept:
                            print(f"\n• {concept['basic_info'].get('name', '')}")
                            if concept.get('definitions'):
                                print(f"  Definition: {concept['definitions'][0].get('value', '')}")
                            if concept.get('semantic_types'):
                                print(f"  Type: {concept['semantic_types'][0].get('name', '')}")
                
                print("\nRelationships:")
                print("-" * 50)
                for rel in result['relationships']:
                    if 'relationLabel' in rel and 'relatedId' in rel:
                        print(f"• {rel.get('sourceName', '')} -> {rel.get('relationLabel', '')} -> {rel.get('relatedName', '')}")
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        print("\nQuestion processing completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 