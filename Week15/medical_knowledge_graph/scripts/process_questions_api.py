# process_questions_api.py
import sys
from pathlib import Path
import os
from typing import Dict, List, Any, Optional
import logging
from dotenv import load_dotenv
from openai import OpenAI
import json
import asyncio

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.processors.umls_api_processor import UMLSAPIProcessor  # Import the updated class

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalQuestionProcessorAPI:
    def __init__(self, llm_function, umls_api_key: str):
        self.llm = llm_function
        self.umls_api = UMLSAPIProcessor(api_key=umls_api_key)
        self.concept_cache = {}

    async def process_medical_question(self, question: str) -> Dict[str, Any]:
        """Process a medical question using UMLS and LLM."""
        try:
            key_terms = await self._extract_key_terms(question)
            logger.info(f"Extracted terms: {key_terms}")
            
            concept_info = {}
            for term in key_terms:
                results = await self.umls_api.search_and_get_info(term)
                if results:
                    concept_info[term] = results
            
            relationships = await self.umls_api._get_concept_relationships(concept_info)
            answer = await self._generate_answer(question, concept_info, relationships)
            
            return answer
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {'error': str(e)}

    async def _extract_key_terms(self, question: str) -> List[str]:
        """Extract medical terms from the question using LLM."""
        prompt = f"""Extract ONLY medical terms from: {question}
        Return ONLY a JSON array of objects with 'term' and 'type' fields.
        Example: [{{"term": "aspirin", "type": "medication"}}, ...]"""  # <-- Escaped braces
        
        try:
            response = self.llm(prompt)
            cleaned = response.strip().strip('```json').strip()
            terms_data = json.loads(cleaned)
            return list({term['term'] for term in terms_data if 'term' in term})
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return []
        except Exception as e:
            logger.error(f"Term extraction failed: {e}")
            return []
        
    async def _get_concept_name(self, cui: str) -> str:
        """Get the name of a concept by its CUI."""
        if not cui:
            return "Unknown"
            
        if cui in self.concept_cache:
            return self.concept_cache[cui]
            
        try:
            results = await self.umls_api.search_and_get_info(cui)
            if results and results[0].get('concept_details'):
                name = results[0]['concept_details'].get('name', 'Unknown')
                self.concept_cache[cui] = name
                return name
        except Exception as e:
            logger.error(f"Name lookup failed for {cui}: {e}")
            
        return "Unknown"
    
    async def _generate_answer(self, question: str, concepts: Dict, relationships: List) -> Dict[str, Any]:
        """Generate an answer using UMLS data and LLM."""
        try:
            umls_data = "UMLS Knowledge Base:\n\nConcepts:\n"
            for term, results in concepts.items():
                for result in results:
                    concept_details = result.get('concept_details', {})
                    # Fix: Correct the f-string formatting
                    umls_data += f"- {concept_details.get('name', term)} (CUI: {concept_details.get('ui', '?')})\n"
                    if result.get('definitions'):
                        def_text = self._select_best_definition(result['definitions'])
                        umls_data += f"  Definition: {def_text[:200]}...\n"
            
            umls_data += "\nRelationships:\n"
            for rel in relationships[:50]:  # Limit to first 50 for brevity
                umls_data += (
                    f"- {rel['sourceName']} ({rel['sourceUi']}) → "
                    f"{rel['relationLabel'].replace('_', ' ').title()} → "
                    f"{rel['relatedName']} ({rel['relatedId']})\n"
                )

            prompt = f"""Answer this medical question using UMLS data:
            Question: {question}
            {umls_data}
            Structure your answer with:
            1. Clinical reasoning
            2. Supporting UMLS concepts
            3. Key relationships"""
            
            llm_response = self.llm(prompt)
            
            return {
                'answer': llm_response,
                'concepts': concepts,
                'relationships': relationships,
                'key_terms': list(concepts.keys())
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {'error': str(e)}

    def _select_best_definition(self, definitions: List[Dict]) -> str:
        """Select the best definition from a list of definitions."""
        preferred_sources = ['NCI', 'MSH', 'SNOMEDCT_US']
        for source in preferred_sources:
            for defn in definitions:
                if defn.get('rootSource') == source:
                    return defn.get('value', '')
        return definitions[0].get('value', '') if definitions else ''

    async def get_relationships_for_concepts(self, concept_results):
        """Get relationships for all concepts and return as formatted data"""
        all_relationships = []
        
        for term, results in concept_results.items():
            for concept in results:
                concept_details = concept.get('concept_details', {})
                cui = concept_details.get('ui')
                
                if cui:
                    concept_name = concept_details.get('name', 'Unknown')
                    logger.info(f"Getting relationships for {concept_name} ({cui})")
                    
                    # Get relationships for this specific concept
                    relationships = await self.umls_api._get_concept_relationships(cui)
                    
                    if relationships:
                        # Add source concept information
                        for rel in relationships:
                            rel['source_concept'] = concept_name
                        
                        all_relationships.extend(relationships)
        
        return all_relationships

async def main():
    processor = None
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get UMLS API key
        umls_key = os.getenv("API_KEY")
        
        # Initialize the processor
        processor = MedicalQuestionProcessorAPI(
            llm_function=lambda x: client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": x}]
            ).choices[0].message.content,
            umls_api_key=umls_key
        )
        
        # Main processing loop
        while True:
            question = input("\nEnter medical question (or 'quit'): ").strip()
            if question.lower() == 'quit':
                break
                
            # Process the question
            result = await processor.process_medical_question(question)
            
            # Display results in a clear, organized format
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("\n" + "="*80)
                print("MEDICAL KNOWLEDGE ANALYSIS".center(80))
                print("="*80)
                
                # Print question
                print("\n📝 QUESTION:")
                print(f"{question}")
                
                # Print key concepts
                print("\n📚 KEY MEDICAL CONCEPTS:")
                for term, concepts in result.get('concepts', {}).items():
                    for i, concept in enumerate(concepts, 1):
                        details = concept.get('concept_details', {})
                        name = details.get('name', term)
                        cui = details.get('ui', 'Unknown')
                        
                        # Get semantic types
                        semantics = []
                        for st in details.get('semanticTypes', []):
                            semantics.append(st.get('name', ''))
                        
                        semantic_text = ", ".join(semantics) if semantics else "Unknown"
                        
                        print(f"{i}. {name} (CUI: {cui}) - Type: {semantic_text}")
                        
                        # Print definition
                        definitions = concept.get('definitions', [])
                        if definitions:
                            definition = definitions[0].get('value', 'No definition available')
                            # Truncate long definitions
                            if len(definition) > 150:
                                definition = definition[:150] + "..."
                            print(f"   Definition: {definition}")
                        print()
                
                # Print relationships
                print("\n🔗 KEY RELATIONSHIPS:")
                if result.get('relationships'):
                    shown_rels = set()  # Track shown relationships to avoid duplicates
                    for i, rel in enumerate(result.get('relationships', []), 1):
                        # Create a unique key to avoid duplicates
                        rel_key = f"{rel['sourceUi']}-{rel['relatedId']}-{rel['relationLabel']}"
                        
                        if rel_key not in shown_rels:
                            print(f"{i}. {rel['sourceName']} --[{rel['relationLabel']}]--> {rel['relatedName']}")
                            shown_rels.add(rel_key)
                        
                        if i >= 15:  # Limit to 15 relationships
                            break
                else:
                    print("   No relationships found in knowledge base")
                
                # Print answer
                print("\n✅ ANSWER:")
                answer_text = result.get('answer', 'No answer generated')
                
                # Add a disclaimer about LLM usage when no relationships found
                if not result.get('relationships'):
                    print("NOTE: Limited knowledge found in UMLS. Answer likely relies on LLM knowledge.")
                
                print(answer_text)
                
                print("\n" + "="*80)
    
    except Exception as e:
        logger.error(f"An error occurred in the main loop: {e}")
    
    finally:
        # Clean up resources
        if processor:
            await processor.umls_api.close()  # Close the UMLS API client
            logger.info("UMLS API client closed successfully")

if __name__ == "__main__":
    asyncio.run(main())