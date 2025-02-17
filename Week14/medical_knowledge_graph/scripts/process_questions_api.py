import sys
from pathlib import Path
import os
from typing import Dict, List, Any
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
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

    async def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key medical terms from the question using LLM"""
        prompt = f"""
        Given this medical question, extract the key medical terms (diseases, symptoms, treatments, etc.):
        
        Question: {question}
        
        Return only the key terms as a comma-separated list.
        """
        
        response = self.llm(prompt)
        terms = [term.strip() for term in response.split(',')]
        return terms

    async def _get_concept_relationships(self, concept_info: Dict) -> List[Dict]:
        """Get relationships between identified concepts"""
        relationships = []
        
        # Get CUIs from concept info
        cuis = []
        for results in concept_info.values():
            for result in results:
                if 'basic_info' in result and 'ui' in result['basic_info']:
                    cuis.append(result['basic_info']['ui'])
        
        # Get relationships for each CUI
        for cui in cuis:
            rels = await self.umls_api.get_concept_relationships(cui)
            relationships.extend(rels)
        
        return relationships

    # async def _generate_answer(self, question: str, concepts: Dict, relationships: List) -> str:
    #   """Generate answer using LLM based on UMLS data"""
    #   # Format UMLS data more explicitly
    #   umls_context = "UMLS Knowledge:\n"
      
    #   # Add concept definitions from UMLS
    #   for term, results in concepts.items():
    #       for result in results:
    #           if 'basic_info' in result and 'definitions' in result:
    #               umls_context += f"\nConcept: {result['basic_info'].get('name', '')}\n"
    #               umls_context += f"CUI: {result['basic_info'].get('ui', '')}\n"
    #               if result['definitions']:
    #                   umls_context += f"Definition: {result['definitions'][0].get('value', '')}\n"
    #               if result['semantic_types']:
    #                   umls_context += f"Semantic Type: {result['semantic_types'][0].get('name', '')}\n"

    #   # Add relationships from UMLS
    #   umls_context += "\nRelationships from UMLS:\n"
    #   for rel in relationships:
    #       if 'relationship_type' in rel and 'related_concept' in rel:
    #           rel_type = rel['relationship_type']
    #           related = rel['related_concept']
    #           umls_context += f"- {rel_type}: {related.get('name', '')}\n"

    #   prompt = f"""
    #   Based ONLY on the following UMLS (Unified Medical Language System) data, answer the medical question.
    #   Do not use any other medical knowledge that you may have. Use only the information provided from UMLS:

    #   Question: {question}

    #   {umls_context}

    #   Please provide:
    #   1. A direct answer using only the UMLS data above
    #   2. Supporting evidence citing specific UMLS concepts and relationships
    #   3. Medical context derived only from the UMLS definitions and relationships provided

    #   If the UMLS data is insufficient to answer any part, state that explicitly.
    #   """
      
    #   return self.llm(prompt)

    async def _generate_answer(self, question: str, concepts: Dict, relationships: List) -> str:
      """Generate answer using both UMLS data and LLM knowledge"""
      try:
          # Format UMLS data concisely
          umls_data = "UMLS Knowledge Base:\n"
          
          # Add relevant UMLS concepts
          for term, results in list(concepts.items())[:3]:
              for result in results[:1]:
                  if 'basic_info' in result:
                      cui = result['basic_info'].get('ui', '')
                      name = result['basic_info'].get('name', '')
                      umls_data += f"\nConcept: {name} (CUI: {cui})\n"
                      
                      if result.get('definitions'):
                          definition = result['definitions'][0].get('value', '')[:200]
                          umls_data += f"Definition: {definition}...\n"
                      
                      if result.get('semantic_types'):
                          sem_type = result['semantic_types'][0].get('name', '')
                          umls_data += f"Type: {sem_type}\n"
          
          # Add key relationships
          umls_data += "\nUMLS Relationships:\n"
          seen_relationships = set()
          for rel in relationships[:10]:
              if 'relationship_type' in rel and 'related_concept' in rel:
                  rel_type = rel['relationship_type']
                  related = rel['related_concept']
                  rel_key = f"{rel_type}:{related.get('name', '')}"
                  
                  if rel_key not in seen_relationships:
                      umls_data += f"- {rel_type}: {related.get('name', '')} (CUI: {related.get('cui', '')})\n"
                      seen_relationships.add(rel_key)

          prompt = f"""
          Answer this medical question using ONLY the provided UMLS data. Do not use any of your previousmedical knowledge.
          
          Question: {question}

          {umls_data}

          Provide your answer in this format:
          1. Answer and Explanation:
            - State the correct option
            - Explain using both UMLS data and medical knowledge
            - Clearly distinguish between UMLS-sourced information and additional medical context
          
          2. UMLS Evidence:
            - Cite relevant UMLS concepts (with CUIs) that support the answer
            - List any relevant UMLS relationships
            - State what information is specifically from UMLS
          
          3. Additional Medical Context:
            - Provide additional medical reasoning not found in UMLS
            - Explain how this complements the UMLS data
            - Note any important medical context that helps understand the answer
          
          Remember to:
          - Clearly distinguish between UMLS data and additional medical knowledge
          - Use UMLS data to validate your medical knowledge where possible
          - Be explicit when making connections beyond UMLS data
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
                
                print("\nRelevant Concepts:")
                print("-" * 50)
                for term, concepts in result['concepts'].items():
                    for concept in concepts:
                        if 'basic_info' in concept:
                            print(f"• {concept['basic_info'].get('name', '')}")
                            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        print("\nQuestion processing completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 