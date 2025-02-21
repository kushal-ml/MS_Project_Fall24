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
from process_question import MedicalQuestionProcessor  # Import for term extraction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalQuestionProcessorAPI:
    def __init__(self, llm_function, umls_api_key: str):
        self.llm = llm_function
        self.umls_api = UMLSAPIProcessor(api_key=umls_api_key)
        # Initialize Neo4j processor for term extraction
        self.neo4j_processor = MedicalQuestionProcessor(graph=None, llm_function=llm_function)
        
    async def process_medical_question(self, question: str) -> Dict[str, Any]:
        """Process a medical question using UMLS API"""
        try:
            # 1. Extract key medical terms using Neo4j processor's function
            key_terms = self.neo4j_processor._extract_key_terms(question)
            logger.info(f"Extracted terms: {key_terms}")
            
            # 2. Search UMLS for each term and get concept information
            concept_info = {}
            for term_info in key_terms:
                term = term_info['term']
                results = await self.umls_api.search_and_get_info(term)
                if results:
                    concept_info[term] = results
            logger.info(f"Retrieved concepts: {len(concept_info)}")
            
            # 3. Get relationships for all concepts
            relationships = []
            for term, concepts in concept_info.items():
                for concept in concepts:
                    if 'basic_info' in concept:
                        cui = concept['basic_info'].get('ui')
                        if cui:
                            rels = await self.umls_api.get_concept_relationships(cui)
                            relationships.extend(rels)
            logger.info(f"Retrieved relationships: {len(relationships)}")
            
            # 4. Generate answer using LLM
            answer = await self._analyze_and_generate_answer(question, concept_info, relationships)
            
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

    async def _analyze_and_generate_answer(self, question: str, concepts: Dict, relationships: List) -> str:
        """Analyze relationships and generate answer"""
        try:
            # Format context for LLM
            context = self._format_context(concepts, relationships)
            
            prompt = f"""You are a medical expert analyzing UMLS knowledge to answer a question.
            
            Question: {question}

            Available UMLS Knowledge:
            {context}

            Analyze this data and provide:

            1. RELATIONSHIP ANALYSIS:
            - Identify which relationship paths are relevant to the question. Give the relationship path in this format: Concept name (CUI) -> Relationship -> Concept name (CUI).
            - Explain why each path is important
            - Note any missing but needed relationships

            2. CONCEPT ANALYSIS:
            - List key concepts needed for the answer
            - Explain how each concept contributes
            - Note any missing but needed concepts

            3. ANSWER DEDUCTION:
            - State your conclusion
            - Show exact evidence path that led to this conclusion
            - If using elimination method, explain the logic
            - Explicitly state what information comes from UMLS vs what's missing

            4. CONFIDENCE ASSESSMENT:
            - Rate confidence in answer (High/Medium/Low)
            - Explain what would increase confidence
            - List any assumptions made

            Format as:
            RELEVANT RELATIONSHIPS:
            - List each important relationship path
            - Explain its significance

            KEY CONCEPTS USED:
            - List each concept with CUI
            - Explain its role

            DEDUCTION PROCESS:
            - Show step-by-step reasoning based on the retrieved relationships and concepts
            - Highlight evidence used
            - Note any elimination logic

            FINAL ANSWER:
            - State conclusion
            - Explain confidence level
            - Note limitations

            Remember:
            1. Only use information from the provided UMLS data
            2. Be explicit about missing information
            3. Show clear reasoning paths
            4. Cite specific CUIs and relationships
            """
            
            return self.llm(prompt)
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Error: Unable to generate answer."

    def _format_context(self, concepts: Dict, relationships: List) -> str:
        """Format concepts and relationships for LLM context"""
        context = "UMLS Concepts:\n"
        
        # Format concepts
        for term, results in concepts.items():
            context += f"\nTerm: {term}\n"
            for result in results:
                if 'basic_info' in result:
                    info = result['basic_info']
                    context += f"- CUI: {info.get('ui')}\n"
                    context += f"- Name: {info.get('name')}\n"
                    if 'definitions' in result and result['definitions']:
                        context += f"- Definition: {result['definitions'][0].get('value')}\n"
                    if 'semantic_types' in result and result['semantic_types']:
                        context += f"- Type: {result['semantic_types'][0].get('name')}\n"
                    context += "\n"
        
        # Format relationships
        context += "\nUMLS Relationships:\n"
        for rel in relationships:
            if all(k in rel for k in ['relationLabel', 'relatedId', 'relatedName']):
                context += f"- {rel.get('sourceName')} ({rel.get('sourceUi')}) "
                context += f"-> {rel.get('relationLabel')} -> "
                context += f"{rel.get('relatedName')} ({rel.get('relatedId')})\n"
        
        return context

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
                print("\n" + "="*80)
                print("MEDICAL QUESTION ANALYSIS")
                print("="*80)
                
                print("\nQUESTION:")
                print("-"*50)
                print(result['question'])
                
                print("\nEXTRACTED TERMS:")
                print("-"*50)
                for term in result['key_terms']:
                    print(f"• {term['term']} ({term['type']})")
                
                print("\nRETRIEVED CONCEPTS:")
                print("-"*50)
                for term, concepts in result['concepts'].items():
                    for concept in concepts:
                        if 'basic_info' in concept:
                            info = concept['basic_info']
                            print(f"\n• {info.get('name')} (CUI: {info.get('ui')})")
                            if concept.get('definitions'):
                                print(f"  Definition: {concept['definitions'][0].get('value')}")
                            if concept.get('semantic_types'):
                                print(f"  Type: {concept['semantic_types'][0].get('name')}")
                
                print("\nRELEVANT RELATIONSHIPS:")
                print("-"*50)
                for rel in result['relationships']:
                    if all(k in rel for k in ['relationLabel', 'relatedId', 'relatedName']):
                        print(f"\n• {rel.get('sourceName')} -> {rel.get('relationLabel')} -> {rel.get('relatedName')}")
                
                print("\nANALYSIS AND ANSWER:")
                print("-"*50)
                print(result['answer'])
                
                print("\n" + "="*80)
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        print("\nQuestion processing completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())