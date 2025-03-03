import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from dotenv import load_dotenv
import os
from collections import defaultdict

from process_questions_api import MedicalQuestionProcessorAPI
from query_medical_db import query_medical_knowledge, openai_embeddings, llm
from langchain.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    source: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = None

class UnifiedMedicalQA:
    def __init__(self):
        load_dotenv()
        
        # Initialize UMLS processor
        self.umls_processor = MedicalQuestionProcessorAPI(
            llm_function=lambda x: llm.invoke(x).content,
            umls_api_key=os.getenv("API_KEY")
        )
        
        # Define the unified reasoning prompt
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical expert analyzing a USMLE question. 
            Use the provided evidence from both UMLS (official medical knowledge base) 
            and medical textbooks to explain your reasoning.
            
            IMPORTANT:
            1. Structure your response as follows:
               - Question Analysis
               - UMLS Evidence Analysis
                 * Relevant concepts and their relationships
                 * Drug mechanisms and interactions
               - Textbook Evidence Analysis
               - Integrated Reasoning
               - Final Answer with Confidence Level
               - Sources Used (cite specific CUIs and textbook references)
            
            2. For each piece of evidence:
               - Cite specific UMLS concepts with their CUIs
               - Reference specific relationships between concepts
               - Quote relevant textbook passages
               - Explain how each piece supports your reasoning
               
            3. When analyzing medications:
               - Consider their mechanisms of action
               - Review their indications and contraindications
               - Evaluate their appropriateness for multiple conditions
               
            4. Explicitly state any gaps in the available evidence
            """),
            ("user", """
            Question: {question}
            
            UMLS Evidence:
            {umls_data}
            
            Textbook Evidence:
            {textbook_data}
            
            Provide your detailed analysis and reasoning:
            """)
        ])

    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process a medical question using both UMLS and textbook knowledge"""
        try:
            # 1. Parallel search in both knowledge sources
            logger.info("Starting parallel knowledge retrieval...")
            
            umls_task = asyncio.create_task(self._get_umls_knowledge(question))
            textbook_task = asyncio.create_task(self._get_textbook_knowledge(question))
            
            # Wait for both searches to complete
            umls_results, textbook_results = await asyncio.gather(umls_task, textbook_task)
            
            # Debug logging
            logger.info(f"UMLS Results received: {bool(umls_results)}")
            logger.info(f"UMLS Results type: {type(umls_results)}")
            logger.info(f"UMLS Results keys: {umls_results.keys() if isinstance(umls_results, dict) else 'Not a dict'}")
            
            # 2. Integrate and analyze evidence
            integrated_response = await self._integrate_evidence(
                question=question,
                umls_data=umls_results,
                textbook_data=textbook_results
            )
            
            return {
                'question': question,
                'umls_evidence': umls_results,
                'textbook_evidence': textbook_results,
                'integrated_analysis': integrated_response
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise

    async def _get_umls_knowledge(self, question: str) -> Dict[str, Any]:
        """Get knowledge from UMLS"""
        return await self.umls_processor.process_medical_question(question)

    async def _get_textbook_knowledge(self, question: str) -> List[SearchResult]:
        """Get knowledge from textbook database"""
        # Convert the synchronous function to async
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, query_medical_knowledge, question)
        return results

    async def _integrate_evidence(
        self, 
        question: str, 
        umls_data: Dict[str, Any], 
        textbook_data: List[Dict[str, Any]]
    ) -> str:
        """Integrate and analyze evidence from both sources"""
        try:
            # Format UMLS data
            logger.info("Formatting UMLS evidence...")
            umls_evidence = self._format_umls_evidence(umls_data)
            
            # Format textbook data
            logger.info("Formatting textbook evidence...")
            textbook_evidence = self._format_textbook_evidence(textbook_data or [])
            
            # Prepare prompt input
            prompt_input = {
                'question': question,
                'umls_data': umls_evidence,
                'textbook_data': textbook_evidence
            }
            
            logger.info("Sending to LLM with prompt...")
            
            # Generate integrated analysis
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm.invoke(
                    self.reasoning_prompt.format(**prompt_input)
                ).content
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in evidence integration: {e}")
            logger.error(f"UMLS data that caused error: {umls_data}")
            return "Error integrating evidence"

    def _format_umls_evidence(self, umls_data: Dict[str, Any]) -> str:
        """Format UMLS evidence for the prompt"""
        try:
            formatted = ["=== UMLS Evidence ==="]
            
            # Check for error
            if 'error' in umls_data:
                logger.warning(f"UMLS processing error: {umls_data['error']}")
                return "=== UMLS Evidence ===\nNo UMLS data available"
            
            # Format concepts
            if 'concepts' in umls_data and umls_data['concepts']:
                formatted.append("\nKey Medical Concepts:")
                for term, concepts in umls_data['concepts'].items():
                    formatted.append(f"\n[Term: {term}]")
                    for concept in concepts:
                        if isinstance(concept, dict) and 'basic_info' in concept:
                            info = concept['basic_info']
                            formatted.append(f"• Name: {info.get('name', 'N/A')}")
                            formatted.append(f"  CUI: {info.get('ui', 'N/A')}")
                            formatted.append(f"  Type: {info.get('semanticType', 'N/A')}")
                            if concept.get('definitions'):
                                formatted.append(f"  Definition: {concept['definitions'][0].get('value', 'N/A')}")
            
            # Format relationships
            if 'relationships' in umls_data and umls_data['relationships']:
                formatted.append("\nKey Relationships:")
                seen_relationships = set()  # To avoid duplicates
                for rel in umls_data['relationships']:
                    if isinstance(rel, dict):
                        rel_str = f"{rel.get('sourceName', '')} -> {rel.get('relationLabel', '')} -> {rel.get('relatedName', '')}"
                        if rel_str not in seen_relationships and all(x in rel_str for x in ['', '->', '']):
                            seen_relationships.add(rel_str)
                            formatted.append(f"• {rel_str}")
            
            result = "\n".join(formatted)
            logger.info(f"Formatted {len(umls_data.get('concepts', {}))} concepts and {len(umls_data.get('relationships', []))} relationships")
            return result
            
        except Exception as e:
            logger.error(f"Error formatting UMLS evidence: {e}")
            logger.error(f"UMLS data that caused error: {umls_data}")
            return "=== UMLS Evidence ===\nError formatting UMLS data"

    def _format_textbook_evidence(self, textbook_data: List[Dict[str, Any]]) -> str:
        """Format textbook evidence for the prompt"""
        formatted = ["Relevant Textbook Passages:"]
        
        for i, chunk in enumerate(textbook_data, 1):
            formatted.append(f"\nPassage {i} (Source: {chunk['sources']}):")
            formatted.append(f"Relevance Score: {chunk['score']:.2f}")
            formatted.append(f"Content: {chunk['text'][:500]}...")
        
        return "\n".join(formatted)

async def main():
    qa_system = UnifiedMedicalQA()
    
    print("Welcome to the Unified Medical QA System!")
    print("Enter your USMLE question (or 'quit' to exit)")
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() == 'quit':
            break
            
        try:
            print("\nProcessing question...")
            result = await qa_system.process_question(question)
            
            print("\n=== Integrated Analysis ===")
            print(result['integrated_analysis'])
            
            # Optional: Print detailed evidence
            print("\nWould you like to see the detailed evidence? (y/n)")
            if input().lower() == 'y':
                print("\n=== UMLS Evidence ===")
                print(result['umls_evidence'])
                print("\n=== Textbook Evidence ===")
                print(result['textbook_evidence'])
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 