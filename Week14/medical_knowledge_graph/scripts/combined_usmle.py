import sys
from pathlib import Path
import os
from typing import Dict, List, Any
import json
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
import time

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.processors.question_processor import QuestionProcessor
from process_question import MedicalQuestionProcessor
from query_medical_db import combine_similar_chunks

class USMLEProcessor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Neo4j
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j"
        )
        
        # Initialize OpenAI
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        self.pinecone_index = self.pc.Index("medical-textbook-embeddings")
        
        # Initialize processors
        self.kg_processor = MedicalQuestionProcessor(
            graph=self.graph,
            llm_function=lambda x: self.llm.invoke(x).content
        )
        # 1. ANSWER CHOICE: [State the correct answer choice letter]
        self.answer_prompt = """You are a medical expert tasked with answering USMLE questions. You have access to two knowledge sources:
        1. A structured knowledge graph (Neo4j) containing UMLS medical concepts and relationships
        2. Medical textbook passages (retrieved via semantic search)

        Question: {question}

        Knowledge Graph Evidence:
        {kg_evidence}

        Textbook Evidence:
        {textbook_evidence}

        Please provide a detailed answer using ONLY the provided evidence following this format:

        

        1. REASONING PROCESS:
        - Initial Understanding: [Break down the question and what it's asking]
        - Key Findings: [List the relevant facts from both knowledge sources]
        - Chain of Thought: [Explain step-by-step how the facts from the evidence lead to the answer. Do not look for the answer in the evidence, but rather use the evidence to reason to the answer.]
        
        
        2. EVIDENCE USED:
        - Knowledge Graph: [Cite specific concepts and relationships used]
        - Textbook References: [Quote relevant passages with reference numbers]
        
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

    def process_question(self, question: str) -> Dict:
        """Process a USMLE question using both knowledge sources"""
        
        # 1. Query Knowledge Graph
        kg_results = self.kg_processor.process_medical_question(question)
        
        # 2. Query Pinecone
        query_embedding = self.embeddings.embed_query(question)
        pinecone_results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=8,
            include_metadata=True
        )
        
        # Process and combine similar chunks
        chunks = []
        for match in pinecone_results.matches:
            if match.score < 0.5:
                continue
            chunks.append({
                'text': match.metadata.get('text', '').strip(),
                'source': match.metadata.get('source', ''),
                'score': match.score
            })
        
        combined_chunks = combine_similar_chunks(chunks)
        
        # 3. Generate comprehensive answer
        answer = self.llm.invoke(
            self.answer_prompt.format(
                question=question,
                kg_evidence=self._format_kg_evidence(kg_results),
                textbook_evidence=self._format_textbook_evidence(combined_chunks)
            )
        )
        
        return {
            'answer': answer.content,
            'kg_results': kg_results,
            'textbook_results': combined_chunks
        }

    def _format_kg_evidence(self, kg_results: Dict) -> str:
        """Format knowledge graph evidence for LLM consumption"""
        evidence = []
        
        # Add concepts
        if kg_results.get('concepts'):
            evidence.append("Medical Concepts:")
            for concept in kg_results['concepts']:
                evidence.append(f"- {concept['term']} (CUI: {concept['cui']})")
                if concept.get('definition'):
                    evidence.append(f"  Definition: {concept['definition']}")
        
        # Add relationships
        if kg_results.get('relationships'):
            evidence.append("\nRelationships:")
            for rel in kg_results['relationships']:
                evidence.append(
                    f"- {rel['source_name']} -> {rel['relationship_type']} -> {rel['target_name']}"
                )
        
        return "\n".join(evidence)

    def _format_textbook_evidence(self, chunks: List[Dict]) -> str:
        """Format textbook evidence for LLM consumption"""
        evidence = []
        
        for i, chunk in enumerate(chunks, 1):
            evidence.append(f"[REF{i}] Source: {chunk['sources']}")
            evidence.append(f"Relevance Score: {chunk['score']:.2f}")
            evidence.append(f"Content: {chunk['text']}\n")
        
        return "\n".join(evidence)

def main():
    processor = USMLEProcessor()
    
    print("\n🏥 Welcome to the USMLE Question Processor!")
    print("Type 'quit' or 'exit' to end the session")
    
    while True:
        print("\n📝 Enter your USMLE question:")
        question = input().strip()
        
        if question.lower() in ['quit', 'exit']:
            break
        
        print("\n🔍 Processing question...")
        try:
            result = processor.process_question(question)
            print("\n📑 Analysis:")
            print(result['answer'])
            
        except Exception as e:
            print(f"Error processing question: {e}")
    
    print("\nThank you for using the USMLE Question Processor!")

if __name__ == "__main__":
    main()