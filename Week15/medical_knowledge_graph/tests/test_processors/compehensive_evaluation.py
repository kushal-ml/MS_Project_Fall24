import json
import time
import logging
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(root_dir)
from test_umls_processor import KnowledgeGraphEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGEvaluator:
    def __init__(self, kg_processor, llm, vector_store, output_dir: str = "evaluation_results"):
        """
        Initialize the GraphRAGEvaluator.
        
        Args:
            kg_processor: The knowledge graph processor
            llm: The LLM function to use for generating answers
            vector_store: The vector store for RAG retrieval
            output_dir: Directory to save evaluation results
        """
        self.kg_processor = kg_processor
        self.llm = llm
        self.vector_store = vector_store
        self.output_dir = output_dir
        self.results = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize embedding model and vector DB connection
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        self.index = self.pc.Index("medical-textbook-embeddings")
        
        # Define evaluation configurations
        self.configs = ["llm_only", "kg_only", "rag_only", "context_strict", "llm_informed"]

    def load_questions(self, file_path: str, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Load questions from a JSON file.
        
        Args:
            file_path: Path to the questions file
            sample_size: Number of random questions to sample
            
        Returns:
            List of question dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            if not isinstance(questions, list):
                logger.error("Questions file must contain a list")
                return []
            if sample_size and sample_size < len(questions):
                import random
                questions = random.sample(questions, sample_size)
            logger.info(f"Loaded {len(questions)} questions from {file_path}")
            return questions
        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            return []

    def evaluate_all_questions(self, questions: List[Dict], parallel: bool = True) -> List[Dict]:
        """
        Evaluate all questions in the dataset.
        
        Args:
            questions: List of question dictionaries
            parallel: Whether to run evaluation in parallel
            
        Returns:
            List of results for each question
        """
        results = []
        if parallel and len(questions) > 1:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.evaluate_question, q): q for q in questions}
                for future in tqdm(as_completed(futures), total=len(questions), desc="Evaluating"):
                    results.append(future.result())
        else:
            for q in tqdm(questions, desc="Evaluating"):
                results.append(self.evaluate_question(q))
        
        self.results = results
        # Save detailed results
        with open(f"{self.output_dir}/detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def evaluate_question(self, question: Dict) -> Dict:
        """
        Evaluate a single question across all configurations.
        
        Args:
            question: Question dictionary
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            q_text = question["question"]
            options = question.get("options", {})
            correct_answer = question.get("answer", "")
            domain = question.get("domain", "")
            step = question.get("step", "")
            
            formatted_q = f"{q_text}\n\n" + "\n".join([f"{key}. {value}" for key, value in options.items()]) if options else q_text

            # Knowledge Graph Context (UMLS)
            terms = self.kg_processor.extract_key_terms(q_text)
            concepts = self.kg_processor.rerank_concepts(self.kg_processor.get_concepts(terms), q_text)
            relationships = self.kg_processor.rerank_relationships(self.kg_processor.get_relationships(concepts), q_text, concepts)
            multihop_paths = self.kg_processor.rerank_paths(self.kg_processor.find_multihop_paths(concepts, max_depth=3), q_text)
            kg_context = self.kg_processor.format_kg_data(concepts, relationships, multihop_paths)

            # RAG Context (MedQA)
            rag_chunks = self._get_textbook_context(q_text, top_k=8)

            # Generate Answers for each configuration
            config_results = {}
            
            # LLM Only
            llm_only_answer = self._generate_llm_only_answer(formatted_q)
            llm_only_result = self._evaluate_configuration(
                "llm_only", 
                llm_only_answer, 
                formatted_q,
                correct_answer,
                None,  # No KG context
                None   # No RAG context
            )
            config_results["llm_only"] = llm_only_result
            
            # KG Only
            kg_only_answer = self._generate_kg_only_answer(formatted_q, concepts, relationships, multihop_paths)
            kg_only_result = self._evaluate_configuration(
                "kg_only", 
                kg_only_answer, 
                formatted_q,
                correct_answer,
                {"concepts": concepts, "relationships": relationships, "paths": multihop_paths},
                None  # No RAG context
            )
            config_results["kg_only"] = kg_only_result
            
            # RAG Only
            rag_only_answer = self._generate_rag_only_answer(formatted_q, rag_chunks)
            rag_only_result = self._evaluate_configuration(
                "rag_only", 
                rag_only_answer, 
                formatted_q,
                correct_answer,
                None,  # No KG context
                rag_chunks
            )
            config_results["rag_only"] = rag_only_result
            
            # Context Strict (KG + RAG, no LLM knowledge)
            context_strict_answer = self._generate_context_strict_answer(formatted_q, concepts, relationships, multihop_paths, rag_chunks)
            context_strict_result = self._evaluate_configuration(
                "context_strict", 
                context_strict_answer, 
                formatted_q,
                correct_answer,
                {"concepts": concepts, "relationships": relationships, "paths": multihop_paths},
                rag_chunks
            )
            config_results["context_strict"] = context_strict_result
            
            # LLM Informed (KG + RAG + LLM knowledge)
            llm_informed_answer = self._generate_llm_informed_answer(formatted_q, concepts, relationships, multihop_paths, rag_chunks)
            llm_informed_result = self._evaluate_configuration(
                "llm_informed", 
                llm_informed_answer, 
                formatted_q,
                correct_answer,
                {"concepts": concepts, "relationships": relationships, "paths": multihop_paths},
                rag_chunks
            )
            config_results["llm_informed"] = llm_informed_result

            return {
                "question": q_text,
                "options": options,
                "correct_answer": correct_answer,
                "domain": domain,
                "step": step,
                "results": config_results
            }
        except Exception as e:
            logger.error(f"Error evaluating question: {str(e)}")
            return {
                "question": question.get("question", ""), 
                "options": question.get("options", {}),
                "correct_answer": question.get("answer", ""),
                "domain": question.get("domain", ""),
                "step": question.get("step", ""),
                "error": str(e)
            }

    def _evaluate_configuration(self, 
                               config: str, 
                               answer: str, 
                               question: str, 
                               correct_answer: str,
                               kg_context: Optional[Dict] = None,
                               rag_context: Optional[List] = None) -> Dict:
        """
        Evaluate a specific configuration's answer.
        
        Args:
            config: Configuration name
            answer: Generated answer
            question: Original question
            correct_answer: Correct answer
            kg_context: Knowledge graph context if available
            rag_context: RAG context if available
            
        Returns:
            Evaluation results for this configuration
        """
        # Extract the chosen answer
        chosen_answer = self._extract_answer_choice(answer)
        is_correct = chosen_answer == correct_answer if chosen_answer and correct_answer else False
        
        # Evaluate context relevance
        context_relevance = self._evaluate_context_relevance(config, question, answer, kg_context, rag_context)
        
        # Evaluate citation quality
        citation_quality = self._evaluate_citation_quality(config, answer, kg_context, rag_context)
        
        # Evaluate reasoning quality
        reasoning_score = self._evaluate_reasoning_quality(config, question, answer)
        
        # Calculate final score (weighted average)
        weights = {
            "accuracy": 0.4,
            "context_relevance": 0.2,
            "citation_quality": 0.2,
            "reasoning": 0.2
        }
        
        accuracy_score = 10 if is_correct else 0
        
        final_score = (
            weights["accuracy"] * accuracy_score +
            weights["context_relevance"] * context_relevance +
            weights["citation_quality"] * citation_quality +
            weights["reasoning"] * reasoning_score
        )
        
        return {
            "answer": answer,
            "chosen_option": chosen_answer,
            "accuracy_score": accuracy_score,
            "context_relevance_score": context_relevance,
            "citation_quality_score": citation_quality,
            "reasoning_score": reasoning_score,
            "final_score": round(final_score, 2)
        }

    def _generate_llm_only_answer(self, question: str) -> str:
        """Generate answer using only LLM's internal knowledge"""
        prompt = f"""
        Answer the following medical question using ONLY your internal medical knowledge:

        {question}

        Format your answer:
        1. ANSWER CHOICE: [e.g., A]
        2. REASONING: [Detailed explanation based on your knowledge]
        """
        return self.llm(prompt)

    def _generate_kg_only_answer(self, question: str, concepts: List[Dict], relationships: List[Dict], multihop_paths: List[Dict]) -> str:
        """Generate answer using only knowledge graph data"""
        kg_data = self.kg_processor.format_kg_data_for_prompt(concepts, relationships, multihop_paths)
        prompt = f"""
        Answer the following medical question using ONLY the provided UMLS knowledge graph data:

        Question: {question}

        Knowledge Graph Data:
        {kg_data}

        IMPORTANT: Use ONLY the provided data. Do NOT use any internal medical knowledge not in the graph.

        Format your answer:
        1. ANSWER CHOICE: [e.g., A]
        2. REASONING: [Explanation with explicit citations from the graph, e.g., Concept ID: C123, Relationship: R456]
        """
        return self.llm(prompt)

    def _generate_rag_only_answer(self, question: str, chunks: List[Dict]) -> str:
        """Generate answer using only RAG data"""
        formatted_chunks = "\n\n".join([f"DOCUMENT {i+1}:\n{chunk.get('text', '')}" for i, chunk in enumerate(chunks)]) if chunks else "No RAG data available."
        prompt = f"""
        Answer the following medical question using ONLY the provided MedQA context:

        Question: {question}

        Context:
        {formatted_chunks}

        IMPORTANT: Use ONLY the provided context. Do NOT use any internal medical knowledge not in the documents.

        Format your answer:
        1. ANSWER CHOICE: [e.g., A]
        2. REASONING: [Explanation citing specific document content by referring to DOCUMENT X]
        """
        return self.llm(prompt)

    def _generate_context_strict_answer(self, question: str, concepts: List[Dict], relationships: List[Dict], multihop_paths: List[Dict], rag_chunks: List) -> str:
        """Generate answer using only KG and RAG data, no internal knowledge"""
        kg_data = self.kg_processor.format_kg_data_for_prompt(concepts, relationships, multihop_paths)
        rag_data = "\n\n".join([f"DOCUMENT {i+1}:\n{chunk.get('text', '')}" for i, chunk in enumerate(rag_chunks)]) if rag_chunks else "No RAG data."
        prompt = f"""
        Answer the following medical question using ONLY the provided UMLS graph and MedQA RAG context:

        Question: {question}

        Knowledge Graph Data:
        {kg_data}

        RAG Context:
        {rag_data}

        IMPORTANT: Use ONLY the provided data. Do NOT use any internal medical knowledge not in the context.

        Format your answer:
        1. ANSWER CHOICE: [e.g., A]
        2. REASONING: [Explanation citing specific graph data (e.g., Concept ID: C123) and RAG data (e.g., DOCUMENT 2)]
        """
        return self.llm(prompt)

    def _generate_llm_informed_answer(self, question: str, concepts: List[Dict], relationships: List[Dict], multihop_paths: List[Dict], rag_chunks: List) -> str:
        """Generate answer using KG, RAG, and internal LLM knowledge"""
        kg_data = self.kg_processor.format_kg_data_for_prompt(concepts, relationships, multihop_paths)
        rag_data = "\n\n".join([f"DOCUMENT {i+1}:\n{chunk.get('text', '')}" for i, chunk in enumerate(rag_chunks)]) if rag_chunks else "No RAG data."
        prompt = f"""
        Answer the following medical question using the provided UMLS graph and MedQA RAG context, supplemented by your internal knowledge:

        Question: {question}

        Knowledge Graph Data:
        {kg_data}

        RAG Context:
        {rag_data}

        Instructions:
        - Prioritize provided context when available
        - Use your internal knowledge where context is incomplete
        - Always cite your sources explicitly:
          * For KG data: Cite Concept ID or Relationship ID
          * For RAG data: Cite specific DOCUMENT numbers
          * For your internal knowledge: State "Based on medical knowledge..."

        Format your answer:
        1. ANSWER CHOICE: [e.g., A]
        2. REASONING: [Explanation with explicit citations of all sources used]
        """
        return self.llm(prompt)

    def _extract_answer_choice(self, response: str) -> Optional[str]:
        """Extract the chosen answer from the response"""
        # Try to find answer choice in various formats
        patterns = [
            r"(?:ANSWER CHOICE|The correct answer is|I choose option|My answer is):\s*([A-E])",
            r"^\s*([A-E])\s*\.",
            r"answer:\s*([A-E])",
            r"Answer:\s*([A-E])"
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
                
        # Look for options at word boundaries
        options = re.findall(r'\b([A-E])\b', response)
        return options[0].upper() if options else None

    def _evaluate_context_relevance(self, 
                                   config: str, 
                                   question: str, 
                                   answer: str, 
                                   kg_context: Optional[Dict] = None, 
                                   rag_context: Optional[List] = None) -> float:
        """
        Evaluate how relevant the provided context is to the question and answer.
        
        Returns:
            Score from 0-10
        """
        if config == "llm_only":
            # No external context for llm_only
            return 10.0
            
        prompt = f"""
        Evaluate the relevance of the provided context to the question and answer.
        
        Question: {question}
        Answer: {answer}
        
        """
        
        if kg_context and config in ["kg_only", "context_strict", "llm_informed"]:
            kg_data = self.kg_processor.format_kg_data_for_prompt(
                kg_context.get("concepts", []), 
                kg_context.get("relationships", []), 
                kg_context.get("paths", [])
            )
            prompt += f"""
            Knowledge Graph Context:
            {kg_data}
            """
            
        if rag_context and config in ["rag_only", "context_strict", "llm_informed"]:
            rag_data = "\n\n".join([f"DOCUMENT {i+1}:\n{chunk.get('text', '')}" for i, chunk in enumerate(rag_context)])
            prompt += f"""
            RAG Context:
            {rag_data}
            """
            
        prompt += """
        Rate the context relevance on a scale of 0-10:
        - 0-2: Completely irrelevant
        - 3-4: Marginally relevant
        - 5-6: Somewhat relevant
        - 7-8: Highly relevant
        - 9-10: Perfectly relevant
        
        Return ONLY a number between 0 and 10.
        """
        
        try:
            response = self.llm(prompt)
            # Extract numeric score
            score_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0), 10)  # Ensure score is between 0-10
            return 5.0  # Default if no score found
        except Exception as e:
            logger.error(f"Error evaluating context relevance: {str(e)}")
            return 5.0  # Default score on error

    def _evaluate_citation_quality(self, 
                                 config: str, 
                                 answer: str, 
                                 kg_context: Optional[Dict] = None, 
                                 rag_context: Optional[List] = None) -> float:
        """
        Evaluate the quality of citations in the answer.
        
        Returns:
            Score from 0-10
        """
        if config == "llm_only":
            # No citation expected for llm_only
            return 10.0
            
        prompt = f"""
        Evaluate the quality of citations in the following answer:
        
        Answer: {answer}
        
        Configuration: {config}
        
        Citation Expectations:
        """
        
        if config == "kg_only":
            prompt += "- Should cite specific Knowledge Graph concepts and relationships (e.g., Concept ID: C123)"
        elif config == "rag_only":
            prompt += "- Should cite specific RAG documents (e.g., DOCUMENT 1)"
        elif config in ["context_strict", "llm_informed"]:
            prompt += """
            - Should cite both Knowledge Graph concepts (e.g., Concept ID: C123) and RAG documents (e.g., DOCUMENT 1)
            - Citations should be specific and accurate
            """
            
        prompt += """
        Rate the citation quality on a scale of 0-10:
        - 0-2: No citations or completely incorrect citations
        - 3-4: Few citations, mostly vague
        - 5-6: Some citations but lacking specificity
        - 7-8: Good citations, mostly specific
        - 9-10: Excellent citations, very specific and comprehensive
        
        Return ONLY a number between 0 and 10.
        """
        
        try:
            response = self.llm(prompt)
            # Extract numeric score
            score_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0), 10)  # Ensure score is between 0-10
            return 5.0  # Default if no score found
        except Exception as e:
            logger.error(f"Error evaluating citation quality: {str(e)}")
            return 5.0  # Default score on error

    def _evaluate_reasoning_quality(self, config: str, question: str, answer: str) -> float:
        """
        Evaluate the quality of reasoning in the answer.
        
        Returns:
            Score from 0-10
        """
        prompt = f"""
        Evaluate the quality of reasoning in the following answer:
        
        Question: {question}
        Answer: {answer}
        
        Configuration: {config}
        
        Consider:
        - Logical coherence and flow
        - Completeness of explanation
        - Clarity of writing
        - Appropriateness of medical terminology
        - Depth of analysis
        
        Rate the reasoning quality on a scale of 0-10:
        - 0-2: Poor reasoning, major issues
        - 3-4: Basic reasoning with significant gaps
        - 5-6: Adequate reasoning
        - 7-8: Good reasoning, clear and mostly complete
        - 9-10: Excellent reasoning, thorough and insightful
        
        Return ONLY a number between 0 and 10.
        """
        
        try:
            response = self.llm(prompt)
            # Extract numeric score
            score_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0), 10)  # Ensure score is between 0-10
            return 5.0  # Default if no score found
        except Exception as e:
            logger.error(f"Error evaluating reasoning quality: {str(e)}")
            return 5.0  # Default score on error

    def _get_textbook_context(self, question: str, top_k: int) -> List[Dict]:
        """Retrieve relevant context from the vector database"""
        try:
            query_embedding = self.embeddings.embed_query(question)
            results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            return [{"text": m.metadata.get("text", ""), "score": m.score} for m in results.matches if m.score >= 0.5]
        except Exception as e:
            logger.error(f"Error retrieving RAG context: {str(e)}")
            return []

    def generate_summary_report(self) -> Dict:
        """
        Generate a summary report of evaluation results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"summary": "No results to summarize"}
        
        summary = {
            "total_questions": len(self.results),
            "configurations": {}
        }
        
        # Calculate metrics for each configuration
        for config in self.configs:
            config_data = {
                "accuracy": {
                    "correct": 0,
                    "percentage": 0.0
                },
                "average_scores": {
                    "context_relevance": 0.0,
                    "citation_quality": 0.0,
                    "reasoning": 0.0,
                    "final": 0.0
                }
            }
            
            # Collect scores
            scores = {
                "context_relevance": [],
                "citation_quality": [],
                "reasoning": [],
                "final": []
            }
            
            correct_count = 0
            total_valid = 0
            
            for result in self.results:
                if "error" in result:
                    continue
                    
                if config in result.get("results", {}):
                    total_valid += 1
                    config_result = result["results"][config]
                    
                    # Count correct answers
                    if config_result.get("chosen_option") == result.get("correct_answer"):
                        correct_count += 1
                    
                    # Collect scores
                    scores["context_relevance"].append(config_result.get("context_relevance_score", 0))
                    scores["citation_quality"].append(config_result.get("citation_quality_score", 0))
                    scores["reasoning"].append(config_result.get("reasoning_score", 0))
                    scores["final"].append(config_result.get("final_score", 0))
            
            # Calculate averages
            if total_valid > 0:
                config_data["accuracy"]["correct"] = correct_count
                config_data["accuracy"]["percentage"] = round((correct_count / total_valid) * 100, 2)
                
                for score_type, score_list in scores.items():
                    if score_list:
                        config_data["average_scores"][score_type] = round(sum(score_list) / len(score_list), 2)
            
            summary["configurations"][config] = config_data
        
        # Save summary report
        with open(f"{self.output_dir}/summary_report.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary

    def run_evaluation_from_file(self, questions_file: str, sample_size: Optional[int] = None, parallel: bool = True) -> Dict:
        """
        Run the full evaluation process from a questions file.
        
        Args:
            questions_file: Path to the questions JSON file
            sample_size: Number of questions to sample
            parallel: Whether to run in parallel
            
        Returns:
            Summary report dictionary
        """
        questions = self.load_questions(questions_file, sample_size)
        if not questions:
            return {"error": "No questions loaded"}
            
        results = self.evaluate_all_questions(questions, parallel)
        return self.generate_summary_report()

def main():
    import argparse
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_neo4j import Neo4jGraph
    import os
    
    parser = argparse.ArgumentParser(description="Evaluate Graph RAG Pipeline")
    parser.add_argument('--questions', type=str, required=True, help='Path to MedQA JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    parser.add_argument('--sample', type=int, help='Number of questions to sample')
    args = parser.parse_args()
    
    load_dotenv()
    
    # Initialize Neo4j connection
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"), 
        username=os.getenv("NEO4J_USERNAME"), 
        password=os.getenv("NEO4J_PASSWORD"), 
        database="neo4j"
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1, 
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    llm_function = lambda x: llm.invoke(x).content
    
    # Initialize KG processor
    kg_processor = KnowledgeGraphEvaluator(graph=graph, llm_function=llm_function)
    
    # Run evaluation
    evaluator = GraphRAGEvaluator(
        kg_processor=kg_processor, 
        llm=llm_function, 
        vector_store=None, 
        output_dir=args.output
    )
    
    summary = evaluator.run_evaluation_from_file(args.questions, args.sample, args.parallel)
    
    logger.info(f"Evaluation complete. Results in {args.output}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()