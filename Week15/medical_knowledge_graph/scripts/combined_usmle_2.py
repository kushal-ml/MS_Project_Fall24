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
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from langsmith import trace
from langsmith.run_helpers import traceable
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langsmith import Client
# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

# Import the KnowledgeGraphEvaluator from test_umls_processor
from tests.test_processors.test_umls_processor import KnowledgeGraphEvaluator
from query_medical_db import combine_similar_chunks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "medgraphrag")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

def get_benchmark_usmle_questions() -> List[Dict]:
    """
    Returns a list of USMLE-style questions from the usmle_categorized_questions.json file
    """
    try:
        # Path to the JSON file with the questions
        questions_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "tests", 
            "test_processors", 
            "usmle_categorized_questions.json"
        )
        
        # Load the questions from the file
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        # Log the number of questions loaded
        logger.info(f"Loaded {len(questions)} questions from usmle_categorized_questions.json")
        
        # Normalize the format if needed
        for question in questions:
            # Convert options to dictionary if it's a list
            if isinstance(question.get("options"), list):
                options_dict = {}
                for i, option in enumerate(question["options"]):
                    options_dict[chr(65 + i)] = option  # A, B, C, etc.
                question["options"] = options_dict
        
        return questions
    except Exception as e:
        logger.error(f"Error loading benchmark questions: {e}")
        return []

class USMLEProcessor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize LangSmith tracing
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "usmle_processor")
        
        # Explicitly initialize LangSmith client
        self.langsmith_client = Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        )
        
        # Initialize Neo4j
        try:
            logger.info("Initializing Neo4j connection...")
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD"),
                database="neo4j"
            )
            # Test the connection
            test_query = "MATCH (n) RETURN count(n) as count LIMIT 1"
            result = self.graph.query(test_query)
            node_count = result[0]["count"]
            logger.info(f"Successfully connected to Neo4j. Database contains {node_count} nodes.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
        # Initialize OpenAI with o3-mini model
        self.llm = ChatOpenAI(
            model="o3-mini", 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Function to pass to KG Evaluator
        self.llm_function = lambda x: self.llm.invoke(x).content
        
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
        
        # Initialize KnowledgeGraphEvaluator
        try:
            logger.info("Initializing KnowledgeGraphEvaluator...")
            self.kg_evaluator = KnowledgeGraphEvaluator(
                graph=self.graph,
                llm_function=self.llm_function
            )
            logger.info("KnowledgeGraphEvaluator initialized successfully")
            logger.info(f"KG Evaluator settings: {self.kg_evaluator.settings}")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphEvaluator: {str(e)}")
            raise
        
        # Configure KG Evaluator settings for optimal performance
        self.kg_evaluator.settings = {
            "top_k_concepts": 50,  
            "top_k_relationships": 100,  
            "vector_search_enabled": True,
            "multihop_enabled": True, 
            "multihop_max_depth": 3,
            "vector_search_threshold": 0.3,
            "visualization_dir": "usmle_evaluation_results"  # Make sure this path is accessible
        }
        
        # Store evaluation results
        self.evaluation_results = []
        
        # Initialize benchmark questions
        self.benchmark_questions = get_benchmark_usmle_questions()

    @traceable(run_type="chain")
    def process_question(self, question_input):
        """Process a USMLE-style question with knowledge graph and textbook context"""
        
        # Handle both string input and dictionary input
        if isinstance(question_input, dict):
            question_dict = question_input
            question = question_dict.get('question', '')
        else:
            question_dict = {'question': question_input}
            question = question_input
        
        logger.info(f"Processing question: {question[:100]}...")
        
        # Debug: Check if KG evaluator is properly initialized
        logger.info(f"KG Evaluator type: {type(self.kg_evaluator)}")
        logger.info(f"KG Evaluator settings: {self.kg_evaluator.settings}")
        
        # Extract terms
        start_time = time.time()
        extracted_terms = self.kg_evaluator.extract_key_terms(question)
        term_extraction_time = time.time() - start_time
        logger.info(f"Extracted terms: {extracted_terms}")
        
        # Get concepts
        start_time = time.time()
        concepts = self.kg_evaluator.get_concepts(extracted_terms)
        concept_retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(concepts)} concepts")
        logger.info(f"Sample concepts: {concepts[:2]}")  # Show first 2 concepts
        
        # Re-rank concepts
        start_time = time.time()
        concepts = self.kg_evaluator.rerank_concepts(concepts, question)
        concept_rerank_time = time.time() - start_time
        logger.info(f"Reranked concepts, top 3: {concepts[:3]}")
        
        # Get relationships
        start_time = time.time()
        relationships = self.kg_evaluator.get_relationships(concepts)
        relationship_retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(relationships)} relationships")
        logger.info(f"Sample relationships: {relationships[:2]}")
        
        # Re-rank relationships
        start_time = time.time()
        relationships = self.kg_evaluator.rerank_relationships(relationships, question, concepts)
        relationship_rerank_time = time.time() - start_time
        logger.info(f"Reranked relationships, top 3: {relationships[:3]}")
        
        # Find multihop paths
        start_time = time.time()
        multihop_paths = self.kg_evaluator.find_multihop_paths(concepts)
        multihop_path_finding_time = time.time() - start_time
        logger.info(f"Found {len(multihop_paths)} multihop paths")
        logger.info(f"Sample paths: {multihop_paths[:2]}")
        
        # Re-rank paths
        start_time = time.time()
        multihop_paths = self.kg_evaluator.rerank_paths(multihop_paths, question)
        multihop_rerank_time = time.time() - start_time
        
        # Format knowledge graph data for LLM context
        formatted_knowledge = self.format_kg_data_for_llm(concepts, relationships, multihop_paths)
        logger.info("Formatted KG data length: %d chars", len(formatted_knowledge))
        logger.info("First 500 chars of formatted KG data: %s", formatted_knowledge[:500])
        
        # Debug: Check if formatted knowledge is empty
        if not formatted_knowledge.strip():
            logger.warning("WARNING: Formatted knowledge graph data is empty!")
            logger.warning("Concepts count: %d", len(concepts))
            logger.warning("Relationships count: %d", len(relationships))
            logger.warning("Multihop paths count: %d", len(multihop_paths))
        
        # Textbook retrieval
        start_time = time.time()
        textbook_context = self.retrieve_from_pinecone(question)
        pinecone_time = time.time() - start_time
        logger.info("Retrieved textbook context length: %d chars", len(textbook_context))
        
        # Save the KG context to file for debugging
        debug_context_file = self.save_kg_context_to_file(formatted_knowledge, question_dict.get('id', None))
        logger.info(f"Saved debug context to: {debug_context_file}")
        
        # Generate answers with debug info
        logger.info("Generating LLM-only answer...")
        llm_only_answer = self.generate_llm_only_answer(self.format_question_with_options(question_dict))
        
        logger.info("Generating context-strict answer...")
        context_strict_answer = self.generate_context_strict_answer(
            self.format_question_with_options(question_dict),
            formatted_knowledge,
            textbook_context
        )
        
        logger.info("Generating LLM-informed answer...")
        combined_answer = self.generate_llm_informed_answer(
            self.format_question_with_options(question_dict),
            formatted_knowledge,
            textbook_context
        )
        
        # Timing information
        timing_info = {
            'term_extraction': term_extraction_time,
            'concept_retrieval': concept_retrieval_time,
            'concept_reranking': concept_rerank_time,
            'relationship_retrieval': relationship_retrieval_time,
            'relationship_reranking': relationship_rerank_time,
            'multihop_path_finding': multihop_path_finding_time,
            'multihop_reranking': multihop_rerank_time,
            'pinecone_retrieval': pinecone_time,
            'llm_informed_generation': (pinecone_time + concept_retrieval_time + concept_rerank_time + 
                                       relationship_retrieval_time + relationship_rerank_time +
                                       multihop_path_finding_time + multihop_rerank_time),
            'total_processing': (term_extraction_time + concept_retrieval_time + concept_rerank_time + 
                               relationship_retrieval_time + relationship_rerank_time +
                               multihop_path_finding_time + multihop_rerank_time +
                               pinecone_time + (pinecone_time + concept_retrieval_time + concept_rerank_time + 
                                               relationship_retrieval_time + relationship_rerank_time +
                                               multihop_path_finding_time + multihop_rerank_time))
        }
        
        # Prepare results
        result = {
            'question': question,
            'kg_results': {
                'terms': extracted_terms,
                'concepts': concepts,
                'relationships': relationships,
                'multihop_paths': multihop_paths,
                'formatted_data': formatted_knowledge
            },
            'pinecone_results': textbook_context,
            'answers': {
                'llm_only': llm_only_answer,
                'context_strict': context_strict_answer,
                'llm_informed': combined_answer
            },
            'timing': timing_info
        }
        
        return result

    def save_kg_context_to_file(self, kg_context, question_id=None):
        """Save the knowledge graph context to a file for analysis"""
        # Create directory if it doesn't exist
        kg_context_dir = os.path.join(self.kg_evaluator.settings.get('visualization_dir', 'evaluation_results'), 'kg_contexts')
        os.makedirs(kg_context_dir, exist_ok=True)
        
        # Create a unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        question_id = question_id or f"question_{timestamp}"
        filename = os.path.join(kg_context_dir, f"kg_context_{question_id}.json")
        
        # Add some statistics about the context
        context_stats = {
            "timestamp": timestamp,
            "question_id": question_id,
            "context_size": len(kg_context),
            "context": kg_context,
            # Parse the context to count entities
            "stats": {
                "concept_count": kg_context.count("Concept:"),
                "relationship_count": kg_context.count("Relationship:"),
                "path_count": kg_context.count("Path:") 
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(context_stats, f, indent=2)
        
        logger.info(f"Knowledge graph context saved to {filename}")
        return filename

    def format_kg_data_for_llm(self, concepts, relationships, multihop_paths):
        """Format knowledge graph data for LLM consumption with consistent formatting"""
        formatted_text = []
        
        # Format concepts section
        formatted_text.append("=== MEDICAL CONCEPTS ===")
        for i, concept in enumerate(concepts[:30], 1):  # Limit to top 30 concepts
            cui = concept.get('cui', 'Unknown')
            term = concept.get('term', 'Unknown')
            definition = concept.get('definition', 'No definition available')
            formatted_text.append(f"{i}. {cui} - {term}")
            formatted_text.append(f"   Definition: {definition}\n")
        
        # Format relationships section
        formatted_text.append("=== RELATIONSHIPS ===")
        for i, rel in enumerate(relationships[:50], 1):  # Limit to top 50 relationships
            source = rel.get('source_name', 'Unknown')
            rel_type = rel.get('relationship_type', 'related_to')
            target = rel.get('target_name', 'Unknown')
            formatted_text.append(f"{i}. {source} → {rel_type} → {target}")
        
        # Format multihop paths section
        if multihop_paths:
            formatted_text.append("\n=== COMPLEX KNOWLEDGE PATHS ===")
            for i, path in enumerate(multihop_paths[:20], 1):  # Limit to top 20 paths
                path_desc = path.get('path_description', 'Unknown path')
                formatted_text.append(f"{i}. {path_desc}")
        
        return "\n\n".join(formatted_text)

    def _format_textbook_evidence(self, chunks: List[Dict]) -> str:
        """Format textbook evidence for LLM consumption"""
        evidence = []
        
        for i, chunk in enumerate(chunks, 1):
            evidence.append(f"[REF{i}] Source: {chunk.get('sources', 'Unknown')}")
            evidence.append(f"Relevance Score: {chunk.get('score', 0):.2f}")
            evidence.append(f"Content: {chunk.get('text', '')}\n")
        
        return "\n".join(evidence)
    
    def evaluate_evidence_based_answer(self, result: Dict) -> Dict:
        """Evaluate the answer using hybrid approach (semantic similarity + evidence quality)"""
        logger.info("Evaluating answer with hybrid approach...")
        
        question = result['question']
        llm_only_answer = result['answers']['llm_only']
        context_strict_answer = result['answers']['context_strict']
        llm_informed_answer = result['answers']['llm_informed']
        
        # Prepare structured kg_data for evaluation
        structured_kg_data = {
            'concepts': result['kg_results']['concepts'],
            'relationships': result['kg_results']['relationships'],
            'multihop_paths': result['kg_results']['multihop_paths']
        }
        
        # Create a question dict with any available options and answers
        question_dict = {
            "id": "q_" + str(len(self.evaluation_results) + 1),
            "question": question,
            "answer": "Unknown"  # Will be replaced if we have ground truth
        }
        
        # If options are included in the result, add them to the question dict
        if 'options' in result:
            question_dict["options"] = result['options']
        if 'answer' in result:
            question_dict["answer"] = result['answer']
        
        # First evaluate evidence quality
        llm_only_evidence = self.evaluate_evidence_quality(llm_only_answer, structured_kg_data)
        context_strict_evidence = self.evaluate_evidence_quality(context_strict_answer, structured_kg_data)
        llm_informed_evidence = self.evaluate_evidence_quality(llm_informed_answer, structured_kg_data)
        
        # Then evaluate correctness with binary scoring
        llm_only_correctness = self.kg_evaluator.evaluate_correctness_with_similarity(llm_only_answer, question_dict)
        context_strict_correctness = self.kg_evaluator.evaluate_correctness_with_similarity(context_strict_answer, question_dict)
        llm_informed_correctness = self.kg_evaluator.evaluate_correctness_with_similarity(llm_informed_answer, question_dict)
        
        # Calculate combined scores using our custom binary-compatible function
        llm_only_combined = self._calculate_binary_combined_score(llm_only_evidence, llm_only_correctness)
        context_strict_combined = self._calculate_binary_combined_score(context_strict_evidence, context_strict_correctness)
        llm_informed_combined = self._calculate_binary_combined_score(llm_informed_evidence, llm_informed_correctness)
        
        # Prepare hybrid evaluation results
        hybrid_evaluation = {
            "llm_only": llm_only_combined,
            "context_strict": context_strict_combined,
            "llm_informed": llm_informed_combined
        }
        
        # Still run the original evaluation for consistency (if needed)
        try:
            llm_only_evidence_eval = self.kg_evaluator.evaluate_with_evidence_priority(
                llm_only_answer, question_dict, structured_kg_data)
            context_strict_evidence_eval = self.kg_evaluator.evaluate_with_evidence_priority(
                context_strict_answer, question_dict, structured_kg_data)
            llm_informed_evidence_eval = self.kg_evaluator.evaluate_with_evidence_priority(
                llm_informed_answer, question_dict, structured_kg_data)
                
            evidence_based_evaluation = {
                    "llm_only": llm_only_evidence_eval,
                    "context_strict": context_strict_evidence_eval,
                    "llm_informed": llm_informed_evidence_eval
                }
        except Exception as e:
            logger.error(f"Error in evidence-based evaluation: {str(e)}")
            # Create fallback evaluation with just our binary scores
            evidence_based_evaluation = {
                "llm_only": llm_only_combined,
                "context_strict": context_strict_combined,
                "llm_informed": llm_informed_combined
            }
        
        # Rest of the evaluation functions
        citation_kg_data = structured_kg_data.copy()
        for concept in citation_kg_data['concepts']:
            concept['id'] = concept['cui']
        
        citation_strict = self.kg_evaluator.evaluate_citation_quality(context_strict_answer, citation_kg_data)
        citation_informed = self.kg_evaluator.evaluate_citation_quality(llm_informed_answer, citation_kg_data)
        
        kg_coverage = self.kg_evaluator.evaluate_kg_coverage(
            question, result['kg_results']['terms'], result['kg_results']['concepts']
        )
        
        context_contribution = self.kg_evaluator.evaluate_kg_contribution(
            llm_informed_answer, llm_only_answer
        )
        
        # Create final evaluation result
        evaluation_result = {
            "question_id": question_dict["id"],
            "question": question,
            "hybrid_evaluation": hybrid_evaluation,
            "evidence_based_evaluation": evidence_based_evaluation,
            "citation_quality": {
                "context_strict": citation_strict,
                "llm_informed": citation_informed
            },
            "kg_coverage": kg_coverage,
            "context_contribution": context_contribution
        }
        
        # Add to evaluation results
        self.evaluation_results.append({**result, "evaluation": evaluation_result})
        
        return evaluation_result

    def _calculate_binary_combined_score(self, evidence_metrics, correctness_metrics):
        """Calculate combined score with binary correctness (0 or 10)"""
        # Extract scores
        evidence_score = evidence_metrics.get("score", 0)
        correctness_score = correctness_metrics.get("score", 0)
        
        # Binary correctness logic (40% weight)
        # - If correctness is 10, add 4 points (40% of total)
        # - If correctness is 0, add 0 points
        correctness_contribution = 4.0 if correctness_score == 10.0 else 0.0
        
        # Evidence contributes 60% of the score
        evidence_contribution = evidence_score * 0.6
        
        # Combined score
        combined_score = evidence_contribution + correctness_contribution
        
        # Return full score object with all metrics
        return {
            "evidence_score": evidence_score,
            "correctness_score": correctness_score,
            "combined_score": combined_score,
            "evidence_weight": 0.6,
            "correctness_weight": 0.4,
            "evidence_explanation": evidence_metrics.get("explanation", ""),
            "correctness_explanation": correctness_metrics.get("explanation", "")
        }

    def summarize_evaluation_results(self) -> Dict:
        """Summarize evaluation results across all questions"""
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        # Initialize summary
        summary = {
            "questions_processed": len(self.evaluation_results),
            "kg_coverage": {
                "average_coverage": 0,
                "coverage_by_question": {}
            },
            "answer_quality": {
                # Original approach
                "evidence_based": {
                    "llm_only": 0,
                    "context_strict": 0,
                    "llm_informed": 0,
                },
                # New hybrid approach
                "hybrid": {
                    "llm_only": 0,
                    "context_strict": 0, 
                    "llm_informed": 0,
                },
                "quality_by_question": {}
            },
            "context_contribution": {
                "average_value_added": 0,
                "contribution_by_question": {}
            },
            "timing": {
                "average_total_processing": 0,
                "average_evaluation_time": 0,
                "timing_by_question": {}
            }
        }
        
        # Collect metrics across questions
        for r in self.evaluation_results:
            question_id = r.get("evaluation", {}).get("question_id", f"q_{len(summary['answer_quality']['quality_by_question']) + 1}")
            
            # Process answer quality - UPDATED to include hybrid
            try:
                # Get evidence-based evaluation scores
                llm_only_score = float(r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_only", {}).get("combined_score", 0))
                context_strict_score = float(r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("context_strict", {}).get("combined_score", 0))
                llm_informed_score = float(r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_informed", {}).get("combined_score", 0))
                
                # Get hybrid evaluation scores
                llm_only_hybrid = float(r.get("evaluation", {}).get("hybrid_evaluation", {}).get("llm_only", {}).get("combined_score", 0))
                context_strict_hybrid = float(r.get("evaluation", {}).get("hybrid_evaluation", {}).get("context_strict", {}).get("combined_score", 0))
                llm_informed_hybrid = float(r.get("evaluation", {}).get("hybrid_evaluation", {}).get("llm_informed", {}).get("combined_score", 0))
                
                summary["answer_quality"]["quality_by_question"][question_id] = {
                    "evidence_based": {
                        "llm_only": llm_only_score,
                        "context_strict": context_strict_score,
                        "llm_informed": llm_informed_score
                    },
                    "hybrid": {
                        "llm_only": llm_only_hybrid,
                        "context_strict": context_strict_hybrid,
                        "llm_informed": llm_informed_hybrid
                    }
                }
            except Exception as e:
                logger.error(f"Error processing answer quality for question {question_id}: {str(e)}")
                # Handle error case...
        
        # Calculate averages for both evaluation methods
        if summary["questions_processed"] > 0:
            # Calculate averages for evidence-based evaluation
            evidence_llm_only = [q["evidence_based"]["llm_only"] for q in summary["answer_quality"]["quality_by_question"].values()]
            evidence_context_strict = [q["evidence_based"]["context_strict"] for q in summary["answer_quality"]["quality_by_question"].values()]
            evidence_llm_informed = [q["evidence_based"]["llm_informed"] for q in summary["answer_quality"]["quality_by_question"].values()]
            
            summary["answer_quality"]["evidence_based"]["llm_only"] = sum(evidence_llm_only) / len(evidence_llm_only) if evidence_llm_only else 0
            summary["answer_quality"]["evidence_based"]["context_strict"] = sum(evidence_context_strict) / len(evidence_context_strict) if evidence_context_strict else 0
            summary["answer_quality"]["evidence_based"]["llm_informed"] = sum(evidence_llm_informed) / len(evidence_llm_informed) if evidence_llm_informed else 0
            
            # Calculate averages for hybrid evaluation
            hybrid_llm_only = [q["hybrid"]["llm_only"] for q in summary["answer_quality"]["quality_by_question"].values()]
            hybrid_context_strict = [q["hybrid"]["context_strict"] for q in summary["answer_quality"]["quality_by_question"].values()]
            hybrid_llm_informed = [q["hybrid"]["llm_informed"] for q in summary["answer_quality"]["quality_by_question"].values()]
            
            summary["answer_quality"]["hybrid"]["llm_only"] = sum(hybrid_llm_only) / len(hybrid_llm_only) if hybrid_llm_only else 0
            summary["answer_quality"]["hybrid"]["context_strict"] = sum(hybrid_context_strict) / len(hybrid_context_strict) if hybrid_context_strict else 0
            summary["answer_quality"]["hybrid"]["llm_informed"] = sum(hybrid_llm_informed) / len(hybrid_llm_informed) if hybrid_llm_informed else 0
        
        # Process context contribution
        try:
            value_added = float(r.get("evaluation", {}).get("context_contribution", {}).get("value_added_score", 0))
            summary["context_contribution"]["contribution_by_question"][question_id] = value_added
        except Exception as e:
            logger.error(f"Error processing context contribution for question {question_id}: {str(e)}")
            summary["context_contribution"]["contribution_by_question"][question_id] = 0
        
        # Process timing
        try:
            total_time = r.get("processing_time", 0)
            eval_time = r.get("evaluation", {}).get("timing", {}).get("total_evaluation_time", 0)
            
            summary["timing"]["timing_by_question"][question_id] = {
                "total_processing": total_time,
                "evaluation_time": eval_time
            }
        except Exception as e:
            logger.error(f"Error processing timing for question {question_id}: {str(e)}")
            summary["timing"]["timing_by_question"][question_id] = {
                "total_processing": 0,
                "evaluation_time": 0,
                "error": str(e)
            }
        
        # Calculate averages
        if summary["questions_processed"] > 0:
            # Average KG coverage
            summary["kg_coverage"]["average_coverage"] = sum(summary["kg_coverage"]["coverage_by_question"].values()) / summary["questions_processed"]
            
            # Average context contribution
            summary["context_contribution"]["average_value_added"] = sum(summary["context_contribution"]["contribution_by_question"].values()) / summary["questions_processed"]
            
            # Average timing
            summary["timing"]["average_total_processing"] = sum([t["total_processing"] for t in summary["timing"]["timing_by_question"].values()]) / summary["questions_processed"]
            summary["timing"]["average_evaluation_time"] = sum([t["evaluation_time"] for t in summary["timing"]["timing_by_question"].values()]) / summary["questions_processed"]
        
        return summary

    def generate_visualizations(self):
        """Generate visualizations from evaluation results"""
        # Create visualization directory if it doesn't exist
        visualization_dir = self.kg_evaluator.settings.get("visualization_dir", "kg_evaluation")
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Get summary
        try:
            summary = self.summarize_evaluation_results()
        except Exception as e:
            logger.error(f"Error summarizing evaluation results: {str(e)}", exc_info=True)
            print(f"Error generating visualizations: {str(e)}")
            return
        
        # Make sure we have data to visualize
        if summary.get("questions_processed", 0) == 0:
            print("No data to visualize.")
            return
        
        # Generate detailed CSV report
        try:
            df_data = []
            for r in self.evaluation_results:
                question_id = r.get("evaluation", {}).get("question_id", "unknown")
                question_text = r.get("question", "")[:50] + "..." if len(r.get("question", "")) > 50 else r.get("question", "")
                
                # Extract data
                kg_coverage = r.get("evaluation", {}).get("kg_coverage", {}).get("coverage_percentage", 0)
                
                # Use evidence_based_evaluation instead of answer_quality
                llm_only_score = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_only", {}).get("combined_score", 0)
                context_strict_score = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("context_strict", {}).get("combined_score", 0)
                llm_informed_score = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_informed", {}).get("combined_score", 0)
                
                # Extract evidence and correctness scores
                llm_only_evidence = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_only", {}).get("evidence_score", 0)
                context_strict_evidence = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("context_strict", {}).get("evidence_score", 0)
                llm_informed_evidence = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_informed", {}).get("evidence_score", 0)
                
                llm_only_correctness = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_only", {}).get("correctness_score", 0)
                context_strict_correctness = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("context_strict", {}).get("correctness_score", 0)
                llm_informed_correctness = r.get("evaluation", {}).get("evidence_based_evaluation", {}).get("llm_informed", {}).get("correctness_score", 0)
                
                value_added = r.get("evaluation", {}).get("context_contribution", {}).get("value_added_score", 0)
                
                processing_time = r.get("processing_time", 0)
                
                # Add to data
                df_data.append({
                    "QuestionID": question_id,
                    "Question": question_text,
                    "KG_Coverage_Percentage": kg_coverage,
                    "LLM_Only_Score": llm_only_score,
                    "Context_Strict_Score": context_strict_score,
                    "LLM_Informed_Score": llm_informed_score,
                    "LLM_Only_Evidence": llm_only_evidence,
                    "Context_Strict_Evidence": context_strict_evidence,
                    "LLM_Informed_Evidence": llm_informed_evidence,
                    "LLM_Only_Correctness": llm_only_correctness,
                    "Context_Strict_Correctness": context_strict_correctness,
                    "LLM_Informed_Correctness": llm_informed_correctness,
                    "Context_Value_Added": value_added,
                    "Processing_Time_Seconds": processing_time
                })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(visualization_dir, "evaluation_results.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved evaluation results to {csv_path}")
            
            # Now generate actual visualizations
            # 1. Answer Quality Comparison Chart
            plt.figure(figsize=(10, 6))
            answer_types = ['LLM Only', 'Context Strict', 'LLM Informed']
            scores = [
                summary['answer_quality']['evidence_based']['llm_only'],
                summary['answer_quality']['evidence_based']['context_strict'],
                summary['answer_quality']['evidence_based']['llm_informed']
            ]
            
            bars = plt.bar(answer_types, scores, color=['blue', 'green', 'red'])
            plt.ylim(0, 10)
            plt.ylabel('Quality Score (0-10)')
            plt.title('Answer Quality Comparison')
            plt.axhline(y=5, color='gray', linestyle='--')  # Add a reference line at score=5
            
            # Add score labels above bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, 'answer_quality_comparison.png'))
            plt.close()
            
            # 2. KG Coverage vs Answer Quality Scatter Plot
            plt.figure(figsize=(10, 6))
            coverage_values = []
            quality_values = []
            
            for question in df_data:
                coverage_values.append(question['KG_Coverage_Percentage'])
                quality_values.append(question['LLM_Informed_Score'])
            
            plt.scatter(coverage_values, quality_values, alpha=0.7)
            plt.xlabel('Knowledge Graph Coverage (%)')
            plt.ylabel('LLM Informed Answer Quality (0-10)')
            plt.title('Relationship Between KG Coverage and Answer Quality')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Add trend line
            if len(coverage_values) > 1:
                import numpy as np
                z = np.polyfit(coverage_values, quality_values, 1)
                p = np.poly1d(z)
                plt.plot(coverage_values, p(coverage_values), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, 'coverage_vs_quality.png'))
            plt.close()
            
            # 3. Context Value Added Chart
            plt.figure(figsize=(10, 6))
            
            # Sort questions by value added score
            value_added_data = sorted([
                (q["QuestionID"], q["Context_Value_Added"])
                for q in df_data
            ], key=lambda x: x[1], reverse=True)
            
            question_ids = [x[0] for x in value_added_data]
            value_added_scores = [x[1] for x in value_added_data]
            
            bars = plt.bar(question_ids, value_added_scores, color='purple')
            plt.ylim(0, 10)
            plt.ylabel('Value Added Score (0-10)')
            plt.title('Context Contribution to Answer Quality by Question')
            plt.axhline(y=5, color='gray', linestyle='--')  # Add a reference line
            plt.xticks(rotation=45, ha='right')
            
            # Add horizontal line for average
            avg_value = summary['context_contribution']['average_value_added']
            plt.axhline(y=avg_value, color='red', linestyle='-')
            plt.text(0, avg_value + 0.2, f'Avg: {avg_value:.1f}', color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, 'context_value_added.png'))
            plt.close()
            
            # 4. Evidence vs Correctness Chart (2D Comparison)
            plt.figure(figsize=(10, 6))
            
            # Plot individual data points instead of just averages
            for question in df_data:
                plt.scatter(question['LLM_Only_Evidence'], question['LLM_Only_Correctness'], 
                            color='blue', alpha=0.5, marker='o')
                plt.scatter(question['Context_Strict_Evidence'], question['Context_Strict_Correctness'], 
                            color='green', alpha=0.5, marker='s')
                plt.scatter(question['LLM_Informed_Evidence'], question['LLM_Informed_Correctness'], 
                            color='red', alpha=0.5, marker='^')

            # Add larger markers for the averages
            avg_llm_only_evidence = sum(q['LLM_Only_Evidence'] for q in df_data) / len(df_data)
            avg_llm_only_correctness = sum(q['LLM_Only_Correctness'] for q in df_data) / len(df_data)

            avg_context_strict_evidence = sum(q['Context_Strict_Evidence'] for q in df_data) / len(df_data)
            avg_context_strict_correctness = sum(q['Context_Strict_Correctness'] for q in df_data) / len(df_data)

            avg_llm_informed_evidence = sum(q['LLM_Informed_Evidence'] for q in df_data) / len(df_data)
            avg_llm_informed_correctness = sum(q['LLM_Informed_Correctness'] for q in df_data) / len(df_data)

            # Plot averages with larger markers
            plt.scatter(avg_llm_only_evidence, avg_llm_only_correctness, 
                       color='blue', s=150, label='LLM Only', edgecolor='black')
            plt.scatter(avg_context_strict_evidence, avg_context_strict_correctness, 
                       color='green', s=150, label='Context Strict', edgecolor='black')
            plt.scatter(avg_llm_informed_evidence, avg_llm_informed_correctness, 
                       color='red', s=150, label='LLM Informed', edgecolor='black')

            plt.xlabel('Evidence Score (0-10)')
            plt.ylabel('Correctness Score (0-10)')
            plt.title('Evidence vs Correctness Comparison')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()

            # Fix the range to show full scale
            plt.xlim(0, 10)
            plt.ylim(0, 10)

            # Draw quadrant lines
            plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

            # Label quadrants
            plt.text(7.5, 7.5, 'High Evidence\nHigh Correctness', ha='center')
            plt.text(2.5, 7.5, 'Low Evidence\nHigh Correctness', ha='center')
            plt.text(7.5, 2.5, 'High Evidence\nLow Correctness', ha='center')
            plt.text(2.5, 2.5, 'Low Evidence\nLow Correctness', ha='center')

            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, 'evidence_vs_correctness.png'))
            plt.close()
            
            # 5. Processing Time Analysis
            plt.figure(figsize=(10, 6))
            
            # Sort questions by processing time
            time_data = sorted([
                (q["QuestionID"], q["Processing_Time_Seconds"])
                for q in df_data
            ], key=lambda x: x[1], reverse=True)
            
            question_ids = [x[0] for x in time_data]
            processing_times = [x[1] for x in time_data]
            
            bars = plt.bar(question_ids, processing_times, color='orange')
            plt.ylabel('Processing Time (seconds)')
            plt.title('Question Processing Time')
            plt.xticks(rotation=45, ha='right')
            
            # Add horizontal line for average
            avg_time = summary['timing']['average_total_processing']
            plt.axhline(y=avg_time, color='red', linestyle='-')
            plt.text(0, avg_time + 1, f'Avg: {avg_time:.1f}s', color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, 'processing_time.png'))
            plt.close()
            
            # 6. Combined Metrics Summary Chart
            plt.figure(figsize=(12, 6))
            
            metrics = [
                'LLM Only Score', 
                'Context Strict Score', 
                'LLM Informed Score',
                'KG Coverage (%)', 
                'Context Value Added'
            ]
            
            values = [
                summary['answer_quality']['evidence_based']['llm_only'],
                summary['answer_quality']['evidence_based']['context_strict'],
                summary['answer_quality']['evidence_based']['llm_informed'],
                summary['kg_coverage']['average_coverage'],
                summary['context_contribution']['average_value_added']
            ]
            
            # Adjust KG Coverage to 0-10 scale for comparison
            values[3] = values[3] / 10
            
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            
            bars = plt.bar(metrics, values, color=colors)
            plt.ylim(0, 10)
            plt.ylabel('Score (0-10)')
            plt.title('Summary of Key Metrics')
            plt.axhline(y=5, color='gray', linestyle='--')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if i == 3:  # KG Coverage
                    label = f'{height*10:.1f}%'
                else:
                    label = f'{height:.1f}'
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        label, ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_dir, 'metrics_summary.png'))
            plt.close()
            
            logger.info(f"Generated 6 visualization charts in {visualization_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())

    @traceable(run_type="llm", name="LLM Only Answer Generation")
    def generate_llm_only_answer(self, question_with_options: str) -> str:
        """Generate an answer using only the LLM (no context) but with question options included"""
        
        # System message - sets the role
        system_message = """You are a medical expert answering clinical questions based on your knowledge.
        Provide consistent, well-reasoned answers and follow the exact response format specified."""
        
        # User message with question and format instructions
        user_message = f"""# QUESTION
{question_with_options}

Please provide a detailed answer based SOLELY on your medical expertise. 
Format your answer EXACTLY as follows:

**1. REASONING PROCESS:**
- Initial Understanding: [break down the question]
- Medical Knowledge: [relevant medical concepts]
- Chain of Thought: [step-by-step reasoning]

**2. DIFFERENTIAL ANALYSIS:**
- [Why the chosen answer is correct - explain pathophysiology]
- [Why other options are incorrect - systematically rule out alternatives]

**3. CLINICAL CONTEXT:**
- [Pathophysiology explanation]
- [Connection to patient presentation]
- [Clinical management implications]

**4. ANSWER CHOICE:**
- [Single option letter only, e.g. "C"]

**5. CONFIDENCE AND LIMITATIONS:**
- [Confidence level statement]
- [Any critical missing information]
- [Uncertainties in your reasoning]
"""
        
        # Call the LLM with structured messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Use model.invoke with messages list
        response = self.llm.invoke(messages).content
        
        return response
    
    @traceable(run_type="llm", name="Context Strict Answer Generation")  
    def generate_context_strict_answer(self, question_with_options: str, kg_evidence: str, textbook_evidence: str) -> str:
        """Generate an answer using only the provided KG and textbook evidence (no LLM knowledge)"""
        
        # System message with explicit instructions for strict evidence usage
        system_message = """You are a medical expert answering questions STRICTLY based on the provided evidence.
        CRITICAL RULES:
        1. You MUST NOT use ANY external medical knowledge beyond what is explicitly provided.
        2. Every single claim or statement you make MUST be supported by specific evidence.
        3. You MUST cite the exact CUI code (e.g., C0027051) when referencing any medical concept.
        4. You MUST cite the complete relationship path when using knowledge graph relationships.
        5. You MUST quote the exact text with citation numbers when using textbook evidence.
        6. If the evidence is insufficient to make a claim or answer the question, you MUST explicitly state this.
        7. You MUST NOT make assumptions or inferences beyond what is directly supported by the evidence.
        8. You MUST NOT generate or make up any CUI codes - only use codes that appear in the provided evidence.
        
        VIOLATION OF THESE RULES IS NOT ALLOWED UNDER ANY CIRCUMSTANCES."""
        
        # User message with structured evidence and question, prioritizing evidence presentation
        user_message = f"""# KNOWLEDGE GRAPH EVIDENCE
{self._format_for_deterministic_output(kg_evidence)}

# TEXTBOOK EVIDENCE
{self._format_for_deterministic_output(textbook_evidence)}

# QUESTION
{question_with_options}

REQUIRED ANSWER FORMAT:

**1. EVIDENCE INVENTORY:**
- Available Knowledge Graph Concepts: [List ALL provided CUI codes and their terms]
- Available Knowledge Graph Relationships: [List ALL provided relationships]
- Available Textbook Evidence: [List ALL relevant textbook passages with citation numbers]

**2. EVIDENCE ANALYSIS:**
- Directly Relevant Evidence: [List evidence that directly addresses the question]
- Indirectly Relevant Evidence: [List evidence that provides context or supporting information]
- Missing Critical Evidence: [List any critical information that is not provided]

**3. REASONING PROCESS:**
For each step in your reasoning:
a) State the claim/inference
b) Cite the SPECIFIC evidence supporting it:
   - For KG concepts: "As evidenced by concept [CUI: C######]"
   - For relationships: "As shown by relationship [C###### → RELATIONSHIP → C######]"
   - For textbook evidence: "As stated in [REF#]: 'exact quote'"
c) If evidence is insufficient, explicitly state: "Insufficient evidence to support this claim"

**4. DIFFERENTIAL ANALYSIS:**
For each option:
a) List supporting evidence with exact citations
b) List contradicting evidence with exact citations
c) State if there is insufficient evidence to evaluate the option

**5. ANSWER CHOICE:**
- If evidence is sufficient: [Single option letter, e.g., "C"]
- If evidence is insufficient: "Insufficient evidence to determine answer"

**6. EVIDENCE ASSESSMENT:**
- Evidence Completeness: [Assess what evidence is available vs. missing]
- Evidence Quality: [Assess strength and relevance of available evidence]
- Confidence Level: [State confidence based ONLY on evidence completeness]
- Evidence Gaps: [List specific missing evidence that would help answer the question]

REMINDER: EVERY claim must be supported by SPECIFIC evidence citations. DO NOT use any knowledge not provided in the evidence."""
        
        # Create message list with proper system and user messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Call OpenAI with messages format
        response = self.llm.invoke(messages).content
        
        return response
    
    @traceable(run_type="llm", name="LLM Informed Answer Generation")
    def generate_llm_informed_answer(self, question_with_options: str, kg_evidence: str, textbook_evidence: str) -> str:
        """Generate an answer using combined approach (LLM knowledge + provided evidence) with improved formatting"""
        
        # System message with strict rules about evidence usage
        system_message = """You are a medical expert tasked with answering complex clinical questions by combining provided evidence with your medical knowledge.

        CRITICAL RULES FOR EVIDENCE USAGE:
        1. When citing knowledge graph concepts, you MUST ONLY use CUI codes that appear in the provided evidence
        2. NEVER generate, create, or make up CUI codes
        3. When using knowledge graph evidence, you must cite the exact CUI code from the evidence
        4. When using textbook evidence, you must quote the exact text with citation numbers
        5. You may use your internal medical knowledge to:
           - Explain concepts in more detail
           - Make connections between evidence
           - Fill gaps where evidence is incomplete
           - Provide additional context
        6. Clearly distinguish between evidence-based claims and your expert knowledge
           - For evidence: "According to [CUI/Reference]..."
           - For your knowledge: "Based on medical expertise..."

        VIOLATION OF THESE RULES IS NOT ALLOWED."""
        
        # User message with structured evidence and instructions
        user_message = f"""# KNOWLEDGE GRAPH EVIDENCE
{self._format_for_deterministic_output(kg_evidence)}

# TEXTBOOK EVIDENCE
{self._format_for_deterministic_output(textbook_evidence)}

# QUESTION
{question_with_options}

Follow these instructions precisely:

**1. EVIDENCE ANALYSIS:**
First, analyze ALL available evidence:
- List all relevant CUI codes and their concepts
- List all relevant relationships between concepts
- List all relevant textbook passages

**2. KNOWLEDGE INTEGRATION:**
For each part of your answer, clearly indicate the source:
a) When using knowledge graph evidence:
   - Cite the exact CUI code: "Concept [CUI: C######]"
   - Cite the exact relationship: "[C###### → relationship → C######]"
b) When using textbook evidence:
   - Quote the exact text: "As stated in [REF#]: 'exact quote'"
c) When using your medical knowledge:
   - Explicitly state: "Based on medical expertise..."
   - Explain how it complements the evidence

Format your answer EXACTLY as follows:

**1. EVIDENCE INVENTORY:**
- Available Knowledge Graph Concepts: [List ONLY concepts with CUI codes from evidence]
- Available Knowledge Graph Relationships: [List ONLY relationships from evidence]
- Available Textbook Evidence: [List relevant passages with citation numbers]

**2. REASONING PROCESS:**
- Initial Assessment: [Combine evidence and medical knowledge]
- Evidence-Based Findings: [Cite specific evidence]
- Expert Interpretation: [Clearly marked as medical expertise]

**3. CLINICAL CONTEXT:**
- Evidence-Based Pathophysiology: [Cite evidence]
- Expert Clinical Insights: [Mark as medical expertise]
- Integrated Understanding: [Show how evidence and expertise combine]

**4. DIFFERENTIAL ANALYSIS:**
For each option:
- Supporting Evidence: [Cite specific evidence]
- Expert Medical Knowledge: [Mark as medical expertise]
- Combined Assessment: [Integrate both sources]

**5. ANSWER CHOICE:**
- [Single option letter only, e.g. "C"]
- Justify with both evidence and expertise

**6. CONFIDENCE AND LIMITATIONS:**
- Evidence Strength: [Assess available evidence]
- Knowledge Contribution: [How medical expertise filled gaps]
- Combined Confidence: [Overall assessment]

REMINDER: NEVER create or make up CUI codes. Only use codes that appear in the provided evidence."""
        
        # Create message list with proper system and user messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Use model.invoke with messages list
        response = self.llm.invoke(messages).content
        
        logger.info("Generated LLM-informed answer (length: %d chars)", len(response))
        return response

    def retrieve_from_pinecone(self, query_text, top_k=8):
        """Retrieve relevant context from Pinecone vector database"""
        with trace("pinecone_retrieval_and_processing") as parent_run:
            try:
                # Initialize OpenAI embeddings if not already done
                if not hasattr(self, 'openai_embeddings'):
                    from langchain_openai import OpenAIEmbeddings
                    self.openai_embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-large"
                    )
                
                # Initialize Pinecone if not already done
                if not hasattr(self, 'pinecone_index'):
                    from pinecone import Pinecone
                    import os
                    
                    # Get API keys
                    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
                    PINECONE_ENV = os.getenv("PINECONE_ENV")
                    
                    # Initialize Pinecone
                    pc = Pinecone(
                        api_key=PINECONE_API_KEY,
                        environment=PINECONE_ENV
                    )
                    
                    # Connect to existing index
                    index_name = "medical-textbook-embeddings"
                    self.pinecone_index = pc.Index(index_name)
                
                # Get query embedding
                with trace("embedding_generation", parent_run=parent_run) as child_run:
                    query_embedding = self.openai_embeddings.embed_query(query_text)
                
                with trace("pinecone_query", parent_run=parent_run) as child_run:
                    results = self.pinecone_index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True
                    )
                
                # Prepare chunks for combination
                chunks = []
                for match in results.matches:
                    if match.score < 0.5:  # Threshold for relevance
                        continue
                        
                    text = match.metadata.get('text', '').strip()
                    if text.startswith('.'):  # Clean fragmented sentences
                        text = text[1:].strip()
                    
                    chunks.append({
                        'text': text,
                        'source': match.metadata.get('source', ''),
                        'score': match.score
                    })

                if not chunks:
                    logger.warning("No highly relevant information found in the medical database")
                    return ""

                # Combine similar chunks using the function from query_medical_db.py
                combined_results = self._combine_similar_chunks(chunks)
                
                # Format the context for LLM
                context = "\n\n".join([
                    f"[SOURCE {i+1}] From: {', '.join(chunk['sources'])}\nRelevance: {chunk['score']:.2f}\n{chunk['text'][:1000]}"
                    for i, chunk in enumerate(combined_results)
                ])
                
                logger.info(f"Retrieved {len(combined_results)} combined chunks of textbook content")
                
                return context
                
            except Exception as e:
                logger.error(f"Error retrieving from Pinecone: {str(e)}")
                return ""  # Return empty string on error to avoid breaking the pipeline

    def _combine_similar_chunks(self, chunks, similarity_threshold=0.3):
        """Combine similar text chunks into coherent passages"""
        from difflib import SequenceMatcher
        
        def similarity_score(text1, text2):
            """Calculate text similarity using SequenceMatcher"""
            return SequenceMatcher(None, text1, text2).ratio()
        
        combined_chunks = []
        used_chunks = set()

        for i, chunk in enumerate(chunks):
            if i in used_chunks:
                continue

            related_text = [chunk['text']]
            related_chunks = [i]
            used_chunks.add(i)

            # Look for similar chunks
            for j, other_chunk in enumerate(chunks):
                if j not in used_chunks:
                    if (similarity_score(chunk['text'], other_chunk['text']) > similarity_threshold or
                        any(text in other_chunk['text'] for text in related_text) or
                        any(text in chunk['text'] for text in [other_chunk['text']])):
                        
                        related_text.append(other_chunk['text'])
                        related_chunks.append(j)
                        used_chunks.add(j)

            # Combine related chunks and their sources
            combined_text = ' '.join(related_text)
            sources = []
            for idx in related_chunks:
                source = chunks[idx]['source']
                if source and source not in sources:
                    sources.append(source)
                
            avg_score = sum(chunks[idx]['score'] for idx in related_chunks) / len(related_chunks)

            combined_chunks.append({
                'text': combined_text,
                'sources': sources,
                'score': avg_score
            })

        return combined_chunks

    def format_question_with_options(self, question_dict):
        """Format a question with its multiple-choice options if available"""
        if isinstance(question_dict, str):
            # If it's just a string, return it as is
            return question_dict
        
        # Extract question text
        question_text = question_dict.get('question', '')
        
        # Check if there are options
        options = question_dict.get('options', None)
        
        if not options:
            # No options available, return just the question
            return question_text
        
        # Format question with options
        formatted_question = question_text + "\n\n"
        
        # Add options
        if isinstance(options, dict):
            # Handle dictionary format (key-value pairs)
            # Sort to ensure consistent ordering
            sorted_options = sorted(options.items())
            for option_key, option_text in sorted_options:
                formatted_question += f"{option_key}. {option_text}\n"
        elif isinstance(options, list):
            # Handle list format (convert to option letters dynamically)
            for i, option_text in enumerate(options):
                # Use ASCII value starting from 65 ('A') for as many options as we have
                option_letter = chr(65 + i)
                formatted_question += f"{option_letter}. {option_text}\n"
        
        return formatted_question

    def evaluate_evidence_quality(self, answer_text, kg_data):
        """Use LLM to evaluate evidence quality and citations in the answer with detailed breakdown"""
        
        prompt = f"""You are evaluating the quality of evidence and citations in a medical answer. 
        Provide a detailed scoring breakdown for each category (0-10 points each):

        Answer to evaluate:
        {answer_text}

        Evaluate and score these specific aspects:

        1. Citation Quality (0-10):
           - UMLS Concept Citations: Are specific CUIs (e.g., C0027051) mentioned and explained?
           - Relationship Citations: Are connections between concepts explicitly stated?
           - Source References: Are textbook or knowledge graph references properly cited?
           Score each sub-component and provide specific examples found.

        2. Evidence Integration (0-10):
           - Evidence-Reasoning Connection: How well is evidence linked to conclusions?
           - Multi-Source Integration: Are different evidence sources (KG, textbook) combined effectively?
           - Evidence Relevance: Is the cited evidence directly relevant to the question?
           Provide specific examples of strong or weak integration.

        3. Reasoning Structure (0-10):
           - Logical Flow: Is there a clear progression of thought?
           - Alternative Considerations: Are other possibilities discussed with evidence?
           - Clinical Application: Is evidence connected to clinical context?
           Point out specific examples of reasoning patterns.

        Format your response as:
        DETAILED_SCORES:
        - Citation Quality: [score]/10
          * Concept Citations: [examples found]
          * Relationship Citations: [examples found]
          * Source References: [examples found]

        - Evidence Integration: [score]/10
          * Connection Examples: [specific instances]
          * Integration Strengths/Weaknesses: [details]
          * Relevance Analysis: [assessment]

        - Reasoning Structure: [score]/10
          * Logic Flow: [assessment]
          * Alternatives: [examples]
          * Clinical Application: [examples]

        FINAL_EVIDENCE_SCORE: [average of three categories]
        SUMMARY: [Brief explanation of the overall score]
        """

        try:
            response = self.llm_function(prompt)
            
            # Extract detailed scores
            citation_score = re.search(r'Citation Quality:\s*(\d+(?:\.\d+)?)/10', response)
            integration_score = re.search(r'Evidence Integration:\s*(\d+(?:\.\d+)?)/10', response)
            reasoning_score = re.search(r'Reasoning Structure:\s*(\d+(?:\.\d+)?)/10', response)
            final_score = re.search(r'FINAL_EVIDENCE_SCORE:\s*(\d+(?:\.\d+)?)', response)
            summary = re.search(r'SUMMARY:\s*(.+?)(?=\n|$)', response, re.DOTALL)

            # Create detailed breakdown
            detailed_explanation = {
                "citation_quality": {
                    "score": float(citation_score.group(1)) if citation_score else 0.0,
                    "details": self._extract_section_details(response, "Citation Quality")
                },
                "evidence_integration": {
                    "score": float(integration_score.group(1)) if integration_score else 0.0,
                    "details": self._extract_section_details(response, "Evidence Integration")
                },
                "reasoning_structure": {
                    "score": float(reasoning_score.group(1)) if reasoning_score else 0.0,
                    "details": self._extract_section_details(response, "Reasoning Structure")
                },
                "final_score": float(final_score.group(1)) if final_score else 0.0,
                "summary": summary.group(1).strip() if summary else "No summary provided"
            }

            return {
                "score": detailed_explanation["final_score"],
                "explanation": detailed_explanation,
                "summary": detailed_explanation["summary"]
            }
        except Exception as e:
            logger.error(f"Error in detailed evidence evaluation: {str(e)}")
            return {"score": 0, "explanation": f"Error in evaluation: {str(e)}"}

    def _extract_section_details(self, response, section_name):
        """Extract detailed examples and explanations for each scoring section"""
        section_pattern = f"{section_name}.*?\\n(.*?)(?=\\n\\n|$)"
        section_match = re.search(section_pattern, response, re.DOTALL)
        if section_match:
            details = section_match.group(1).strip()
            # Extract bullet points
            bullet_points = re.findall(r'\* ([^\n]+)', details)
            return bullet_points
        return []

    def run_benchmark_evaluation(self):
        """Run evaluation on benchmark questions"""
        logger.info("Starting benchmark evaluation...")
        
        # Create benchmark directory with parents=True to ensure all directories in path are created
        benchmark_dir = os.path.join(self.kg_evaluator.settings.get("visualization_dir", "kg_evaluation"), "benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Also ensure the parent directory exists
        visualization_dir = self.kg_evaluator.settings.get("visualization_dir", "kg_evaluation")
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Get benchmark questions
        benchmark_questions = get_benchmark_usmle_questions()
        logger.info(f"Processing {len(benchmark_questions)} benchmark questions")
        
        # Prepare CSV data with simplified format
        csv_data = []
        
        # Process each question
        for i, question in enumerate(benchmark_questions):
            logger.info(f"Processing benchmark question {i+1}/{len(benchmark_questions)}: {question['id']}")
            
            try:
                # Process the question
                start_time = time.time()
                result = self.process_question(question)
                processing_time = time.time() - start_time
                
                # Extract key information
                question_text = question['question']
                options = question.get('options', [])
                ground_truth = question['answer']
                
                # Get the answers
                llm_only_answer = result['answers']['llm_only']
                context_strict_answer = result['answers']['context_strict']
                llm_informed_answer = result['answers']['llm_informed']
                
                # Extract final answers
                llm_only_extracted = self._extract_final_answer(llm_only_answer)
                context_strict_extracted = self._extract_final_answer(context_strict_answer)
                llm_informed_extracted = self._extract_final_answer(llm_informed_answer)
                
                # Map to full text for better comparison
                llm_only_mapped = self._map_answer_to_options(llm_only_extracted, options)
                context_strict_mapped = self._map_answer_to_options(context_strict_extracted, options)
                llm_informed_mapped = self._map_answer_to_options(llm_informed_extracted, options)
                
                # Prepare structured kg_data for evidence evaluation
                structured_kg_data = {
                    'concepts': result['kg_results']['concepts'],
                    'relationships': result['kg_results']['relationships'],
                    'multihop_paths': result['kg_results']['multihop_paths']
                }
                
                # Create question dict for correctness evaluation
                question_dict = {
                    "id": question['id'],
                    "question": question_text,
                    "answer": ground_truth,
                    "options": options
                }
                
                # Evaluate evidence quality using LLM
                llm_only_evidence = self.evaluate_evidence_quality(llm_only_answer, structured_kg_data)
                context_strict_evidence = self.evaluate_evidence_quality(context_strict_answer, structured_kg_data)
                llm_informed_evidence = self.evaluate_evidence_quality(llm_informed_answer, structured_kg_data)
                
                # Evaluate correctness (binary scoring - either 0 or 4 points, which is 40% of total)
                llm_only_correctness = self.kg_evaluator.evaluate_correctness_with_similarity(llm_only_mapped, question_dict)
                context_strict_correctness = self.kg_evaluator.evaluate_correctness_with_similarity(context_strict_mapped, question_dict)
                llm_informed_correctness = self.kg_evaluator.evaluate_correctness_with_similarity(llm_informed_mapped, question_dict)
                
                # Calculate final scores for each method (evidence + correctness)
                # Evidence score is out of 10 but weighted at 60%, so multiply by 0.6
                # Correctness score is binary (0 or 10) but weighted at 40%, so use 0 or 4
                
                # Format options for CSV
                options_str = "; ".join(options) if isinstance(options, list) else str(options)
                
                # Inside the question processing loop, update these calculations:
                
                # Calculate final scores for each method
                llm_only_score = {
                    "Question_ID": question['id'],
                    "Question": question_text,
                    "Options": options_str,
                    "Ground_Truth": ground_truth,
                    "Method": "LLM Only",
                    "Predicted_Answer": llm_only_extracted,
                    "Full_Answer": llm_only_mapped,
                    "Evidence_Score_Raw": llm_only_evidence.get("score", 0),
                    "Evidence_Score_Weighted": llm_only_evidence.get("score", 0) * 0.6,  # 60% weight
                    "Evidence_Details": {
                        "Citation_Quality": llm_only_evidence.get("explanation", {}).get("citation_quality", {}),
                        "Evidence_Integration": llm_only_evidence.get("explanation", {}).get("evidence_integration", {}),
                        "Reasoning_Structure": llm_only_evidence.get("explanation", {}).get("reasoning_structure", {})
                    },
                    "Evidence_Summary": llm_only_evidence.get("summary", ""),
                    "Correctness_Score_Raw": 10.0 if llm_only_extracted.upper() == ground_truth.upper() else 0.0,  # Direct comparison
                    "Correctness_Score_Weighted": 4.0 if llm_only_extracted.upper() == ground_truth.upper() else 0.0,  # 40% weight
                    "Total_Score": (llm_only_evidence.get("score", 0) * 0.6) + 
                                  (4.0 if llm_only_extracted.upper() == ground_truth.upper() else 0.0)
                }

                # Similar updates for context_strict and llm_informed scores
                context_strict_score = {
                    "Question_ID": question['id'],
                    "Question": question_text,
                    "Options": options_str,
                    "Ground_Truth": ground_truth,
                    "Method": "Context Strict",
                    "Predicted_Answer": context_strict_extracted,
                    "Full_Answer": context_strict_mapped,
                    "Evidence_Score_Raw": context_strict_evidence.get("score", 0),
                    "Evidence_Score_Weighted": context_strict_evidence.get("score", 0) * 0.6,  # 60% weight
                    "Evidence_Details": {
                        "Citation_Quality": context_strict_evidence.get("explanation", {}).get("citation_quality", {}),
                        "Evidence_Integration": context_strict_evidence.get("explanation", {}).get("evidence_integration", {}),
                        "Reasoning_Structure": context_strict_evidence.get("explanation", {}).get("reasoning_structure", {})
                    },
                    "Evidence_Summary": context_strict_evidence.get("summary", ""),
                    "Correctness_Score_Raw": 10.0 if context_strict_extracted.upper() == ground_truth.upper() else 0.0,
                    "Correctness_Score_Weighted": 4.0 if context_strict_extracted.upper() == ground_truth.upper() else 0.0,
                    "Total_Score": (context_strict_evidence.get("score", 0) * 0.6) + 
                                   (4.0 if context_strict_extracted.upper() == ground_truth.upper() else 0.0)
                }

                llm_informed_score = {
                    "Question_ID": question['id'],
                    "Question": question_text,
                    "Options": options_str,
                    "Ground_Truth": ground_truth,
                    "Method": "LLM Informed",
                    "Predicted_Answer": llm_informed_extracted,
                    "Full_Answer": llm_informed_mapped,
                    "Evidence_Score_Raw": llm_informed_evidence.get("score", 0),
                    "Evidence_Score_Weighted": llm_informed_evidence.get("score", 0) * 0.6,  # 60% weight
                    "Evidence_Details": {
                        "Citation_Quality": llm_informed_evidence.get("explanation", {}).get("citation_quality", {}),
                        "Evidence_Integration": llm_informed_evidence.get("explanation", {}).get("evidence_integration", {}),
                        "Reasoning_Structure": llm_informed_evidence.get("explanation", {}).get("reasoning_structure", {})
                    },
                    "Evidence_Summary": llm_informed_evidence.get("summary", ""),
                    "Correctness_Score_Raw": 10.0 if llm_informed_extracted.upper() == ground_truth.upper() else 0.0,
                    "Correctness_Score_Weighted": 4.0 if llm_informed_extracted.upper() == ground_truth.upper() else 0.0,
                    "Total_Score": (llm_informed_evidence.get("score", 0) * 0.6) + 
                                   (4.0 if llm_informed_extracted.upper() == ground_truth.upper() else 0.0)
                }

                # Add scores to csv_data
                csv_data.extend([llm_only_score, context_strict_score, llm_informed_score])
                
                logger.info(f"Completed processing question {i+1}: {question['id']}")
                
                # Add logging to verify extracted answers
                logger.info(f"Question {question['id']} - Ground Truth: {ground_truth}")
                logger.info(f"LLM Only extracted: {llm_only_extracted}")
                logger.info(f"Context Strict extracted: {context_strict_extracted}")
                logger.info(f"LLM Informed extracted: {llm_informed_extracted}")
                
            except Exception as e:
                logger.error(f"Error processing benchmark question {question['id']}: {str(e)}", exc_info=True)
        
        # Create CSV file with all the results
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(benchmark_dir, "simple_benchmark_results.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Benchmark results saved to {csv_path}")
            
            # Also save a version with just the key columns for easier viewing
            simplified_df = df[["Question_ID", "Ground_Truth", "Method", "Predicted_Answer", 
                               "Evidence_Score_Raw", "Correctness_Score_Raw", "Total_Score"]]
            simplified_csv_path = os.path.join(benchmark_dir, "simplified_results.csv")
            simplified_df.to_csv(simplified_csv_path, index=False)
            logger.info(f"Simplified results saved to {simplified_csv_path}")
            
            # When saving to CSV, include a detailed report
            detailed_report_path = os.path.join(benchmark_dir, "detailed_evidence_evaluation.json")
            with open(detailed_report_path, 'w') as f:
                json.dump({
                    "questions": [{
                        "id": score["Question_ID"],
                        "method": score["Method"],
                        "evidence_details": score["Evidence_Details"],
                        "evidence_summary": score["Evidence_Summary"]
                    } for score in csv_data]
                }, f, indent=2)
        
        return {
            "questions_evaluated": len(benchmark_questions),
            "csv_path": csv_path if csv_data else None
        }

    def _extract_final_answer(self, answer_text):
        """Extract the final answer using LLM instead of regex patterns"""
        if not answer_text:
            return ""
        
        prompt = f"""Extract ONLY the letter choice from this medical answer.
        Rules:
        1. Look for explicit answer statements like "ANSWER CHOICE:" or "The best answer is..."
        2. Return ONLY the letter, nothing else
        3. If multiple letters are mentioned, identify the final chosen answer
        4. If no clear letter choice is found, return "NONE"

        Answer text:
        {answer_text}

        Return format:
        ANSWER: [single letter only]
        """
        
        try:
            response = self.llm_function(prompt)
            # Try multiple patterns to extract the answer
            patterns = [
                r'ANSWER:\s*([A-Z])',  # Standard format
                r'Answer:\s*([A-Z])',  # Alternative capitalization
                r'[^a-zA-Z]([A-Z])[^a-zA-Z]',  # Standalone letter
                r'\*\*([A-Z])\*\*',  # Markdown bold
                r'The (?:correct )?answer is\s*([A-Z])',  # Natural language
                r'Option\s*([A-Z])',  # Option format
                r'Choice\s*([A-Z])'  # Choice format
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
            
            # If no match found in the LLM response, try the original answer text
            for pattern in patterns:
                match = re.search(pattern, answer_text, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
                    
            return "NONE"
        except Exception as e:
            logger.error(f"Error in LLM answer extraction: {str(e)}")
            return "NONE"

    def _map_answer_to_options(self, answer, options):
        """Map between letter answers and option text"""
        if not answer or not options:
            return answer
            
        # Create mapping
        option_mapping = {}
        reverse_mapping = {}
        
        if isinstance(options, list):
            for i, option_text in enumerate(options):
                # Map any letter based on position
                letter = chr(65 + i)  # Starting from 'A'
                option_mapping[letter] = option_text
                option_mapping[f"{letter}."] = option_text
                reverse_mapping[option_text.lower()] = letter
        elif isinstance(options, dict):
            # Sort to ensure consistent ordering
            sorted_options = sorted(options.items())
            for key, value in sorted_options:
                option_mapping[key] = value
                option_mapping[f"{key}."] = value
                reverse_mapping[value.lower()] = key
        
        # Check if answer is a letter
        if re.match(r'^[A-Z]\.?$', answer.upper()):
            letter = answer.replace(".", "").upper()
            if letter in option_mapping:
                return option_mapping[letter]
        
        # Check if answer is text that matches an option
        lower_answer = answer.lower()
        for option_text in option_mapping.values():
            if lower_answer == option_text.lower():
                return option_text
                
        # Unable to map, return original
        return answer

    def _format_for_deterministic_output(self, evidence_text):
        """Format evidence while preserving the standardized structure"""
        if not evidence_text:
            return ""
            
        # If the text already has our standard formatting, return it as is
        if "=== MEDICAL CONCEPTS ===" in evidence_text or \
           "=== RELATIONSHIPS ===" in evidence_text or \
           "=== COMPLEX KNOWLEDGE PATHS ===" in evidence_text:
            return evidence_text
        
        # For textbook evidence, format with clear citation markers
        if "[SOURCE" in evidence_text or "[REF" in evidence_text:
            # Clean up the text while preserving citation structure
            lines = evidence_text.split('\n')
            formatted_lines = []
            current_source = None
            
            for line in lines:
                if line.strip().startswith('[SOURCE') or line.strip().startswith('[REF'):
                    current_source = line.strip()
                    formatted_lines.append(current_source)
                elif current_source and line.strip():
                    formatted_lines.append(f"  {line.strip()}")
            
            return '\n'.join(formatted_lines)
        
        # For any other type of evidence, just clean up whitespace
        return ' '.join(evidence_text.split())

def main():
    processor = USMLEProcessor()
    
    print("\n🏥 Welcome to the USMLE Question Processor!")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'evaluate' to see evaluation results")
    print("Type 'visualize' to generate evaluation reports and graphs")
    print("Type 'benchmark' to run evaluation on standard USMLE questions")
    
    while True:
        print("\n📝 Enter your USMLE question (or command):")
        user_input = input().strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        elif user_input.lower() == 'evaluate':
            if processor.evaluation_results:
                summary = processor.summarize_evaluation_results()
                print("\n📊 Evaluation Summary:")
                print(f"Questions processed: {summary['questions_processed']}")
                print(f"Average KG coverage: {summary['kg_coverage']['average_coverage']:.2f}%")
                print(f"Answer Quality Comparison:")
                print(f"  - LLM Only: {summary['answer_quality']['evidence_based']['llm_only']:.2f}/10")
                print(f"  - Context Strict: {summary['answer_quality']['evidence_based']['context_strict']:.2f}/10")
                print(f"  - LLM Informed: {summary['answer_quality']['evidence_based']['llm_informed']:.2f}/10")
                print(f"Value Added by Context: {summary['context_contribution']['average_value_added']:.2f}/10")
                print(f"Average processing time: {summary['timing']['average_total_processing']:.2f} seconds")
            else:
                print("No evaluation results available. Process some questions first.")
            continue
        elif user_input.lower() == 'visualize':
            if processor.evaluation_results:
                print("\n📊 Generating visualizations and reports...")
                processor.generate_visualizations()
                print(f"Visualizations and reports saved to {processor.kg_evaluator.settings['visualization_dir']}")
            else:
                print("No evaluation results available. Process some questions first.")
            continue
        elif user_input.lower() == 'benchmark':
            print("\n🔬 Running benchmark evaluation on standard USMLE questions...")
            benchmark_results = processor.run_benchmark_evaluation()
            print(f"\n✅ Benchmark evaluation complete! Processed {benchmark_results['questions_evaluated']} questions.")
            print(f"Reports saved to {processor.kg_evaluator.settings['visualization_dir']}/benchmark")
            continue
        
        print("\n🔍 Processing question...")
        try:
            from langsmith import trace
            
            with trace("question_processing_full_pipeline") as span:
                span.metadata["question"] = user_input  # Add metadata
                start_time = time.time()
                result = processor.process_question(user_input)
                processing_time = time.time() - start_time
                span.metadata["processing_time"] = processing_time
                
            # Print all three answer types
            print("\n=================================================================")
            print("📝 LLM-ONLY ANSWER (No Knowledge Graph):")
            print("=================================================================")
            print(result['answers']['llm_only'])
            
            print("\n=================================================================")
            print("📚 CONTEXT-STRICT ANSWER (Only Knowledge Graph and Textbook Data):")
            print("=================================================================")
            print(result['answers']['context_strict'])
            
            print("\n=================================================================")
            print("🔍 LLM-INFORMED ANSWER (Combined Knowledge):")
            print("=================================================================")
            print(result['answers']['llm_informed'])
            
            # Add a summary of what knowledge was used
            kg_concepts = len(result['kg_results']['concepts'])
            kg_relations = len(result['kg_results']['relationships'])
            kg_terms = len(result['kg_results']['terms']) if 'terms' in result['kg_results'] else 0
            
            print("\n📊 Knowledge Used:")
            print(f"- {kg_concepts} medical concepts from knowledge graph")
            print(f"- {kg_relations} relationships between concepts")
            print(f"- {kg_terms} medical terms extracted from question")
            
            # Evaluate the answer
            print("\n🔍 Evaluating answer quality...")
            evaluation = processor.evaluate_evidence_based_answer(result)
            
            # Show brief evaluation summary
            print("\n📊 Evaluation Summary:")
            try:
                # Access the combined_score inside each dictionary with better error handling
                def safe_get_score(eval_dict, key, default=0.0):
                    try:
                        if key in eval_dict:
                            score = float(eval_dict[key])
                            if score == 0.0:  # If score is exactly zero, likely a parsing error
                                return default
                            return score
                        return default
                    except (TypeError, ValueError):
                        return default
                
                llm_only_score = safe_get_score(
                    evaluation['evidence_based_evaluation']['llm_only'], 'combined_score', 5.0)
                context_strict_score = safe_get_score(
                    evaluation['evidence_based_evaluation']['context_strict'], 'combined_score', 5.0)
                llm_informed_score = safe_get_score(
                    evaluation['evidence_based_evaluation']['llm_informed'], 'combined_score', 5.0)
                
                # Extract evidence and correctness scores with fallbacks
                llm_informed_evidence = safe_get_score(
                    evaluation['evidence_based_evaluation']['llm_informed'], 'evidence_score', 5.0)
                llm_informed_correctness = safe_get_score(
                    evaluation['evidence_based_evaluation']['llm_informed'], 'correctness_score', 5.0)
                
                # Extract value_added_score from context_contribution
                value_added = safe_get_score(evaluation['context_contribution'], 'value_added_score', 5.0)
                
                # Print the metrics with added debug info
                print(f"- LLM Only Quality: {llm_only_score:.1f}/10")
                print(f"- Context Strict Quality: {context_strict_score:.1f}/10")
                print(f"- LLM Informed Quality: {llm_informed_score:.1f}/10")
                print(f"- KG Coverage: {float(evaluation['kg_coverage'].get('coverage_percentage', 0)):.1f}%")
                print(f"- Value Added by Context: {value_added:.1f}/10")
                print(f"- Evidence Quality (LLM Informed): {llm_informed_evidence:.1f}/10")
                print(f"- Correctness (LLM Informed): {llm_informed_correctness:.1f}/10")
                
                # Add debug information
                print("\nScore Details:")
                print(f"- Correctness Weight: 60%")
                print(f"- Evidence Weight: 40%") 
                if llm_only_score == 5.0 and context_strict_score == 5.0 and llm_informed_score == 5.0:
                    print("\n⚠️ NOTE: Default fallback scores were used due to evaluation parsing issues.")
                    print("This usually happens when the LLM output format wasn't as expected.")
            except Exception as e:
                logger.error(f"Error printing evaluation details: {str(e)}", exc_info=True)
                print("- Some evaluation metrics could not be displayed")
                print(f"- Error: {str(e)}")
            print("\nType 'evaluate' for full summary or 'visualize' for reports and graphs")
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            print(f"Error processing question: {e}")
    
    print("\nThank you for using the USMLE Question Processor!")

if __name__ == "__main__":
    main()