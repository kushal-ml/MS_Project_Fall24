import sys
from pathlib import Path
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from openai import OpenAI
import logging
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import concurrent.futures
import re


# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add these constants at the top of the file
EVALUATION_WEIGHTS = {
    'evidence_quality': 0.6,  # 60% weight
    'correctness': 0.4        # 40% weight
}

SUB_METRICS = {
    'evidence_quality': {
        'citation_density': 0.3,       # Number of concepts/relationships cited per 100 words
        'source_diversity': 0.2,       # Balance between KG and textbook evidence usage
        'conflict_resolution': 0.2,    # Handling of evidence conflicts
        'traceability': 0.3            # Explicit citations for all claims
    },
    'correctness': {
        'factual_accuracy': 0.7,       # Medical accuracy
        'error_detection': 0.3         # Identification of incorrect evidence
    }
}

class KnowledgeGraphEvaluator:
    def __init__(self, graph: Neo4jGraph, llm_function):
        """Initialize the KG evaluator with graph and LLM access"""
        self.graph = graph
        self.llm = llm_function
        self.results = {}
      
        # Initialize embedding model - MATCH THE DATABASE MODEL
        try:
            self.embedding_model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
        
        logger.info("Knowledge Graph Evaluator initialized")
        
        # Set default evaluation settings
        self.settings = {
            "test_size": 20,  # Increased test size for more comprehensive evaluation
            "top_k_concepts": 30,  # Increased from 10 to 100
            "top_k_relationships": 50,  # Increased from 50 to 200
            "visualization_dir": "evaluation_results",
            "vector_search_enabled": True,
            "multihop_enabled": True,
            "multihop_max_depth":3, 
            "vector_search_threshold": 0.5  # Reduced from 0.3 to 0.25 for higher recall
        }
    def load_test_questions(self, path: str = None, n: int = None) -> List[Dict]:
        """Load test questions from file or use built-in examples"""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                questions = json.load(f)
                return questions[:n] if n else questions
        
        # Default USMLE-style questions if no file provided
        sample_questions = [
            {
                "id": "q1",
                "question": "A 45-year-old man presents with crushing chest pain for 2 hours. ECG shows ST elevation in leads II, III, and aVF. Which artery is most likely occluded?",
                "answer": "Right coronary artery",
                "options": ["Right coronary artery", "Left anterior descending artery", "Left circumflex artery", "Left main coronary artery", "Posterior descending artery"],
                "difficulty": "medium"
            },
            {
                "id": "q2", 
                "question": "A 67-year-old woman with congenital bicuspid aortic valve is admitted with fever and systolic murmur. Blood cultures show viridans streptococci. What antibiotic synergistic to penicillin would help shorten treatment duration?",
                "answer": "Gentamicin",
                "options": ["Gentamicin", "Ceftriaxone", "Vancomycin", "Ciprofloxacin", "Linezolid"],
                "difficulty": "hard"
            },
            {
                "id": "q3",
            "question": "A 32-year-old woman with type 1 diabetes mellitus has had progressive renal failure during the past 2 years. She has not yet started dialysis. Examination shows no abnormalities. Her hemoglobin concentration is 9 g/dL, hematocrit is 28%, and mean corpuscular volume is 94 μm3. A blood smear shows normochromic, normocytic cells. Which of the following is the most likely cause?",
            "answer": "Erythropoietin deficiency",
            "options": ["Acute blood loss", "Chronic lymphocytic leukemia", "Erythrocyte enzyme deficiency", "Erythropoietin deficiency", "Immunohemolysis", "Microangiopathic hemolysis", "Polycythemia vera", "Sickle cell disease", "Sideroblastic anemia", "β-Thalassemia trait"],
                "difficulty": "medium"
            },
            {
                "id": "q4",
                "question": "A 5-year-old girl is brought to the emergency department by her mother because of multiple episodes of nausea and vomiting that last about 2 hours. During this period, she has had 6–8 episodes of bilious vomiting and abdominal pain. The vomiting was preceded by fatigue. The girl feels well between these episodes. She has missed several days of school and has been hospitalized 2 times during the past 6 months for dehydration due to similar episodes of vomiting and nausea. The patient has lived with her mother since her parents divorced 8 months ago. Her immunizations are up-to-date. She is at the 60th percentile for height and 30th percentile for weight. She appears emaciated. Her temperature is 36.8°C (98.8°F), pulse is 99/min, and blood pressure is 82/52 mm Hg. Examination shows dry mucous membranes. The lungs are clear to auscultation. Abdominal examination shows a soft abdomen with mild diffuse tenderness with no guarding or rebound. The remainder of the physical examination shows no abnormalities. Which of the following is the most likely diagnosis?",
                "answer": "Cyclic vomiting syndrome",
                "options": ["Cyclic vomiting syndrome", "Gastroenteritis", "Hypertrophic pyloric stenosis", "Gastroesophageal reflux disease"],
                "difficulty": "medium"
            }
            # More questions as needed...
        ]
        
        limit = min(n, len(sample_questions)) if n else len(sample_questions)
        return sample_questions[:limit]
    
    def extract_key_terms(self, question: str) -> List[Dict]:
        """Enhanced extraction of key medical terms from question using LLM"""
        prompt = f"""
        Extract medical terms from this question and categorize them.
        Be VERY comprehensive - include ALL possible medical concepts, including:
        
        - Disease: Medical conditions/disorders (e.g., diabetes, endocarditis)
        - Drug: Medications/substances (e.g., penicillin, gentamicin)
        - Symptom: Clinical findings (e.g., fever, murmur)
        - Anatomy: Body structures (e.g., heart, aortic valve)
        - Procedure: Medical procedures (e.g., dialysis, ECG)
        - ClinicalScenario: Patient context (e.g., "67-year-old woman", "pregnancy")
        - LaboratoryValue: Lab results (e.g., "hemoglobin 9 g/dL", "blood cultures")
        
        IMPORTANT: For demographic information, always include age, gender, and any descriptors as ClinicalScenario.
        For lab values, include both the test name and result.
        Break compound terms into individual medical concepts when appropriate.
        
        Question: {question}
        
        Return JSON list with:
        [
            {{
                "term": "term",
                "type": "node_type",
                "priority": 1-3 (1=highest priority, 3=lowest)
            }}
        ]
        Return ONLY the JSON array.
        """
        
        response = self.llm(prompt)
        try:
            # Improved JSON extraction with regex
            json_match = re.search(r'(\[.*?\])(?:\s*$|\n)', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            elif "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            else:
                cleaned_response = response.strip().replace('json', '').replace('```', '')
                json_text = cleaned_response
            
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error parsing terms: {str(e)}")
            return []
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings using SentenceTransformer"""
        if not texts:
            return np.array([])
        
        try:
            # First try with the embedding model
            if self.embedding_model:
                logger.info(f"Computing embeddings for {len(texts)} texts")
                embeddings = self.embedding_model.encode(texts)
                return embeddings
            else:
                # Fallback to OpenAI embeddings if available
                if hasattr(self, 'embeddings') and self.embeddings:
                    logger.info("Using OpenAI embeddings as fallback")
                    embeddings = []
                    for text in texts:
                        emb = self.embeddings.embed_query(text)
                        embeddings.append(emb)
                    return np.array(embeddings)
                else:
                    logger.error("No embedding model available")
                    return np.array([])
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            return np.array([])

    
    def vector_similarity_search(self, query_term: str, concept_terms: List[str], concept_data: List[Dict]) -> List[Dict]:
        if not concept_data or not query_term:
            return []

        try:
            # Get the query term embedding
            query_embedding = self.compute_embeddings([query_term])[0]
            
            # More detailed debug info
            logger.info(f"Found {len(concept_data)} candidate concepts for vector search")
            concepts_with_embeddings = [c for c in concept_data if 'embedding' in c]
            logger.info(f"Candidates with embeddings: {len(concepts_with_embeddings)}")
            
            if len(concepts_with_embeddings) > 0:
                # Log the first embedding to verify format
                sample_embedding = concepts_with_embeddings[0].get('embedding')
                logger.info(f"Sample embedding type: {type(sample_embedding)}")
                if isinstance(sample_embedding, str):
                    logger.info("Embeddings appear to be stored as strings, converting to numpy arrays")
                    
            # Extract precomputed embeddings directly from the graph
            # Handle different embedding formats (list, string, etc.)
            concept_embeddings = []
            for c in concept_data:
                if 'embedding' in c:
                    emb = c['embedding']
                    # Convert string to list if needed
                    if isinstance(emb, str):
                        try:
                            # Try parsing as JSON string
                            import json
                            emb = json.loads(emb)
                        except:
                            # Try parsing as literal string representation of list
                            import ast
                            try:
                                emb = ast.literal_eval(emb)
                            except:
                                logger.warning(f"Could not parse embedding: {emb[:30]}...")
                                continue
                
                    # Add to embeddings list
                    concept_embeddings.append(np.array(emb))
            
            concept_embeddings = np.array(concept_embeddings)
            
            if len(concept_embeddings) == 0:
                logger.warning("No embeddings found in candidates")
                return []
            
            # Calculate similarity
            similarities = cosine_similarity([query_embedding], concept_embeddings)[0]
            
            # Debug similarity values
            logger.info(f"Similarity scores - min: {np.min(similarities):.4f}, max: {np.max(similarities):.4f}")
            
            # Fix normalization - avoid division by zero
            if np.max(similarities) > np.min(similarities):
                similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))
            else:
                # If all similarities are equal, use the raw values
                logger.warning(f"All similarity scores are identical: {np.min(similarities):.4f}")
                # If they're all high (>0.5), keep them; otherwise set to 0
                if np.min(similarities) < 0.5:
                    similarities = np.zeros_like(similarities)
            
            # Set a reasonable threshold if needed
            threshold = self.settings["vector_search_threshold"]
            
            # Create results with scores above threshold
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    # Make sure we have a valid index
                    if i < len(concept_data):
                        result = concept_data[i].copy() 
                        result["similarity_score"] = float(similarity)
                        result["vector_match"] = True
                        results.append(result)
                        logger.info(f"Found vector match: {result.get('term', 'Unknown')} with score {similarity:.4f}")

            # Sort by similarity
            results = sorted(results, key=lambda x: x.get("similarity_score", 0), reverse=True)
            logger.info(f"Returning {len(results)} vector matches")
            return results
        
        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            logger.exception("Full traceback:")
            return []

    
    def get_concepts(self, terms: List[Dict]) -> List[Dict]:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Initialize concepts list
        concepts = []
        
        # Skip if no terms provided
        if not terms:
            logger.warning("No terms provided for concept retrieval")
            return concepts
        
        # Get a list of all term strings (regardless of priority)
        # Remove parenthetical content for better matching
        term_strings = []
        term_dict = {}  # To track which cleaned term came from which original term
        
        logger.info("DEBUG: Original terms:")
        for i, term in enumerate(terms):
            # Clean term for better matching
            term_text = term.get("term", "").strip()
            logger.info(f"DEBUG: Term {i}: {term_text} (type: {term.get('type', 'Unknown')})")
            
            # Remove parenthetical content
            cleaned_term = re.sub(r'\s*\([^)]*\)', '', term_text).strip()
            if cleaned_term:
                term_strings.append(cleaned_term)
                term_dict[cleaned_term] = term_text
                logger.info(f"DEBUG: Cleaned term {i}: '{cleaned_term}' from original '{term_text}'")
        
        logger.info(f"Looking up concepts for {len(term_strings)} terms")
        logger.info(f"Processed terms for query: {term_strings}")
        
        # IMPROVEMENT 1: Direct case-insensitive matching for all terms
        if term_strings:
            logger.info("DEBUG: ===== EXECUTING DIRECT MATCHING QUERY =====")
            logger.info(f"DEBUG: Query parameters: {term_strings}")
            
            # Execute individual queries for each term for debugging
            for i, term in enumerate(term_strings):
                debug_query = """
                MATCH (c:Concept)
                WHERE toLower(c.term) = toLower($term) OR toLower(c.term) CONTAINS toLower($term)
                OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d)
                RETURN c.cui as cui, c.term as term, d.text as definition,
                    labels(c) as labels
                LIMIT 5
                """
                debug_results = self.graph.query(debug_query, {"term": term})
                logger.info(f"DEBUG: Direct query results for term '{term}': {len(debug_results)} matches")
                for j, result in enumerate(debug_results[:5]):  # Log first 5 results
                    logger.info(f"DEBUG:   Result {j}: CUI={result.get('cui', 'None')}, Term='{result.get('term', 'None')}'")
            
            
            direct_query = """
                MATCH (c:Concept)
            WHERE toLower(c.term) IN $terms OR 
                ANY(t IN $terms WHERE toLower(c.term) CONTAINS toLower(t))
            OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d)
            RETURN c.cui as cui, c.term as term, d.text as definition,
                labels(c) as labels
            """
            
            raw_results = self.graph.query(direct_query, {"terms": term_strings})
            logger.info(f"DEBUG: Combined direct query returned {len(raw_results)} total results")
            
            # Process and add direct match concepts
            for i, result in enumerate(raw_results):
                logger.info(f"DEBUG: Processing direct match result {i}: {result.get('cui', 'None')} - '{result.get('term', 'None')}'")
                
                # Check which term(s) this matched with
                matched_with = []
                for term_str in term_strings:
                    if term_str.lower() in result.get("term", "").lower() or result.get("term", "").lower() in term_str.lower():
                        matched_with.append(term_str)
                
                logger.info(f"DEBUG:   Matched with terms: {matched_with}")
                
                concept_data = {
                    "cui": result.get("cui", ""),
                    "term": result.get("term", ""),
                    "definition": result.get("definition", "No definition available"),
                    "labels": result.get("labels", ["Concept"]),
                    "match_type": "direct_match",
                    "vector_match": False,
                    "confidence": 1.0,
                    "original_term": term_dict.get(matched_with[0] if matched_with else "", "Unknown")
                }
                concepts.append(concept_data)
        
        # IMPROVEMENT 2: Special handling for medication terms
        medication_terms = [term["term"].lower() for term in terms if term.get("type") == "Drug"]
        if medication_terms:
            logger.info("DEBUG: ===== EXECUTING MEDICATION MATCHING QUERY =====")
            logger.info(f"DEBUG: Medication terms: {medication_terms}")
            
            # Debug each medication term individually
            for i, med_term in enumerate(medication_terms):
                debug_med_query = """
                MATCH (c:Concept) 
                WHERE toLower(c.term) = toLower($term) OR toLower(c.term) CONTAINS toLower($term)
                OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d)
                RETURN c.cui as cui, c.term as term, d.text as definition,
                    labels(c) as labels
                LIMIT 5
                """
                debug_med_results = self.graph.query(debug_med_query, {"term": med_term})
                logger.info(f"DEBUG: Medication query results for '{med_term}': {len(debug_med_results)} matches")
                for j, result in enumerate(debug_med_results[:5]):
                    logger.info(f"DEBUG:   Med Result {j}: CUI={result.get('cui', 'None')}, Term='{result.get('term', 'None')}'")
            
           
            direct_med_query = """
            MATCH (c:Concept) 
            WHERE toLower(c.term) IN $terms OR
                ANY(t IN $terms WHERE toLower(c.term) CONTAINS toLower(t))
            OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d)
            RETURN c.cui as cui, c.term as term, d.text as definition,
                labels(c) as labels
            """
            
            results = self.graph.query(direct_med_query, params={"terms": medication_terms})
            logger.info(f"DEBUG: Combined medication query returned {len(results)} total results")
            
            # Add medication concepts with high confidence
            med_concepts_added = 0
            for i, result in enumerate(results):
                logger.info(f"DEBUG: Processing medication result {i}: {result.get('cui', 'None')} - '{result.get('term', 'None')}'")
                
                # Find which term this matched with
                matching_term = None
                for term in terms:
                    if term.get("type") == "Drug" and (term.get("term", "").lower() in result.get("term", "").lower() or 
                                            result.get("term", "").lower() in term.get("term", "").lower()):
                        matching_term = term.get("term", "")
                        break
                
                logger.info(f"DEBUG:   Matched with medication term: {matching_term}")
                
                if matching_term:
                    # Avoid duplicates
                    is_duplicate = any(c.get("cui") == result.get("cui", "") for c in concepts)
                    if is_duplicate:
                        logger.info(f"DEBUG:   Skipping duplicate medication concept: {result.get('cui', 'None')}")
                    else:
                        concepts.append({
                            "cui": result.get("cui", ""),
                            "term": result.get("term", ""),
                            "definition": result.get("definition", "No definition available"),
                            "labels": result.get("labels", []),
                            "match_type": "med_match",
                            "vector_match": False,
                            "confidence": 1.0,
                            "original_term": matching_term
                        })
                        med_concepts_added += 1
            
            logger.info(f"DEBUG: Added {med_concepts_added} medication concepts")
        
        # IMPROVEMENT 3: Vector search with lower threshold
        # Lower the threshold from default 0.3 to 0.2
        vector_search_threshold = self.settings.get("vector_search_threshold", 0.3)
        improved_threshold = max(0.2, vector_search_threshold * 0.8)  # Lower by 20% but not below 0.2
        logger.info(f"DEBUG: Vector search threshold: {improved_threshold} (original: {vector_search_threshold})")
        
        # Get concept terms from the graph for vector matching
        if self.settings.get("vector_searchEnabled", True):
            logger.info("DEBUG: ===== EXECUTING VECTOR SIMILARITY SEARCH =====")
            
            # Get all concept terms from the graph (limited for performance)
            concept_query = """
            MATCH (c:Concept)
            OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d)
            RETURN c.cui as cui, c.term as term, d.text as definition, c.embedding as embedding
            LIMIT 2000
            """
            concept_data = self.graph.query(concept_query)
            concept_terms = [c["term"] for c in concept_data if c.get("term")]
            logger.info(f"DEBUG: Retrieved {len(concept_data)} concepts with embeddings for vector matching")
            
            # Perform vector similarity search for each term
            vector_concepts_added = 0
            for i, term in enumerate(terms):
                term_text = term.get("term", "").strip()
                if not term_text or len(term_text) < 3:
                    logger.info(f"DEBUG: Skipping term '{term_text}' (too short or empty)")
                    continue
                
                logger.info(f"DEBUG: Executing vector search for term {i}: '{term_text}'")
                similar_concepts = self.vector_similarity_search(
                    term_text, concept_terms, concept_data
                )
                
                logger.info(f"DEBUG: Vector search returned {len(similar_concepts)} similar concepts for '{term_text}'")
                # Log top 5 matches
                for j, concept in enumerate(similar_concepts[:5]):
                    logger.info(f"DEBUG:   Vector match {j}: CUI={concept.get('cui', 'None')}, Term='{concept.get('term', 'None')}', Score={concept.get('score', 0)}")
                
                # Filter by improved threshold and add to concepts list
                concepts_added_for_term = 0
                for concept in similar_concepts:
                    # Skip if score is below improved threshold
                    if concept.get("score", 0) < improved_threshold:
                        continue
                    
                    # Skip if already in concepts list
                    is_duplicate = any(c.get("cui") == concept.get("cui") for c in concepts)
                    if is_duplicate:
                        continue
                    
                    # Add to concepts list
                    concepts.append({
                        "cui": concept.get("cui"),
                        "term": concept.get("term"),
                        "definition": concept.get("definition"),
                        "vector_match": True,
                        "match_type": "vector_match",
                        "confidence": concept.get("score", 0),
                        "source_term": term_text,
                        "original_term": term_text
                    })
                    concepts_added_for_term += 1
                    vector_concepts_added += 1
                
                logger.info(f"DEBUG: Added {concepts_added_for_term} vector-matched concepts for term '{term_text}'")
            
            logger.info(f"DEBUG: Added {vector_concepts_added} total vector-matched concepts")
        
        # Deduplicate concepts based on CUI
        pre_dedup_count = len(concepts)
        unique_concepts = {}
        for concept in concepts:
            cui = concept.get("cui")
            if cui:
                # Keep the concept with the highest confidence
                if cui not in unique_concepts or concept.get("confidence", 0) > unique_concepts[cui].get("confidence", 0):
                    unique_concepts[cui] = concept
        
        logger.info(f"DEBUG: Deduplication: {pre_dedup_count} concepts → {len(unique_concepts)} unique concepts")
        
        # Determine missing terms
        input_terms = {t['term'] for t in terms}
        matched_terms = set(c.get('original_term', '') for c in unique_concepts.values() if c.get('original_term'))
        missing_terms = [t['term'] for t in terms if t['term'] not in matched_terms and len(t['term']) > 2]
        
        if missing_terms:
            logger.warning(f"Could not find concepts for terms: {', '.join(missing_terms)}")
        
        # Log success statistics
        logger.info(f"Found {len(unique_concepts)} unique concepts from {len(terms)} terms")
        
        # Return the list of unique concepts
        return list(unique_concepts.values())
            
    def get_relationships(self, concepts: List[Dict]) -> List[Dict]:
        """Get all relationships between found concepts without filtering by relationship type"""
        try:
            if not concepts:
                return []
                
            cuis = [c.get('cui') for c in concepts if c and c.get('cui')]
            if not cuis:
                return []
                
            # Modified query to retrieve all relationship types
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
            LIMIT $limit
            """
            results = self.graph.query(query, {
                'cuis': cuis, 
                'limit': self.settings["top_k_relationships"]
            })
            valid_results = [r for r in results if r and 'source_name' in r and 'relationship_type' in r and 'target_name' in r]
            return valid_results
        except Exception as e:
            logger.error(f"Relationship search error: {str(e)}")
            return []

    def find_multihop_paths(self, concepts: List[Dict], max_depth: int = None) -> List[Dict]:
        """Find multi-hop paths with all relationship types"""
        if not self.settings["multihop_enabled"]:
            return []
        
        if max_depth is None:
            max_depth = self.settings["multihop_max_depth"]
        
        try:
            if not concepts or len(concepts) < 2:
                return []
            
            # Get CUIs of concepts
            cuis = [c.get('cui') for c in concepts if c and c.get('cui')]
            if len(cuis) < 2:
                return []
            
            # Use top 8 concepts instead of just 5
            top_cuis = cuis[:min(8, len(cuis))]
            
            paths = []
            for i, source_cui in enumerate(top_cuis):
                for target_cui in top_cuis[i+1:]:
                    if source_cui == target_cui:
                        continue
                    
                    # Modified query to use all relationship types without filtering
                    query = f"""
                    MATCH path = (start:Concept {{cui: $source_cui}})-[*1..{max_depth}]-(end:Concept {{cui: $target_cui}})
                    WITH path, start, end, relationships(path) as rels, nodes(path) as nodes
                    ORDER BY length(path) 
                    LIMIT 3
                    RETURN 
                        start.cui as source_cui,
                        start.term as source_term,
                        end.cui as target_cui,
                        end.term as target_term,
                        [node in nodes | {{
                            cui: node.cui, 
                            term: node.term,
                            semantic_type: node.semantic_type,
                            labels: labels(node)
                        }}] as path_nodes,
                        [rel in rels | {{type: type(rel)}}] as path_rels,
                        length(path) as path_length
                    """
                    
                    results = self.graph.query(query, {'source_cui': source_cui, 'target_cui': target_cui})
                    valid_results = [r for r in results if r and 'source_cui' in r and 'target_cui' in r]
                    paths.extend(valid_results)
            
            # Format the best 30 paths
            formatted_paths = []
            for path in sorted(paths, key=lambda x: x.get('path_length', 99))[:30]:
                try:
                    nodes = path.get('path_nodes', [])
                    rels = path.get('path_rels', [])
                    
                    path_description = []
                    path_description.append(f"({nodes[0]['term']})")
                    
                    for i in range(len(rels)):
                        rel_type = rels[i]['type']
                        next_node = nodes[i+1] if i+1 < len(nodes) else {}
                        
                        # Get appropriate label based on node type
                        node_label = "Unknown"
                        
                        # First try term for Concept nodes
                        if 'term' in next_node and next_node['term']:
                            node_label = next_node['term']
                        # Then try semantic_type for SemanticType nodes
                        elif 'semantic_type' in next_node and next_node['semantic_type']:
                            node_label = next_node['semantic_type']
                        # Fallback to labels as a hint for debugging
                        elif 'labels' in next_node and next_node['labels']:
                            node_label = f"Unknown:{','.join(next_node['labels'])}"
                        
                        path_description.append(f"-[{rel_type}]->({node_label})")
                    
                    formatted_paths.append({
                        'source_term': path['source_term'],
                        'target_term': path['target_term'],
                        'path_length': path['path_length'],
                        'path_description': ''.join(path_description),
                        'path_nodes': path['path_nodes'],
                        'path_rels': path['path_rels']
                    })
                except Exception as e:
                    logger.error(f"Error formatting path: {str(e)}")
            
            return formatted_paths
            
        except Exception as e:
            logger.error(f"Error finding multihop paths: {str(e)}")
            return []
    def format_kg_data(self, concepts: List[Dict], relationships: List[Dict], multihop_paths: List[Dict] = None) -> Dict:
        """Format KG data with IDs for citation tracking"""
        formatted_data = {
            "concepts": [],
            "relationships": [],
            "multihop_paths": []
        }
        
        # Format concepts with IDs
        for i, concept in enumerate(concepts):
            concept_id = f"C{i+1}"
            formatted_data["concepts"].append({
                "id": concept_id,
                "cui": concept.get("cui", ""),
                "term": concept.get("term", ""),
                "definition": concept.get("definition", "No definition available"),
                "types": concept.get("types", []),
                "vector_match": concept.get("vector_match", False),
                "similarity_score": concept.get("similarity_score", None)
            })
        
        # Format relationships with IDs
        for i, rel in enumerate(relationships):
            rel_id = f"R{i+1}"
            formatted_data["relationships"].append({
                "id": rel_id,
                "type": rel.get("relationship_type", ""),
                "source_term": rel.get("source_name", ""),
                "source_cui": rel.get("source_cui", ""),
                "target_term": rel.get("target_name", ""),
                "target_cui": rel.get("target_cui", "")
            })
        
        # Format multihop paths with IDs
        if multihop_paths:
            for i, path in enumerate(multihop_paths):
                path_id = f"P{i+1}"
                formatted_data["multihop_paths"].append({
                    "id": path_id,
                    "source_term": path.get("source_term", ""),
                    "target_term": path.get("target_term", ""),
                    "path_length": path.get("path_length", 0),
                    "path_description": path.get("path_description", "")
            })
            
        return formatted_data
    
    def format_kg_data_for_prompt(self, concepts: List[Dict], relationships: List[Dict], multihop_paths: List[Dict] = None) -> str:
        """Format KG data as concise bullet points for better citation"""
        output_parts = []
        
        # Format concepts section with clearer IDs
        output_parts.append("===== CONCEPTS =====")
        for i, concept in enumerate(concepts):
            concept_id = f"C{i+1}"
            concept_type = next(iter(concept.get("types", ["Unknown"])), "Unknown")
            
            # Bold IDs for better visibility
            concept_line = f"• [ID: {concept_id}] {concept.get('term', 'Unknown')} ({concept_type})"
            
            # Add clinical significance for better context  
            if concept.get("cui"):
                concept_line += f" [CUI: {concept.get('cui')}]"
            if concept.get("vector_match"):
                concept_line += " [Vector Match]"
            output_parts.append(concept_line)
            
            # Format definitions to be more citable
            definition_value = concept.get("definition")
            definition = "" if definition_value is None else str(definition_value).strip()
            
            if definition and len(definition) > 150:
                definition = definition[:147] + "..."
            
            if definition:
                output_parts.append(f"  - Definition ({concept_id}): {definition}")
        
        # Improve relationship formatting
        if relationships:
            output_parts.append("\n===== RELATIONSHIPS =====")
            # Group relationships by type for better organization
            rel_by_type = {}
            for i, rel in enumerate(relationships):
                rel_type = rel.get("relationship_type", "Unknown")
                if rel_type not in rel_by_type:
                    rel_by_type[rel_type] = []
                rel_by_type[rel_type].append((i, rel))
            
            # Format with clearer relationship IDs
            for rel_type, rels in rel_by_type.items():
                output_parts.append(f"• {rel_type} ({len(rels)}):")
                for idx, (i, rel) in enumerate(rels[:5]):
                    rel_id = f"R{i+1}"
                    # Display relationship clearly as a medical fact
                    output_parts.append(f"  - [ID: {rel_id}] {rel.get('source_name', '')} → {rel.get('target_name', '')}")
                
                if len(rels) > 5:
                    output_parts.append(f"  - ... and {len(rels)-5} more {rel_type} relationships")
        
        # Format multihop paths section
        if multihop_paths:
            output_parts.append("\n===== MULTIHOP PATHS =====")
            for i, path in enumerate(multihop_paths):
                path_id = f"P{i+1}"
                output_parts.append(f"• {path_id}: {path.get('source_term', '')} → {path.get('target_term', '')} (length: {path.get('path_length', 0)})")
                # Simplified path description to save tokens
                if 'path_description' in path:
                    # Truncate very long path descriptions
                    path_desc = path['path_description']
                    if len(path_desc) > 100:
                        path_desc = path_desc[:97] + "..."
                    output_parts.append(f"  - {path_desc}")
        
        return "\n".join(output_parts)

    def verify_citations(self, answer: str, kg_data: Dict) -> str:
        """Verify and improve citations in the answer"""
        
        # Create maps of IDs to their content
        concept_map = {c["id"]: c for c in kg_data.get("concepts", [])}
        relationship_map = {r["id"]: r for r in kg_data.get("relationships", [])}
        path_map = {p["id"]: p for p in kg_data.get("multihop_paths", [])}
        
        # Use regex to find citation patterns
        citation_pattern = r'\b([CRP]\d+)\b'
        citations = re.findall(citation_pattern, answer)
        
        # Verify each citation
        citation_issues = []
        for citation in citations:
            if citation.startswith('C') and citation not in concept_map:
                citation_issues.append(f"Citation {citation} not found in concepts")
            elif citation.startswith('R') and citation not in relationship_map:
                citation_issues.append(f"Citation {citation} not found in relationships")
            elif citation.startswith('P') and citation not in path_map:
                citation_issues.append(f"Citation {citation} not found in paths")
        
        # If issues found, ask LLM to fix them
        if citation_issues:
            fix_prompt = f"""
            Your answer has the following citation issues:
            {chr(10).join(citation_issues)}
            
            Please review and correct your answer, using only valid citations from the knowledge graph.
            Original answer:
            {answer}
            """
            return self.llm(fix_prompt)
        
        return answer

    def generate_answer(self, question: str, concepts: List[Dict], relationships: List[Dict], multihop_paths: List[Dict] = None) -> str:
        """Generate answer using LLM with KG data in a concise format with improved citation prompting"""
        
        # Keep track of the original formatted data for citation evaluation
        kg_data = self.format_kg_data(concepts, relationships, multihop_paths)
        
        # Use the concise bullet point format for the LLM prompt
        concise_kg_data = self.format_kg_data_for_prompt(concepts, relationships, multihop_paths)
        
        # Create an improved prompt that emphasizes citation requirements
        prompt = f"""
        Answer the following medical question using ONLY the provided knowledge graph data.
        For EVERY medical claim in your answer, you MUST cite the specific concept or relationship IDs.
        
        Question: {question}
        
        Knowledge Graph Data:
        {concise_kg_data}
        
        IMPORTANT CITATION RULES:
        1. EVERY medical claim requires at least one citation
        2. Use format [C1] for concepts, [R3] for relationships, and [P2] for paths
        3. Place citations immediately after the claim they support
        4. If you need to state a limitation (missing information), cite this as [LIMITATION]
        5. Citations must be specific and directly support the claim being made
        
        EXAMPLE OF GOOD CITATION:
        "Penicillin [C4] is effective against viridans streptococci [C7] as indicated by the MAY_TREAT relationship [R2]. However, information about synergistic antibiotics is not available [LIMITATION]."
        
        Organize your response as follows:
        
        ANSWER: Direct answer to the question with citations for each claim
        REASONING: Step-by-step reasoning using the graph data (with citations)
        CONNECTIONS: Any insights from multihop paths (with citations)
        LIMITATIONS: What information was missing from the graph (with [LIMITATION] citations)
        """
        
        # Store the KG data for evaluation purposes
        self._last_kg_data = kg_data
        
        # Get the initial response
        initial_answer = self.llm(prompt)
        
        # Add citation verification
        if not self._check_citations(initial_answer):
            # If insufficient citations, prompt for improvement
            improved_prompt = f"""
            Your previous answer did not include enough citations.
            
            EVERY medical claim must be cited with a specific ID from the knowledge graph.
            If information is missing from the graph, cite it as [LIMITATION].
            
            Previous answer:
            {initial_answer}
            
            Knowledge Graph Data:
            {concise_kg_data}
            
            Please revise with proper citations for EVERY claim.
            """
            return self.llm(improved_prompt)
        
        return initial_answer

    def _check_citations(self, answer: str) -> bool:
        """Check if the answer has sufficient citations"""
        # Count sentences vs citations
        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        citations = re.findall(r'\[(C\d+|R\d+|P\d+|LIMITATION)\]', answer)
        
        # Simple heuristic: at least 70% of sentences should have citations
        citation_ratio = len(citations) / max(len(sentences), 1)
        return citation_ratio >= 0.7
    
    def generate_llm_only_answer(self, question: str) -> str:
        """Generate answer using only the LLM, without KG data"""
        prompt = f"""
        Answer the following medical question based on your knowledge:
        
        Question: {question}
        
        Provide a detailed response with your reasoning.
        """
        
        return self.llm(prompt)
    
    def generate_kg_informed_answer(self, question: str, formatted_kg_data: Dict) -> str:
        """Generate answer with optimized KG-informed approach"""
        # Extract the original data structure
        concepts = formatted_kg_data.get("concepts", [])
        relationships = formatted_kg_data.get("relationships", [])
        multihop_paths = formatted_kg_data.get("multihop_paths", [])
        
        # Focus on top 25 concepts and 50 relationships
        key_concepts = concepts[:25]
        key_relationships = relationships[:50]
        key_paths = multihop_paths[:10]
        
        # Format with minimal representation to reduce tokens
        from collections import defaultdict
        
        # Group relationships by type for more targeted prompting
        rel_by_type = defaultdict(list)
        for rel in key_relationships:
            rel_type = rel.get("relationship_type", "")
            rel_by_type[rel_type].append(rel)
        
        # Create focused KG info sections for the prompt
        kg_sections = []
        
        # Add important concepts
        kg_sections.append("CONCEPTS:")
        for i, concept in enumerate(key_concepts[:15]):
            concept_type = next(iter(concept.get("types", ["Unknown"])), "Unknown")
            kg_sections.append(f"• {concept.get('term', 'Unknown')} ({concept_type})")
        
        # Add key relationship types with examples
        for rel_type in ["MAY_TREAT", "MAY_BE_TREATED_BY", "CAUSE_OF", "DISEASE_HAS_FINDING", "CONTRAINDICATED_WITH_DISEASE", "MAY_PREVENT"]:
            if rel_type in rel_by_type and rel_by_type[rel_type]:
                kg_sections.append(f"\n{rel_type}:")
                for rel in rel_by_type[rel_type][:3]:
                    kg_sections.append(f"• {rel.get('source_term', '')} → {rel.get('target_term', '')}")
        
        # Add a few multihop paths if available
        if key_paths:
            kg_sections.append("\nKEY CONNECTIONS:")
            for path in key_paths[:3]:
                kg_sections.append(f"• {path.get('path_description', '')}")
        
        concise_kg_data = "\n".join(kg_sections)
        
        # Create improved prompt for quality
        prompt = f"""
        

        
            You are a medical expert answering clinical questions. Your response must mimic a clinician's thinking process, integrating evidence from the knowledge graph with your medical expertise to provide a well-structured, evidence-based answer.
                    QUESTION: {question}
                    
                    KNOWLEDGE GRAPH DATA:
                    {concise_kg_data}
            Instructions:

            1. Assess the Context:

                -First, evaluate whether the knowledge graph data provided is sufficient to answer the question.

                -If critical information is missing, use your internal medical knowledge to fill the gaps. Clearly state when you are doing so.

            2. Structure Your Response:

                -Direct Answer: Start with a concise, direct answer to the question.

                -Explanation: Provide a detailed explanation of your reasoning, integrating evidence from the knowledge graph and your medical expertise.

                -Clinical Context: Place the answer in a clinical context, explaining how it applies to the patient's presentation.

                -Differential Diagnosis: If applicable, briefly discuss alternative diagnoses and explain why they are less likely.

                -Management Considerations: Suggest next steps for diagnosis or management, if relevant.

            3. Integrate Knowledge Graph Data:

                -Prioritize information from the knowledge graph, especially relationship data.

                -When referencing specific facts or relationships from the knowledge graph, use language like:

                "Medical literature indicates [specific fact from KG]."

                "According to clinical data [specific fact from KG]."

                "Research shows that [specific relationship from KG]."

            4. Distinguish Between Sources:

                -Clearly differentiate between information derived from the knowledge graph and your internal medical knowledge.

                -For any critical claim, identify the specific knowledge source (e.g., knowledge graph, medical guidelines, or your expertise).

            5. Mimic Clinicians Thinking Process:

                -Begin with the most likely diagnosis or answer based on the evidence.

                -Rule out alternative explanations systematically.

                -Consider the patient history, physical examination, and laboratory findings in your reasoning.

                -Acknowledge limitations or missing information that could affect the diagnosis or management.
        """
        
        return self.llm(prompt)
    
    def evaluate_citation_quality(self, answer: str, kg_data: Dict) -> Dict:
        import logging
        import re
        import json

        # Set up logging (assuming logger is configured elsewhere; adjust as needed)
        logger = logging.getLogger(__name__)

        # Ensure all concepts have an 'id'
        for idx, concept in enumerate(kg_data.get("concepts", []), start=1):
            if "id" not in concept:
                concept["id"] = f"C{idx}"
                logger.warning(f"Assigned default id {concept['id']} to concept: {concept}")

        # Ensure all relationships have an 'id'
        for idx, rel in enumerate(kg_data.get("relationships", []), start=1):
            if "id" not in rel:
                rel["id"] = f"R{idx}"
                logger.warning(f"Assigned default id {rel['id']} to relationship: {rel}")

        # Ensure all multihop paths have an 'id'
        for idx, path in enumerate(kg_data.get("multihop_paths", []), start=1):
            if "id" not in path:
                path["id"] = f"P{idx}"
                logger.warning(f"Assigned default id {path['id']} to multihop path: {path}")

        # Use the last stored KG data if not provided
        if not kg_data and hasattr(self, '_last_kg_data'):
            kg_data = self._last_kg_data

        # Extract all citations from the answer
        citation_pattern = r'\b([CRP]\d+)\b'
        citations_used = re.findall(citation_pattern, answer)

        # Create maps for quick lookup
        concept_map = {c["id"]: c for c in kg_data.get("concepts", [])}
        relationship_map = {r["id"]: r for r in kg_data.get("relationships", [])}
        path_map = {p["id"]: p for p in kg_data.get("multihop_paths", [])}

        # Count valid vs. invalid citations
        valid_citations = [c for c in citations_used if
                        (c.startswith('C') and c in concept_map) or
                        (c.startswith('R') and c in relationship_map) or
                        (c.startswith('P') and c in path_map)]

        invalid_citations = [c for c in citations_used if c not in valid_citations]

        # Calculate citation statistics
        citation_stats = {
            "total_citations_used": len(citations_used),
            "valid_citations": len(valid_citations),
            "invalid_citations": len(invalid_citations),
            "validity_ratio": len(valid_citations) / max(len(citations_used), 1) * 100,
            "citation_density": len(citations_used) / (len(answer.split()) / 100)  # citations per 100 words
        }

        # Prepare data for LLM evaluation
        concise_kg_data = self.format_kg_data_for_prompt(
            kg_data.get("concepts", []),
            kg_data.get("relationships", []),
            kg_data.get("multihop_paths", [])
        )

        # Enhanced LLM prompt for citation quality assessment
        prompt = f"""
        Evaluate the citation quality in this medical answer:
        
        Answer with citations: 
        {answer}

        Citation Statistics:
        - Total citations: {citation_stats['total_citations_used']}
        - Valid citations: {citation_stats['valid_citations']}
        - Invalid citations: {citation_stats['invalid_citations']}
        - Citation density: {citation_stats['citation_density']:.2f} per 100 words
        
        Knowledge Graph Data that was available:
        {concise_kg_data}
        
        Please analyze:
        1. Are claims properly supported by relevant citations?
        2. Are the citations used in a way that shows understanding of the medical connections?
        3. Does each major claim have a citation?
        4. Are unsupported claims clearly marked as limitations?
        
        Return a JSON with:
        {{
          "total_claims": number,
          "cited_claims": number,
        "accurate_citations": number (citations that actually support the claim),
          "citation_accuracy": percentage,
          "unsupported_claims": number,
        "citation_quality_score": 0-10,
        "reasoning": "explanation of the score"
        }}
        """
        
        response = self.llm(prompt)
        try:
            # Fix JSON extraction with proper regex pattern
            json_match = re.search(r'({.*?})(?:\s*$|\n)', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            elif "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            else:
                json_text = response.strip()
                
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error parsing citation evaluation: {str(e)}")
            return {
                "total_claims": 0,
                "cited_claims": 0,
                "accurate_citations": 0,
                "citation_accuracy": 0,
                "unsupported_claims": 0,
                "citation_quality_score": 0,
                "reasoning": f"Error parsing: {str(e)}"
            }
    def evaluate_kg_coverage(self, question: str, extracted_terms: List[Dict], concepts: List[Dict]) -> Dict:
        """Evaluate how well the KG covers the terms needed for the question"""
        # Calculate term coverage
        found_terms = set()
        for concept in concepts:
            # Try multiple properties that might contain the original term
            original_term = concept.get('original_term', '') or concept.get('source_term', '')
            
            # If those aren't found, we could try matching by comparing the concept term with extracted terms
            if not original_term:
                concept_term = concept.get('term', '').lower()
                for term in extracted_terms:
                    if term['term'].lower() in concept_term or concept_term in term['term'].lower():
                        original_term = term['term']
                        break
                        
            if original_term:
                found_terms.add(original_term)
        
        all_terms = set(term['term'] for term in extracted_terms)
        missing_terms = all_terms - found_terms
        
        coverage_pct = (len(found_terms) / len(all_terms)) * 100 if all_terms else 0
        
        # Categorize found and missing terms by type
        found_by_type = {}
        missing_by_type = {}
        
        for term in extracted_terms:
            term_type = term['type']
            term_text = term['term']
            
            if term_text in found_terms:
                if term_type not in found_by_type:
                    found_by_type[term_type] = []
                found_by_type[term_type].append(term_text)
            else:
                if term_type not in missing_by_type:
                    missing_by_type[term_type] = []
                missing_by_type[term_type].append(term_text)
        
        return {
            "question": question,
            "total_terms": len(all_terms),
            "found_terms": len(found_terms),
            "coverage_percentage": coverage_pct,
            "missing_terms": list(missing_terms),
            "found_by_type": found_by_type,
            "missing_by_type": missing_by_type
        }
    
    def evaluate_llm_answer(self, llm_answer: str, question: Dict) -> Dict:
        """Evaluate the quality of the LLM-only answer"""
        correct_answer = question.get("answer", "")
        options = question.get("options", [])
        
        prompt = f"""
        Evaluate this answer to a medical question:
        
        Question: {question.get('question', '')}
        
        Answer provided: 
        {llm_answer}
        
        Correct answer: {correct_answer}
        
        Please analyze:
        1. Is the answer correct? (Yes/No/Partial)
        2. How comprehensive is the explanation? (0-10 scale)
        3. How confident does the answer appear? (0-10 scale)
        4. Are there any factual errors?
        
        Return a JSON with:
        {{
          "correctness": "Yes/No/Partial",
          "accuracy_score": 0-10,
          "comprehensiveness": 0-10,
          "confidence": 0-10,
          "factual_errors": [list of errors, if any],
          "overall_quality": 0-10
        }}
        """
        
        response = self.llm(prompt)
        try:
            # Fix JSON extraction with proper regex pattern
            json_match = re.search(r'({.*?})(?:\s*$|\n)', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            elif "```json" in response:
                # Try to extract from code blocks
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                # Fallback to any code block
                json_text = response.split("```")[1].split("```")[0].strip()
            else:
                # Just use the response as is
                json_text = response.strip()
                
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error parsing LLM evaluation: {str(e)}")
            return {
                "correctness": "Unknown",
                "accuracy_score": 0,
                "comprehensiveness": 0,
                "confidence": 0,
                "factual_errors": [],
                "overall_quality": 0,
                "error": str(e)
        }
    
    def run_ablation_study(self, question: Dict) -> Dict:
        """Run ablation study comparing different KG usage strategies with context awareness"""
        question_text = question["question"]
        options = question.get("options", [])
        logger.info(f"Running ablation study for question: {question_text[:50]}...")
        
        result = {
            "question_id": question.get("id", "unknown"),
            "question": question_text,
            "options": options,
            "correct_answer": question.get("answer", ""),
            "timing": {},
            "kg_coverage": {},
            "answers": {},
            "evaluations": {},
            "vector_stats": {}
        }
        
        # 1. Extract terms
        start_time = time.time()
        extracted_terms = self.extract_key_terms(question_text)
        result["timing"]["term_extraction"] = time.time() - start_time
        result["extracted_terms"] = extracted_terms
        
        # 1.5 NEW: Analyze question type for context-aware processing
        start_time = time.time()
        question_analysis = self.analyze_question_type(question_text, extracted_terms)
        result["question_analysis"] = question_analysis
        result["timing"]["question_analysis"] = time.time() - start_time
        
        # 2. Get concepts
        start_time = time.time()
        concepts = self.get_concepts(extracted_terms)
        result["timing"]["concept_retrieval"] = time.time() - start_time
        result["concepts_count"] = len(concepts)
        result["concepts"] = concepts
        
        # 3. Get relationships
        start_time = time.time()
        relationships = self.get_relationships(concepts)
        result["timing"]["relationship_retrieval"] = time.time() - start_time
        result["relationships_count"] = len(relationships)
        
        # Vector match statistics
        vector_matches = sum(1 for c in concepts if c.get("vector_match", False))
        result["vector_stats"]["vector_matches_found"] = vector_matches
        result["vector_stats"]["vector_match_terms"] = [c.get("term", "Unknown") for c in concepts if c.get("vector_match", False)]
        
        # 4. Get multihop paths with context awareness
        multihop_paths = []
        if self.settings["multihop_enabled"]:
            start_time = time.time()
            multihop_paths = self.find_multihop_paths(concepts)
            result["timing"]["multihop_path_finding"] = time.time() - start_time
            result["multihop_paths_count"] = len(multihop_paths)
        
        # 5. Evaluate KG coverage
        result["kg_coverage"] = self.evaluate_kg_coverage(question_text, extracted_terms, concepts)
        
        # 6. Format KG data
        kg_data = self.format_kg_data(concepts, relationships, multihop_paths)
        
        # 7. Generate answers with different approaches
        
        # LLM-only approach
        start_time = time.time()
        llm_only_answer = self.generate_llm_only_answer(question_text)
        result["timing"]["llm_only_generation"] = time.time() - start_time
        result["answers"]["llm_only"] = llm_only_answer
        
        # KG-strict approach
        start_time = time.time()
        kg_strict_answer = self.generate_answer(question_text, concepts, relationships, multihop_paths)
        result["timing"]["kg_strict_generation"] = time.time() - start_time
        result["answers"]["kg_strict"] = kg_strict_answer
        
        # KG-informed approach
        start_time = time.time()
        kg_informed_answer = self.generate_kg_informed_answer(question_text, kg_data)
        result["timing"]["kg_informed_generation"] = time.time() - start_time
        result["answers"]["kg_informed"] = kg_informed_answer
        
        # 8. Evaluate citation quality
        start_time = time.time()
        citation_evaluation = self.evaluate_citation_quality(kg_strict_answer, kg_data)
        result["timing"]["citation_evaluation"] = time.time() - start_time
        result["evaluations"]["citation_quality"] = citation_evaluation
        
        # 9. Evaluate KG contribution
        start_time = time.time()
        contribution_vs_llm = self.evaluate_kg_contribution(kg_strict_answer, llm_only_answer)
        result["timing"]["contribution_evaluation_strict"] = time.time() - start_time
        result["evaluations"]["kg_contribution_strict"] = contribution_vs_llm
        
        start_time = time.time()
        contribution_informed = self.evaluate_kg_contribution(kg_informed_answer, llm_only_answer)
        result["timing"]["contribution_evaluation_informed"] = time.time() - start_time
        result["evaluations"]["kg_contribution_informed"] = contribution_informed
        
        # 10. Calculate total processing time
        result["timing"]["total_processing"] = sum(result["timing"].values())
        
        # Evaluate LLM-only answer
        llm_evaluation = self.evaluate_llm_answer(llm_only_answer, question)
        result["timing"]["llm_evaluation"] = time.time() - start_time
        result["evaluations"]["llm_quality"] = llm_evaluation
        
        return result
    
    def run_evaluation(self, questions: List[Dict] = None, n: int = None, parallel: bool = True) -> Dict:
        """Run evaluation with parallel processing for faster results"""
        if not questions:
            questions = self.load_test_questions(n=n or self.settings["test_size"])
        
        results = []
        
        if parallel and len(questions) > 1:
            # Process questions in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(questions))) as executor:
                future_to_question = {executor.submit(self.run_ablation_study, q): q for q in questions}
                for future in tqdm(concurrent.futures.as_completed(future_to_question), 
                                  total=len(questions), 
                                  desc="Evaluating questions"):
                    try:
                        question_result = future.result()
                        results.append(question_result)
                        
                        # Save intermediate results
                        self.results = results
                        if not os.path.exists(self.settings["visualization_dir"]):
                            os.makedirs(self.settings["visualization_dir"])
                        with open(f"{self.settings['visualization_dir']}/results.json", 'w') as f:
                            json.dump(results, f, indent=2)
                    except Exception as e:
                        logger.error(f"Error processing question: {str(e)}")
        else:
                    # Process questions sequentially
            for question in tqdm(questions, desc="Evaluating questions"):
                question_result = self.run_ablation_study(question)
                results.append(question_result)
                
                # Save intermediate results
                self.results = results
                if not os.path.exists(self.settings["visualization_dir"]):
                    os.makedirs(self.settings["visualization_dir"])
                with open(f"{self.settings['visualization_dir']}/results.json", 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
        
    def summarize_results(self) -> Dict:
        """Summarize evaluation results"""
        if not self.results:
            logger.warning("No results to summarize. Run evaluation first.")
            return {}
        
        summary = {
            "questions_processed": len(self.results),
            "kg_coverage": {
                "average_coverage": np.mean([r["kg_coverage"]["coverage_percentage"] for r in self.results]),
                "coverage_by_type": {}
            },
            "kg_contribution": {
                "average_strict_value": np.mean([r["evaluations"]["kg_contribution_strict"]["value_added_score"] 
                                              for r in self.results if "kg_contribution_strict" in r["evaluations"]]),
                "average_informed_value": np.mean([r["evaluations"]["kg_contribution_informed"]["value_added_score"] 
                                               for r in self.results if "kg_contribution_informed" in r["evaluations"]])
            },
            "citation_quality": {
                "average_score": np.mean([r["evaluations"]["citation_quality"]["citation_quality_score"] 
                                        for r in self.results if "citation_quality" in r["evaluations"]]),
                "average_accuracy": np.mean([r["evaluations"]["citation_quality"]["citation_accuracy"] 
                                          for r in self.results if "citation_quality" in r["evaluations"]])
            },
            "multihop_stats": {
                "average_paths_found": np.mean([r.get("multihop_paths_count", 0) for r in self.results]),
                "questions_with_paths": sum(1 for r in self.results if r.get("multihop_paths_count", 0) > 0)
            },
            "vector_search_stats": {
                "vector_matches_found": sum(r.get("vector_stats", {}).get("vector_matches_found", 0) for r in self.results),
                "questions_with_vector_matches": sum(1 for r in self.results if r.get("vector_stats", {}).get("vector_matches_found", 0) > 0)
            },
            "timing": {
                "average_total": np.mean([r["timing"]["total_processing"] for r in self.results]),
                "average_by_stage": {}
            },
            "llm_performance": {
                "average_accuracy": np.mean([r["evaluations"].get("llm_quality", {}).get("accuracy_score", 0) 
                                            for r in self.results if "llm_quality" in r["evaluations"]]),
                "average_comprehensiveness": np.mean([r["evaluations"].get("llm_quality", {}).get("comprehensiveness", 0) 
                                                for r in self.results if "llm_quality" in r["evaluations"]]),
                "average_overall": np.mean([r["evaluations"].get("llm_quality", {}).get("overall_quality", 0) 
                                          for r in self.results if "llm_quality" in r["evaluations"]]),
            },
        }
        
        # Calculate average timing by stage
        timing_keys = self.results[0]["timing"].keys()
        for key in timing_keys:
            summary["timing"]["average_by_stage"][key] = np.mean([r["timing"].get(key, 0) for r in self.results])
        
        # Calculate coverage by term type
        term_types = set()
        for r in self.results:
            for term_type in r["kg_coverage"].get("found_by_type", {}).keys():
                term_types.add(term_type)
            for term_type in r["kg_coverage"].get("missing_by_type", {}).keys():
                term_types.add(term_type)
        
        for term_type in term_types:
            coverage_values = []
            for r in self.results:
                found = len(r["kg_coverage"].get("found_by_type", {}).get(term_type, []))
                missing = len(r["kg_coverage"].get("missing_by_type", {}).get(term_type, []))
                total = found + missing
                coverage = (found / total) * 100 if total > 0 else 0
                coverage_values.append(coverage)
            
            summary["kg_coverage"]["coverage_by_type"][term_type] = np.mean(coverage_values)
        
        return summary
    
    def generate_visualizations(self) -> None:
        """Generate visualizations of evaluation results"""
        if not self.results:
            logger.warning("No results to visualize. Run evaluation first.")
            return
        
        summary = self.summarize_results()
        
        # Ensure output directory exists
        if not os.path.exists(self.settings["visualization_dir"]):
            os.makedirs(self.settings["visualization_dir"])
        
        # 1. KG Coverage by Term Type Visualization
        plt.figure(figsize=(10, 6))
        coverage_by_type = summary["kg_coverage"]["coverage_by_type"]
        types = list(coverage_by_type.keys())
        values = [coverage_by_type[t] for t in types]
        plt.bar(types, values, color='skyblue')
        plt.axhline(y=summary["kg_coverage"]["average_coverage"], color='r', linestyle='-', label=f'Avg: {summary["kg_coverage"]["average_coverage"]:.1f}%')
        plt.title('Knowledge Graph Coverage by Term Type')
        plt.xlabel('Term Type')
        plt.ylabel('Coverage (%)')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.settings['visualization_dir']}/kg_coverage_by_type.png")
        
        # 2. KG Contribution Comparison
        plt.figure(figsize=(8, 6))
        contribution_types = ['KG Strict', 'KG Informed']
        values = [
            summary["kg_contribution"]["average_strict_value"],
            summary["kg_contribution"]["average_informed_value"]
        ]
        plt.bar(contribution_types, values, color=['coral', 'mediumseagreen'])
        plt.title('Knowledge Graph Contribution (0-10 scale)')
        plt.xlabel('Approach')
        plt.ylabel('Value Added Score')
        plt.ylim(0, 10)
        plt.savefig(f"{self.settings['visualization_dir']}/kg_contribution.png")
        
        # 3. Timing Analysis
        plt.figure(figsize=(12, 6))
        stages = list(summary["timing"]["average_by_stage"].keys())
        times = [summary["timing"]["average_by_stage"][s] for s in stages]
        plt.bar(stages, times, color='lightgreen')
        plt.title('Average Processing Time by Stage')
        plt.xlabel('Stage')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.settings['visualization_dir']}/timing_analysis.png")
        
        # 4. Citation Quality
        plt.figure(figsize=(8, 6))
        citation_metrics = ['Citation Quality Score', 'Citation Accuracy (%)']
        values = [
            summary["citation_quality"]["average_score"],
            summary["citation_quality"]["average_accuracy"]
        ]
        plt.bar(citation_metrics, values, color=['purple', 'gold'])
        plt.title('Citation Quality Metrics')
        plt.ylabel('Score')
        plt.savefig(f"{self.settings['visualization_dir']}/citation_quality.png")
        
        # 5. Advanced Features Effectiveness
        plt.figure(figsize=(8, 6))
        feature_metrics = ['Multihop Paths Found', 'Vector Matches Found']
        values = [
            summary["multihop_stats"]["average_paths_found"],
            summary["vector_search_stats"]["vector_matches_found"] / len(self.results)
        ]
        plt.bar(feature_metrics, values, color=['teal', 'orange'])
        plt.title('Advanced Features Impact')
        plt.ylabel('Average Count per Question')
        plt.savefig(f"{self.settings['visualization_dir']}/advanced_features.png")
        
        # 6. LLM vs KG Performance Comparison
        plt.figure(figsize=(10, 6))
        models = ['LLM Only', 'KG Strict', 'KG Informed']
        performance = [
            summary["llm_performance"]["average_overall"], 
            summary["kg_contribution"]["average_strict_value"],
            summary["kg_contribution"]["average_informed_value"]
        ]
        plt.bar(models, performance, color=['lightblue', 'coral', 'mediumseagreen'])
        plt.title('Performance Comparison (0-10 scale)')
        plt.xlabel('Approach')
        plt.ylabel('Quality Score')
        plt.ylim(0, 10)
        plt.savefig(f"{self.settings['visualization_dir']}/performance_comparison.png")
        
        # 7. Generate detailed report
        report = f"""
        # Knowledge Graph Evaluation Report

        ## Summary
        - Questions Processed: {summary['questions_processed']}
        - Average KG Coverage: {summary['kg_coverage']['average_coverage']:.2f}%
        - Average KG Contribution (Strict): {summary['kg_contribution']['average_strict_value']:.2f}/10
        - Average KG Contribution (Informed): {summary['kg_contribution']['average_informed_value']:.2f}/10
        - Average Citation Quality: {summary['citation_quality']['average_score']:.2f}/10
        - Average Processing Time: {summary['timing']['average_total']:.2f} seconds

        ## Advanced Features Performance
        - Multihop Paths: {summary['multihop_stats']['average_paths_found']:.2f} paths per question
        - Questions with Multihop Paths: {summary['multihop_stats']['questions_with_paths']} out of {summary['questions_processed']}
        - Vector Search Matches: {summary['vector_search_stats']['vector_matches_found']} matches total

        ## KG Coverage by Term Type
        {pd.DataFrame([{'Term Type': t, 'Coverage (%)': f"{v:.2f}%"} for t, v in summary['kg_coverage']['coverage_by_type'].items()]).to_markdown(index=False)}

        ## Processing Time by Stage
        {pd.DataFrame([{'Stage': s, 'Time (s)': f"{t:.2f}s"} for s, t in summary['timing']['average_by_stage'].items()]).to_markdown(index=False)}

        ## Top Performing Questions
        {pd.DataFrame([{
            'Question': r['question'][:50] + '...',
            'KG Coverage': f"{r['kg_coverage']['coverage_percentage']:.2f}%",
            'KG Value Added': f"{r['evaluations'].get('kg_contribution_strict', {}).get('value_added_score', 0)}/10",
            'Multihop Paths': r.get("multihop_paths_count", 0)
        } for r in sorted(self.results, key=lambda x: x['evaluations'].get('kg_contribution_strict', {}).get('value_added_score', 0), reverse=True)[:5]]).to_markdown(index=False)}

        ## Lowest Performing Questions
        {pd.DataFrame([{
            'Question': r['question'][:50] + '...',
            'KG Coverage': f"{r['kg_coverage']['coverage_percentage']:.2f}%",
            'KG Value Added': f"{r['evaluations'].get('kg_contribution_strict', {}).get('value_added_score', 0)}/10",
            'Missing Terms': ', '.join(r['kg_coverage'].get('missing_terms', [])[:3])
        } for r in sorted(self.results, key=lambda x: x['evaluations'].get('kg_contribution_strict', {}).get('value_added_score', 0))[:5]]).to_markdown(index=False)}

        ## Common Missing Term Types
        {pd.DataFrame([{
            'Term Type': t,
            'Frequency Missing': sum(1 for r in self.results if t in r['kg_coverage'].get('missing_by_type', {})),
            'Example Terms': ', '.join([term for r in self.results for term in r['kg_coverage'].get('missing_by_type', {}).get(t, [])[:3]])
        } for t in sorted(set(t for r in self.results for t in r['kg_coverage'].get('missing_by_type', {})))]).to_markdown(index=False)}
        
        ## Notable Multihop Connections
        {pd.DataFrame([{
            'Question': r['question'][:30] + '...',
            'Path': p.get('path_description', 'Unknown path'),
            'Length': p.get('path_length', 0)
        } for r in self.results if 'multihop_paths' in r for p in r.get('multihop_paths', [])[:5]]).to_markdown(index=False) if any('multihop_paths' in r for r in self.results) else "No notable multihop connections found."}

        ## LLM Performance
        - Average Accuracy: {summary["llm_performance"]["average_accuracy"]:.2f}/10
        - Average Comprehensiveness: {summary["llm_performance"]["average_comprehensiveness"]:.2f}/10
        - Average Overall Quality: {summary["llm_performance"]["average_overall"]:.2f}/10

        ## Performance Comparison
        | Approach     | Quality Score |
        |:-------------|:--------------|
        | LLM Only     | {summary["llm_performance"]["average_overall"]:.2f}/10 |
        | KG Strict    | {summary["kg_contribution"]["average_strict_value"]:.2f}/10 |
        | KG Informed  | {summary["kg_contribution"]["average_informed_value"]:.2f}/10 |
        """
        
        with open(f"{self.settings['visualization_dir']}/evaluation_report.md", 'w') as f:
            f.write(report)
        
        logger.info(f"Visualizations and report saved to {self.settings['visualization_dir']}")

    def analyze_question_type(self, question: str, extracted_terms: List[Dict]) -> Dict:
        """Analyze the question to determine its medical focus area"""
        # Check for diagnostic terms
        diagnostic_keywords = ["diagnose", "diagnosis", "cause", "etiology", "differential", "why"]
        
        # Check for treatment terms
        treatment_keywords = ["treat", "therapy", "management", "drug", "medication", "regimen"]
        
        # Check for prognosis terms
        prognosis_keywords = ["prognosis", "outcome", "survival", "mortality", "complication"]
        
        # Count keyword occurrences
        diagnostic_count = sum(1 for keyword in diagnostic_keywords if keyword.lower() in question.lower())
        treatment_count = sum(1 for keyword in treatment_keywords if keyword.lower() in question.lower())
        prognosis_count = sum(1 for keyword in prognosis_keywords if keyword.lower() in question.lower())
        
        # Check for specific term types in extracted terms
        has_disease = any(term['type'] == 'Disease' for term in extracted_terms)
        has_drug = any(term['type'] == 'Drug' for term in extracted_terms)
        has_procedure = any(term['type'] == 'Procedure' for term in extracted_terms)
        
        # Determine primary and secondary types
        question_types = []
        
        if "which of the following is the most likely cause" in question.lower():
            question_types.append("diagnostic")
        elif "treatment" in question.lower() or "manage" in question.lower():
            question_types.append("therapeutic")
        elif diagnostic_count > treatment_count and diagnostic_count > prognosis_count:
            question_types.append("diagnostic")
        elif treatment_count > diagnostic_count and treatment_count > prognosis_count:
            question_types.append("therapeutic")
        elif prognosis_count > diagnostic_count and prognosis_count > treatment_count:
            question_types.append("prognostic")
        
        # Add secondary focus based on extracted terms
        if has_disease and has_drug:
            question_types.append("treatment-disease")
        elif has_disease and not has_drug:
            question_types.append("disease-focused")
        elif has_drug and not has_disease:
            question_types.append("medication-focused")
        
        if has_procedure:
            question_types.append("procedure-related")
        
        return {
            "primary_type": question_types[0] if question_types else "general",
            "secondary_types": question_types[1:],
            "has_disease": has_disease,
            "has_drug": has_drug,
            "has_procedure": has_procedure
        }

    def evaluate_evidence_quality(self, answer: str, kg_data: Dict) -> Dict:
        """Quantify evidence integration in the answer"""
        # 1. Count citations to KG concepts/relationships/paths
        concept_citations = re.findall(r'C\d+', answer)
        relationship_citations = re.findall(r'R\d+', answer)
        path_citations = re.findall(r'P\d+', answer)
        
        # 2. Check textbook quote usage
        textbook_refs = re.findall(r'\[REF\d+\]', answer)
        
        # 3. Detect conflict resolution patterns
        conflict_keywords = ['however', 'despite', 'contrary to', 'resolves this by', 
                             'conflict', 'inconsistent', 'disagree', 'differs']
        conflict_patterns = [kw in answer.lower() for kw in conflict_keywords]
        conflict_score = min(1.0, sum(conflict_patterns) * 0.2)  # Scale up to 1.0 max
        
        # 4. Calculate citation density
        answer_length = len(answer.split())
        citation_count = len(concept_citations) + len(relationship_citations) + len(path_citations) + len(textbook_refs)
        citation_density = min(1.0, citation_count / (answer_length/100) * 0.1)  # Normalized to 0-1
        
        # 5. Source balance (using both KG and textbook)
        kg_citations = len(concept_citations) + len(relationship_citations) + len(path_citations)
        textbook_citations = len(textbook_refs)
        has_kg = kg_citations > 0 
        has_textbook = textbook_citations > 0
        source_balance = 1.0 if (has_kg and has_textbook) else (0.5 if (has_kg or has_textbook) else 0.0)
        
        # 6. Traceability (percentage of concepts cited from available)
        available_concepts = len(kg_data.get('concepts', []))
        concept_citation_ratio = len(set(concept_citations)) / max(1, available_concepts)
        traceability = min(1.0, concept_citation_ratio * 2)  # Scale up, aim for 50%+ coverage
        
        # Bonus for explicit evidence discussion
        if re.search(r'evidence (shows|indicates|suggests|supports|demonstrates)', answer.lower()):
            traceability = min(1.0, traceability + 0.2)
        
        return {
            'citation_density': citation_density, 
            'source_diversity': source_balance,
            'conflict_resolution': conflict_score,
            'traceability': traceability,
            'raw_counts': {
                'concept_citations': len(concept_citations),
                'relationship_citations': len(relationship_citations),
                'path_citations': len(path_citations),
                'textbook_citations': len(textbook_refs),
                'total_citations': citation_count,
                'answer_length': answer_length
            }
        }

    def evaluate_correctness(self, answer: str, question_dict: Dict) -> Dict:
        """Evaluate the correctness of the answer"""
        correct_answer = question_dict.get("answer", "Unknown")
        
        # Basic correctness check
        answer_lower = answer.lower()
        correct_answer_lower = correct_answer.lower()
        
        contains_correct = correct_answer_lower in answer_lower
        
        # Extract answer choice if multiple choice
        extracted_choice = self.extract_answer_choice(answer)
        choice_correct = False
        
        # For multiple choice questions, check if the selected letter is correct
        options = question_dict.get("options", [])
        if options and correct_answer in options:
            correct_index = options.index(correct_answer)
            correct_letter = chr(65 + correct_index)  # A, B, C, D, E...
            choice_correct = extracted_choice == correct_letter
        
        # Check for error detection patterns
        error_detection_phrases = [
            "limitation in the evidence", 
            "evidence is insufficient",
            "evidence does not directly address",
            "lacking direct evidence",
            "conflicting evidence",
            "need additional information"
        ]
        
        error_detection_score = 0.0
        for phrase in error_detection_phrases:
            if phrase.lower() in answer_lower:
                error_detection_score += 0.2
        error_detection_score = min(1.0, error_detection_score)
        
        # Calculate factual accuracy score - MAKE THIS STRICTER
        factual_accuracy = 0.0
        
        # If it's a multiple choice question and we have options
        if extracted_choice and options:
            # If the correct letter was chosen, give full marks
            if choice_correct:
                factual_accuracy = 1.0
            # If wrong letter but right content, give partial marks
            elif contains_correct:
                factual_accuracy = 0.3
            # Otherwise, very low score for wrong answer
            else:
                factual_accuracy = 0.1
        # If not clearly multiple choice, use more lenient content-based scoring
        elif contains_correct or self._partial_match(answer_lower, correct_answer_lower):
            factual_accuracy = 0.8
        
        # Use LLM to evaluate factual accuracy if correct answer isn't known
        if correct_answer == "Unknown":
            prompt = f"""
            As a medical expert, evaluate the factual accuracy of this answer:
            
            Question: {question_dict.get('question', '')}
            
            Answer: {answer}
            
            Rate the factual accuracy from 0.0 to 1.0, where:
            - 0.0: Contains significant medical errors or misinformation
            - 0.5: Contains minor inaccuracies but mostly correct
            - 1.0: Medically accurate with no errors
            
            Return only the numeric score between 0.0 and 1.0.
            """
            try:
                factual_accuracy = float(self.llm(prompt))
            except:
                factual_accuracy = 0.5  # Default if conversion fails
        
        # Scale factual accuracy to 0-10 range for final score
        factual_accuracy_score = factual_accuracy * 10
        
        return {
            'factual_accuracy': factual_accuracy,
            'error_detection': error_detection_score,
            'contains_correct_answer': contains_correct,
            'choice_correct': choice_correct,
            'extracted_choice': extracted_choice,
            'factual_accuracy_score': factual_accuracy_score  # Add this scaled score
        }

    def _partial_match(self, answer: str, correct_answer: str) -> bool:
        """Check for partial matches between the answer and correct answer"""
        # Break down correct answer into key phrases
        key_terms = [term.strip() for term in correct_answer.split() if len(term) > 4]
        
        # Check if at least half of key terms appear
        matches = sum(1 for term in key_terms if term in answer)
        return matches >= len(key_terms) / 2

    def extract_answer_choice(self, answer: str) -> str:
        """Extract answer choice (A, B, C, D, E) from the answer"""
        patterns = [
            r"ANSWER CHOICE:\s*([A-E])",
            r"ANSWER:\s*([A-E])",
            r"The answer is\s*([A-E])\b",
            r"The correct answer is\s*([A-E])\b",
            r"Answer:\s*([A-E])\b",
            r"Option\s*([A-E])\b",
            r"^([A-E])\.\s"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return ""

    def calculate_evidence_based_score(self, evidence_metrics: Dict, correctness_metrics: Dict) -> Dict:
        """Calculate final score with emphasis on evidence quality"""
        # Evidence Quality (60%)
        evidence_score = (
            evidence_metrics['citation_density'] * SUB_METRICS['evidence_quality']['citation_density'] +
            evidence_metrics['source_diversity'] * SUB_METRICS['evidence_quality']['source_diversity'] +
            evidence_metrics['conflict_resolution'] * SUB_METRICS['evidence_quality']['conflict_resolution'] +
            evidence_metrics['traceability'] * SUB_METRICS['evidence_quality']['traceability']
        ) * 10  # Scale to 0-10

        # Correctness (40%)
        correctness_score = (
            correctness_metrics['factual_accuracy'] * SUB_METRICS['correctness']['factual_accuracy'] +
            correctness_metrics['error_detection'] * SUB_METRICS['correctness']['error_detection']
        ) * 10

        # Combined score with weights
        combined_score = (
            (evidence_score * EVALUATION_WEIGHTS['evidence_quality']) +
            (correctness_score * EVALUATION_WEIGHTS['correctness'])
        )
        
        # Thresholds for minimum evidence
        if evidence_metrics['raw_counts']['total_citations'] < 3:
            combined_score = min(combined_score, 7.0)  # Cap at 7/10 if insufficient citations
        
        # Bonus for handling conflicts
        if evidence_metrics['conflict_resolution'] > 0.5:
            combined_score = min(10.0, combined_score + 0.5)
        
        return {
            'evidence_score': round(evidence_score, 2),
            'correctness_score': round(correctness_score, 2),
            'combined_score': round(combined_score, 2),
            'evidence_quality_details': evidence_metrics,
            'correctness_details': correctness_metrics
        }

    def evaluate_with_evidence_priority(self, answer: str, question_dict: Dict, kg_data: Dict) -> Dict:
        """Evaluate an answer with priority given to evidence quality over correctness"""
        evidence_metrics = self.evaluate_evidence_quality(answer, kg_data)
        correctness_metrics = self.evaluate_correctness(answer, question_dict)
        
        return self.calculate_evidence_based_score(evidence_metrics, correctness_metrics)

    def rerank_concepts(self, concepts: List[Dict], question: str) -> List[Dict]:
        """Re-rank concepts based on relevance to the question"""
        if not concepts:
            return []
            
        # Save original ranking
        before_ranking = list(concepts)  # Create a copy
        
        # Calculate relevance scores
        for concept in concepts:
            concept_text = f"{concept['term']} - {concept['definition']}"
            similarity = self.calculate_semantic_similarity(concept_text, question)
            
            original_conf = concept.get('confidence', 0.5)
            concept['relevance_score'] = similarity
            concept['confidence'] = 0.3 * original_conf + 0.7 * similarity
        
        # Sort by combined confidence
        reranked = sorted(concepts, key=lambda x: x.get('confidence', 0), reverse=True)
        reranked = reranked[:self.settings.get("top_k_concepts", 50)]
        
        # Display the comparison
        self.show_ranking_comparison(before_ranking, reranked, 'concepts')
        
        return reranked

    def rerank_relationships(self, relationships: List[Dict], question: str, concepts: List[Dict]) -> List[Dict]:
        """Re-rank relationships based on relevance to the question"""
        if not relationships:
            return []
        
        # Save original ranking
        before_ranking = list(relationships)  # Create a copy
        
        # Calculate concept relevance dict for quick lookup
        concept_relevance = {c.get('cui'): c.get('confidence', 0) for c in concepts}
        
        for rel in relationships:
            source_cui = rel.get('source_cui')
            target_cui = rel.get('target_cui')
            source_relevance = concept_relevance.get(source_cui, 0)
            target_relevance = concept_relevance.get(target_cui, 0)
            
            rel_statement = f"{rel.get('source_name')} {rel.get('relationship_type')} {rel.get('target_name')}"
            semantic_relevance = self.calculate_semantic_similarity(rel_statement, question)
            
            rel['relevance_score'] = (
                0.4 * semantic_relevance +
                0.3 * source_relevance +
                0.3 * target_relevance
            )
        
        # Sort by relevance score
        reranked = sorted(relationships, key=lambda x: x.get('relevance_score', 0), reverse=True)
        reranked = reranked[:self.settings.get("top_k_relationships", 100)]
        
        # Display the comparison
        self.show_ranking_comparison(before_ranking, reranked, 'relationships')
        
        return reranked

    def rerank_paths(self, paths: List[Dict], question: str) -> List[Dict]:
        """Re-rank multihop paths based on relevance to the question"""
        if not paths:
            return []
        
        # Save original ranking
        before_ranking = list(paths)  # Create a copy
        
        for path in paths:
            path_text = path.get('path_description', '')
            
            path_relevance = self.calculate_semantic_similarity(path_text, question)
            
            path_length = path.get('path_length', 10)
            length_penalty = max(0, 1 - (path_length - 1) * 0.1)
            
            path['relevance_score'] = 0.8 * path_relevance + 0.2 * length_penalty
        
        # Sort by relevance score
        reranked = sorted(paths, key=lambda x: x.get('relevance_score', 0), reverse=True)
        reranked = reranked[:30]
        
        # Display the comparison
        self.show_ranking_comparison(before_ranking, reranked, 'paths')
        
        return reranked

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            if not self.embedding_model:
                logger.warning("No embedding model available for semantic similarity")
                return 0.5  # Default similarity score
                
            # Use the SentenceTransformer model
            embedding1 = self.embedding_model.encode(text1[:512], convert_to_tensor=False)
            embedding2 = self.embedding_model.encode(text2[:512], convert_to_tensor=False)
            
            # Calculate cosine similarity
            similarity = self.cosine_similarity(embedding1, embedding2)
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.5  # Default similarity score on error
        
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

    # Add this utility function to show ranking differences
    def show_ranking_comparison(self, before_items, after_items, item_type='concepts', limit=10):
        """Display items before and after re-ranking for comparison"""
        print(f"\n{'='*30} {item_type.upper()} RANKING COMPARISON {'='*30}")
        
        # Format for different item types
        if item_type == 'concepts':
            before_display = [(i, c.get('term', 'Unknown'), c.get('confidence', 0)) 
                              for i, c in enumerate(before_items[:limit])]
            after_display = [(i, c.get('term', 'Unknown'), c.get('confidence', 0), 
                              f"{c.get('relevance_score', 0):.3f}")
                             for i, c in enumerate(after_items[:limit])]
            
            print("\nBEFORE RE-RANKING:")
            print(f"{'#':3} | {'Term':40} | {'Conf':5}")
            print("-" * 55)
            for idx, term, conf in before_display:
                print(f"{idx:3} | {term[:40]:40} | {conf:.3f}")
            
            print("\nAFTER RE-RANKING:")
            print(f"{'#':3} | {'Term':40} | {'Conf':5} | {'Rel':5}")
            print("-" * 65)
            for idx, term, conf, rel in after_display:
                print(f"{idx:3} | {term[:40]:40} | {conf:.3f} | {rel}")
            
        elif item_type == 'relationships':
            before_display = [(i, f"{r.get('source_name', '?')} → {r.get('relationship_type', '?')} → {r.get('target_name', '?')}")
                              for i, r in enumerate(before_items[:limit])]
            
            after_display = [(i, f"{r.get('source_name', '?')} → {r.get('relationship_type', '?')} → {r.get('target_name', '?')}", 
                              f"{r.get('relevance_score', 0):.3f}")
                             for i, r in enumerate(after_items[:limit])]
            
            print("\nBEFORE RE-RANKING:")
            print(f"{'#':3} | {'Relationship':80}")
            print("-" * 85)
            for idx, rel in before_display:
                print(f"{idx:3} | {rel[:80]}")
            
            print("\nAFTER RE-RANKING:")
            print(f"{'#':3} | {'Relationship':80} | {'Score':5}")
            print("-" * 95)
            for idx, rel, score in after_display:
                print(f"{idx:3} | {rel[:80]:80} | {score}")
            
        elif item_type == 'paths':
            before_display = [(i, p.get('path_description', 'Unknown')[:70], p.get('path_length', 0)) 
                              for i, p in enumerate(before_items[:limit])]
            
            after_display = [(i, p.get('path_description', 'Unknown')[:70], 
                              p.get('path_length', 0), f"{p.get('relevance_score', 0):.3f}") 
                             for i, p in enumerate(after_items[:limit])]
            
            print("\nBEFORE RE-RANKING:")
            print(f"{'#':3} | {'Path Description':70} | {'Len':3}")
            print("-" * 80)
            for idx, desc, length in before_display:
                print(f"{idx:3} | {desc:70} | {length:3}")
            
            print("\nAFTER RE-RANKING:")
            print(f"{'#':3} | {'Path Description':70} | {'Len':3} | {'Score':5}")
            print("-" * 90)
            for idx, desc, length, score in after_display:
                print(f"{idx:3} | {desc:70} | {length:3} | {score}")
            
        print(f"{'='*80}\n")

    def evaluate_kg_contribution(self, answer_with_context: str, answer_without_context: str) -> Dict:
        """Evaluate how much the knowledge graph context contributed to the answer"""
        
        prompt = f"""
        Compare these two answers to the same medical question:
        
        ANSWER WITH CONTEXT:
        {answer_with_context}
        
        ANSWER WITHOUT CONTEXT:
        {answer_without_context}
        
        Evaluate how the knowledge graph/context contributed to the first answer compared to the second.
        
        Return a JSON with:
        {{
          "additional_facts": [list of facts present in the first answer but not in the second],
          "improved_reasoning": boolean (true if reasoning is more thorough),
          "confidence_improvement": 0-10 (how much more confident the answer with context seems),
          "information_richness": 0-10 (how much more informative the answer with context is),
          "value_added_score": 0-10 (overall contribution of the context),
          "specific_contributions": [list of specific ways the context improved the answer],
          "analysis": "brief explanation of how context enhanced the answer"
        }}
        """
        
        response = self.llm(prompt)
        
        try:
            # Extract the JSON from the response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()
            else:
                # Try to extract using regex for JSON pattern
                import re
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_text = response
            
            import json
            result = json.loads(json_text)
            
            # Ensure the value_added_score key exists
            if "contribution_score" in result and "value_added_score" not in result:
                result["value_added_score"] = result["contribution_score"]
            
            # If no value_added_score exists, set a default based on other metrics
            if "value_added_score" not in result:
                info_richness = result.get("information_richness", 0)
                confidence = result.get("confidence_improvement", 0)
                result["value_added_score"] = (info_richness + confidence) / 2
            
            # Ensure score is in the right range
            result["value_added_score"] = min(max(result["value_added_score"], 0), 10)
            
            return result
        except Exception as e:
            logger.error(f"Error parsing contribution evaluation: {str(e)}")
            return {
                "additional_facts": [],
                "improved_reasoning": False,
                "confidence_improvement": 0,
                "information_richness": 0,
                "value_added_score": 5.0,  # Default middle score
                "contribution_score": 5.0,  # For backward compatibility
                "specific_contributions": [],
                "analysis": f"Error evaluating contribution: {str(e)}"
            }

    def evaluate_hybrid_answer(self, answer: str, question_dict: Dict, kg_data: Dict) -> Dict:
        """
        Evaluate an answer using a hybrid approach combining semantic similarity for correctness (40%)
        and LLM-based evaluation for evidence quality (60%)
        """
        # Get evidence quality metrics (60%)
        evidence_metrics = self.evaluate_evidence_quality(answer, kg_data)
        
        # Get correctness metrics using semantic similarity (40%)
        correctness_metrics = self.evaluate_correctness_with_similarity(answer, question_dict)
        
        # Calculate the combined score
        combined_score = self.calculate_hybrid_score(evidence_metrics, correctness_metrics)
        
        return combined_score

    def evaluate_correctness_with_similarity(self, answer, question_dict):
        """Evaluate correctness using binary scoring with improved answer extraction and comparison"""
        # Basic validation
        if not answer or not question_dict:
            return {"score": 0, "explanation": "Missing answer or question data"}

        # Extract ground truth
        ground_truth = question_dict.get("answer", "").strip()
        options = question_dict.get("options", [])
        
        # Extract candidate answer
        extracted_answer = self._extract_answer_choice(answer)
        
        # Map option letters to text if necessary
        option_mapping = {}
        if isinstance(options, list):
            for i, option_text in enumerate(options):
                letter = chr(65 + i)  # A, B, C, etc.
                option_mapping[letter] = option_text
                option_mapping[f"{letter}."] = option_text
        elif isinstance(options, dict):
            option_mapping = options
        
        # If extracted answer is just a letter, map it to text
        is_letter_answer = False
        answer_text = extracted_answer
        
        # Check if the extracted answer is a letter (A, B, C, etc.)
        if re.match(r'^[A-Za-z]\.?$', extracted_answer):
            is_letter_answer = True
            letter = extracted_answer.replace(".", "").upper()
            if letter in option_mapping:
                answer_text = option_mapping[letter]
            else:
                return {"score": 0, "explanation": f"Answer '{extracted_answer}' is not a valid option letter"}
        
        # Check for exact match (case-insensitive)
        exact_match = False
        
        # Compare with ground truth
        if ground_truth.lower() == answer_text.lower():
            exact_match = True
        
        # If not an exact match, also check if the letter matches
        if not exact_match and not is_letter_answer:
            # Try to find the matching letter for the text answer
            matching_letter = None
            for letter, text in option_mapping.items():
                if text.lower() == answer_text.lower():
                    matching_letter = letter
                    break
            
            # Check if the ground truth might be provided as a letter
            if matching_letter and (ground_truth.upper() == matching_letter.upper() or 
                                    ground_truth.upper() == matching_letter.upper() + '.'):
                exact_match = True
        
        # Also check if ground truth is a letter and our answer is text
        if not exact_match and not is_letter_answer:
            if ground_truth.upper() in option_mapping:
                if answer_text.lower() == option_mapping[ground_truth.upper()].lower():
                    exact_match = True

        # Binary scoring - 10 if correct, 0 if wrong
        if exact_match:
            explanation = f"The answer '{extracted_answer}' correctly matches the ground truth '{ground_truth}'."
            return {"score": 10.0, "explanation": explanation}
        else:
            explanation = f"The answer '{extracted_answer}' does not match the ground truth '{ground_truth}'."
            return {"score": 0.0, "explanation": explanation}

    def _extract_answer_choice(self, answer: str) -> str:
        """Extract answer choice from text with improved detection of various formats"""
        if not answer:
            return ""
            
        # Search for patterns that indicate an answer choice
        patterns = [
            # Direct answer statements
            r"(?i)answer(?:\s+choice)?[\s:]+(.*?)(?:\.|$|\n)",
            r"(?i)the\s+(?:correct\s+)?answer\s+is\s+(.*?)(?:\.|$|\n)",
            r"(?i)I\s+(?:would\s+)?choose\s+(.*?)(?:\.|$|\n)",
            
            # Option letters
            r"(?i)option\s+([A-Z])(?:\.|$|\n)",
            r"(?i)answer\s+(?:is\s+)?([A-Z])(?:\.|$|\n)",
            
            # Markdown formatting (e.g., **C)
            r"\*\*\s*([A-Z])[.\s\*]*",
            
            # Finding bold text patterns
            r"\*\*([^*]+)\*\*"
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, answer)
            if matches:
                extracted = matches.group(1).strip()
                
                # If we got a longer phrase, check if it starts with a letter answer
                letter_match = re.match(r'^([A-Z])[\.\s)]', extracted)
                if letter_match:
                    return letter_match.group(1)
                    
                # Otherwise return the full extracted text
                return extracted
        
        # If no pattern matched, try to extract any letter that appears to be highlighted or emphasized
        emphasized = re.findall(r'[^a-zA-Z]([A-Z])[^a-zA-Z]', answer)
        if emphasized:
            # Count occurrences of each letter
            letter_counts = {}
            for letter in emphasized:
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
            
            # Get the most frequently emphasized letter
            if letter_counts:
                most_common = max(letter_counts.items(), key=lambda x: x[1])
                return most_common[0]
        
        # If all else fails, look for any standalone option letters
        option_letters = re.findall(r'(?:^|\s+)([A-Z])\.', answer)
        if option_letters:
            return option_letters[0]
            
        # Fallback: Return the first 50 chars
        return answer[:50] + "..."

    def calculate_hybrid_score(self, evidence_metrics: Dict, correctness_metrics: Dict) -> Dict:
        """Calculate final score with hybrid approach - 60% evidence, 40% correctness"""
        # Evidence Quality (60%) - using existing sub-metrics
        evidence_score = (
            evidence_metrics['citation_density'] * SUB_METRICS['evidence_quality']['citation_density'] +
            evidence_metrics['source_diversity'] * SUB_METRICS['evidence_quality']['source_diversity'] +
            evidence_metrics['conflict_resolution'] * SUB_METRICS['evidence_quality']['conflict_resolution'] +
            evidence_metrics['traceability'] * SUB_METRICS['evidence_quality']['traceability']
        ) * 10  # Scale to 0-10

        # Correctness (40%) - using semantic similarity approach
        correctness_score = (
            correctness_metrics['factual_accuracy'] * SUB_METRICS['correctness']['factual_accuracy'] +
            correctness_metrics['error_detection'] * SUB_METRICS['correctness']['error_detection']
        ) * 10

        # Combined score with weights (60% evidence, 40% correctness)
        combined_score = (
            (evidence_score * 0.6) +
            (correctness_score * 0.4)
        )
        
        # Bonus for handling conflicts and citing evidence
        if evidence_metrics['conflict_resolution'] > 0.5 and correctness_metrics['factual_accuracy'] > 0.8:
            combined_score = min(10.0, combined_score + 0.5)
        
        return {
            'evidence_score': round(evidence_score, 2),
            'correctness_score': round(correctness_score, 2),
            'combined_score': round(combined_score, 2),
            'evidence_quality_details': evidence_metrics,
            'correctness_details': correctness_metrics
        }

    def run_hybrid_evaluation(self, questions: List[Dict] = None, n: int = None) -> Dict:
        """Run evaluation using the hybrid approach"""
        if not questions:
            questions = self.load_test_questions(n=n or self.settings["test_size"])
        
        results = []
        
        for question in tqdm(questions, desc="Evaluating with hybrid approach"):
            try:
                # Extract terms from question
                extracted_terms = self.extract_key_terms(question["question"])
                
                # Get concepts, relationships, and paths
                concepts = self.get_concepts(extracted_terms)
                relationships = self.get_relationships(concepts)
                multihop_paths = self.find_multihop_paths(concepts)
                
                # Generate answer with KG data
                kg_data = self.format_kg_data(concepts, relationships, multihop_paths)
                answer = self.generate_answer(question["question"], concepts, relationships, multihop_paths)
                
                # Run hybrid evaluation
                evaluation = self.evaluate_hybrid_answer(answer, question, kg_data)
                
                # Store results
                result = {
                    "question_id": question.get("id", "unknown"),
                    "question": question["question"],
                    "answer": answer,
                    "correct_answer": question.get("answer", ""),
                    "evaluation": evaluation,
                    "kg_stats": {
                        "concepts_count": len(concepts),
                        "relationships_count": len(relationships),
                        "multihop_paths_count": len(multihop_paths) if multihop_paths else 0
                    }
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
        
        # Calculate overall statistics
        evidence_scores = [r["evaluation"]["evidence_score"] for r in results]
        correctness_scores = [r["evaluation"]["correctness_score"] for r in results]
        combined_scores = [r["evaluation"]["combined_score"] for r in results]
        
        summary = {
            "questions_evaluated": len(results),
            "average_evidence_score": np.mean(evidence_scores),
            "average_correctness_score": np.mean(correctness_scores),
            "average_combined_score": np.mean(combined_scores),
            "results": results
        }
        
        return summary

def main():
    """Run the knowledge graph evaluation with optimized performance"""
    try:
        load_dotenv()
        
        # Initialize Neo4j connection
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j"
        )
        
        # Initialize OpenAI with caching to save API calls
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Add caching for LLM calls
        llm_cache = {}
        
        def cached_llm_function(prompt):
            if prompt in llm_cache:
                return llm_cache[prompt]
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a medical AI assistant with expertise in clinical medicine."}, 
                          {"role": "user", "content": prompt}],
                temperature=0.0
            ).choices[0].message.content
            
            llm_cache[prompt] = response
            return response
        
        # Initialize evaluator with cached LLM function
        evaluator = KnowledgeGraphEvaluator(
            graph=graph,
            llm_function=cached_llm_function
        )
        
        # Configure optimized settings with expanded parameters
        evaluator.settings = {
            "test_size": 5,  
            "top_k_concepts": 70,  # Increased from 50 to get more concepts
            "top_k_relationships": 150,  # Increased from 100 to get more relationships
            "visualization_dir": "kg_evaluation_enhanced",
            "vector_search_enabled": True,
            "multihop_enabled": True, 
            "multihop_max_depth": 3,
            "vector_search_threshold": 0.3,
            "enhanced_citations": True  # New flag for citation improvements
        }
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Generate summary and visualizations
        summary = evaluator.summarize_results()
        print(json.dumps(summary, indent=2))
        
        evaluator.generate_visualizations()
        
        print(f"Evaluation complete. Results saved to {evaluator.settings['visualization_dir']}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}", exc_info=True)
    
    # Load medical questions
    questions = []
    with open("usmle_categorized_questions.json", "r") as f:
        questions = json.load(f)
    
    # Run hybrid evaluation
    print("Running hybrid evaluation...")
    hybrid_results = evaluator.run_hybrid_evaluation(questions=questions, n=10)
    
    # Print results
    print(f"\nEvaluation Results (n={hybrid_results['questions_evaluated']}):")
    print(f"Average Evidence Score: {hybrid_results['average_evidence_score']:.2f}/10")
    print(f"Average Correctness Score: {hybrid_results['average_correctness_score']:.2f}/10")
    print(f"Average Combined Score: {hybrid_results['average_combined_score']:.2f}/10")
    
    # Save detailed results
    with open("hybrid_evaluation_results.json", "w") as f:
        json.dump(hybrid_results, f, indent=2)
    
    print("\nDetailed results saved to hybrid_evaluation_results.json")
    
if __name__ == "__main__":
    main()