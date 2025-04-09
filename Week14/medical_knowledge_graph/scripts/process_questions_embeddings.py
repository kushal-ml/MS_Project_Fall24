import sys
from pathlib import Path
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
import re
import time
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

# Add the project root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from openai import OpenAI
import logging
from src.processors.umls_processor import UMLSProcessor, STEncoder
from src.processors.question_processor import QuestionProcessor
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalKnowledgeSearch:
    """Enhanced medical knowledge search combining vector similarity and keyword search"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
        self.embedding_model = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the embedding model lazily (only when needed)"""
        if self.initialized:
            return True
            
        try:
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            logger.info(f"Initializing embedding model: {model_name}")
            
            # Determine if CUDA is available for GPU acceleration
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create a custom encoder using the HuggingFace model
            self.embedding_model = STEncoder(model)
            logger.info(f"Embedding model initialized successfully (using {device})")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            logger.warning("Vector similarity search will not be available")
            return False
    
    def search(self, query: str, question_info: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform hybrid search using both vector similarity and keyword search
        
        Args:
            query: The search query
            question_info: Additional question metadata for context
            
        Returns:
            Tuple of (concepts, relationships)
        """
        # Initialize embedding model if not already done
        if not self.initialized:
            self.initialize()
        
        # Extract key terms for keyword search
        key_terms = self._extract_key_terms_from_info(question_info)
        
        # Perform vector search if model is available
        vector_concepts = []
        if self.embedding_model:
            vector_concepts = self._vector_search_with_retry(query, question_info)
        
        # Perform keyword search
        keyword_concepts = self._keyword_search(key_terms)
        
        # Merge and deduplicate results
        all_concepts = self._merge_concepts(vector_concepts, keyword_concepts)
        
        # Get relationships between concepts
        relationships = self._get_relationships(all_concepts, question_info.get('question_type', ''))
        
        return all_concepts, relationships
    
    def _extract_key_terms_from_info(self, question_info: Dict) -> List[Dict]:
        """Extract key terms from question info"""
        key_terms = question_info.get('key_terms', [])
        
        # Ensure each term has required fields
        formatted_terms = []
        for term in key_terms:
            if isinstance(term, dict) and 'term' in term:
                formatted_term = {
                    'term': term['term'],
                    'type': term.get('type', 'Unknown'),
                    'priority': term.get('priority', 3)
                }
                formatted_terms.append(formatted_term)
                
        return formatted_terms
    
    def _vector_search_with_retry(self, query: str, question_info: Dict) -> List[Dict]:
        """Perform vector search with retry logic for network resilience"""
        if not self.embedding_model:
            return []
            
        try:
            # Extract domain information
            question_type = question_info.get('question_type', '').lower()
            clinical_scenario = question_info.get('clinical_scenario', '')
            specific_question = question_info.get('specific_question', '')
            key_terms = question_info.get('key_terms', [])
            
            # Create priority term lists by category
            diseases = []
            drugs = []
            symptoms = []
            anatomy = []
            procedures = []
            other_terms = []
            
            for term_info in key_terms:
                if not isinstance(term_info, dict) or 'term' not in term_info:
                    continue
                    
                term = term_info['term']
                term_type = term_info.get('type', '').lower()
                
                if term_type == 'disease':
                    diseases.append(term)
                elif term_type == 'drug':
                    drugs.append(term)
                elif term_type == 'symptom':
                    symptoms.append(term)
                elif term_type == 'anatomy':
                    anatomy.append(term)
                elif term_type == 'procedure':
                    procedures.append(term)
                else:
                    other_terms.append(term)
            
            # Priority search phrases
            search_phrases = []
            
            # Add the full query
            search_phrases.append({
                'text': query,
                'weight': 1.0,
                'type': 'full_query'
            })
            
            # Add the clinical scenario with higher weight
            if clinical_scenario:
                search_phrases.append({
                    'text': clinical_scenario,
                    'weight': 1.2,
                    'type': 'clinical_scenario'
                })
            
            # Add the specific question with even higher weight
            if specific_question:
                search_phrases.append({
                    'text': specific_question,
                    'weight': 1.4,
                    'type': 'specific_question'
                })
            
            # Create targeted search phrases by medical domain
            if diseases:
                search_phrases.append({
                    'text': "Medical diseases including " + ", ".join(diseases),
                    'weight': 1.5,
                    'type': 'diseases'
                })
            
            if drugs:
                search_phrases.append({
                    'text': "Medications and pharmaceuticals including " + ", ".join(drugs),
                    'weight': 1.5,
                    'type': 'drugs'
                })
            
            if symptoms:
                search_phrases.append({
                    'text': "Clinical symptoms including " + ", ".join(symptoms),
                    'weight': 1.4,
                    'type': 'symptoms'
                })
                
            # Create domain-specific search phrases based on question type
            if question_type in ['diagnosis', 'pathophysiology']:
                domain_phrase = f"Pathophysiology and diagnosis of medical conditions related to {' '.join(diseases + symptoms)}"
                search_phrases.append({
                    'text': domain_phrase,
                    'weight': 1.6,
                    'type': 'domain_specific'
                })
            elif question_type in ['treatment', 'pharmacology']:
                domain_phrase = f"Treatment and pharmacology for medical conditions including {' '.join(diseases)}"
                search_phrases.append({
                    'text': domain_phrase,
                    'weight': 1.6,
                    'type': 'domain_specific'
                })
            
            # Domain-specific concepts by keyword detection
            if any(word in query.lower() for word in ['blood', 'plasma', 'anemia', 'transfusion', 'rh', 'antibody']):
                search_phrases.append({
                    'text': "Blood disorders, transfusion medicine, and immunohematology",
                    'weight': 1.7,
                    'type': 'domain_specific'
                })
            
            if any(word in query.lower() for word in ['kidney', 'renal', 'nephro', 'creatinine', 'glomerular']):
                search_phrases.append({
                    'text': "Renal physiology, kidney disease, and nephrology",
                    'weight': 1.7,
                    'type': 'domain_specific'
                })
            
            if any(word in query.lower() for word in ['pregnan', 'fetus', 'gestation', 'trimester', 'birth']):
                search_phrases.append({
                    'text': "Pregnancy, fetal development, and obstetrics",
                    'weight': 1.7,
                    'type': 'domain_specific'
                })
            
            # Medical semantic types to prioritize
            priority_types = [
                'T047',  # Disease or Syndrome
                'T048',  # Mental or Behavioral Dysfunction
                'T184',  # Sign or Symptom
                'T121',  # Pharmacologic Substance
                'T060',  # Diagnostic Procedure 
                'T061',  # Therapeutic or Preventive Procedure
                'T023',  # Body Part, Organ, or Organ Component
                'T129',  # Immunologic Factor
                'T116',  # Amino Acid, Peptide, or Protein
                'T123',  # Biologically Active Substance
                'T042',  # Organ or Tissue Function
                'T201',  # Clinical Attribute
            ]
            
            # Types to deprioritize
            low_priority_types = [
                'T071',  # Entity
                'T168',  # Food
                'T169',  # Functional Concept
                'T170',  # Intellectual Product
                'T131',  # Hazardous or Poisonous Substance
            ]
            
            # Collect vector search results
            all_vector_concepts = []
            
            # Define retry parameters
            max_retries = 3
            retry_delay = 2  # seconds
            
            # Process each search phrase
            for phrase_info in search_phrases:
                phrase = phrase_info['text']
                weight = phrase_info['weight']
                phrase_type = phrase_info['type']
                
                # Skip if phrase is empty
                if not phrase.strip():
                    continue
                
                logger.info(f"Searching with phrase: {phrase[:50]}... (type: {phrase_type})")
                
                # Generate embedding for this phrase
                try:
                    phrase_embedding = self.embedding_model.embed_query(phrase)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for phrase: {str(e)}")
                    continue
                
                # Try search with retries
                for attempt in range(max_retries):
                    try:
                        # Execute Neo4j vector search
                        cypher = """
                        CALL db.index.vector.queryNodes(
                            'concept_embeddings',
                            30,  // Increased limit for filtering
                            $embedding
                        ) YIELD node, score
                        WHERE score >= 0.85
                        
                        // Apply semantic type filtering and scoring
                        WITH node, score,
                            CASE
                                WHEN node.semantic_type IN $priority_types THEN score * 1.2
                                WHEN node.semantic_type IN $low_priority_types THEN score * 0.7
                                ELSE score
                            END AS adjusted_score,
                            CASE
                                WHEN node.term IS NULL THEN false
                                WHEN size(node.term) > 100 THEN false
                                WHEN node.term CONTAINS '(' AND node.term CONTAINS ')' AND size(node.term) > 50 THEN false
                                ELSE true
                            END AS valid_concept
                        
                        WHERE valid_concept = true
                        
                        RETURN 
                            node.cui as cui,
                            node.term as term,
                            node.semantic_type as semantic_type,
                            labels(node) as types,
                            adjusted_score as score
                        ORDER BY score DESC
                        LIMIT 20
                        """
                        
                        results = self.graph.query(cypher, {
                            'embedding': phrase_embedding,
                            'priority_types': priority_types,
                            'low_priority_types': low_priority_types
                        })
                        
                        if results:
                            # Process results
                            for result in results:
                                # Handle each result with its own try-except to prevent a single failure from breaking the loop
                                try:
                                    # Get definition
                                    def_query = """
                                    MATCH (c:Concept {cui: $cui})-[:HAS_DEFINITION]->(d:Definition)
                                    RETURN d.text as definition
                                    LIMIT 1
                                    """
                                    def_result = self.graph.query(def_query, {'cui': result['cui']})
                                    definition = def_result[0]['definition'] if def_result else None
                                    
                                    # Apply weighting and boosting
                                    weighted_score = result['score'] * weight
                                    
                                    # Boost score based on relevance to key terms
                                    relevance_boost = 1.0
                                    
                                    # Check if concept name or definition contains key terms
                                    concept_term = result['term'].lower()
                                    
                                    # Domain-specific boosting
                                    if any(term.lower() in concept_term for term in diseases):
                                        relevance_boost *= 1.3
                                    if any(term.lower() in concept_term for term in drugs):
                                        relevance_boost *= 1.2
                                    
                                    # Topic-specific boosting
                                    if question_type == 'pharmacology' and any(term in concept_term for term in ['drug', 'medication', 'dose', 'therapy']):
                                        relevance_boost *= 1.2
                                    elif question_type == 'diagnosis' and any(term in concept_term for term in ['symptom', 'sign', 'test', 'finding']):
                                        relevance_boost *= 1.2
                                    
                                    # Special domain boosts
                                    if any(term in concept_term for term in ['antibody', 'immune', 'rh', 'blood group']):
                                        relevance_boost *= 1.3  # Boost immunology terms
                                    if any(term in concept_term for term in ['pregnan', 'fetus', 'gestation']):
                                        relevance_boost *= 1.2  # Boost pregnancy-related terms
                                    if any(term in concept_term for term in ['kidney', 'renal', 'nephro']):
                                        relevance_boost *= 1.2  # Boost kidney-related terms
                                    
                                    # Calculate final score
                                    final_score = weighted_score * relevance_boost
                                    
                                    # Add to concepts list
                                    concept = {
                                        'cui': result['cui'],
                                        'term': result['term'],
                                        'semantic_type': result['semantic_type'],
                                        'types': result['types'],
                                        'definition': definition,
                                        'vector_score': float(final_score),  # Ensure it's a float for JSON serialization
                                        'source': 'vector',
                                        'phrase_type': phrase_type
                                    }
                                    all_vector_concepts.append(concept)
                                except Exception as result_error:
                                    logger.warning(f"Error processing result {result.get('cui', 'unknown')}: {str(result_error)}")
                        
                        # Break retry loop on success
                        break
                        
                    except Exception as e:
                        # Handle network errors with retries
                        if "ChunkedEncodingError" in str(e) or "ProtocolError" in str(e) or "Response ended prematurely" in str(e):
                            if attempt < max_retries - 1:
                                retry_time = retry_delay * (attempt + 1)
                                logger.warning(f"Network error during vector search: {str(e)}. Retrying in {retry_time}s (attempt {attempt+1}/{max_retries})")
                                time.sleep(retry_time)
                                continue
                        
                        logger.error(f"Failed vector search after {attempt+1} attempts: {str(e)}")
                        break
            
            # Step 1: Deduplicate by CUI
            unique_cuis = {}
            for concept in all_vector_concepts:
                cui = concept['cui']
                if cui not in unique_cuis or concept['vector_score'] > unique_cuis[cui]['vector_score']:
                    unique_cuis[cui] = concept
            
            # Step 2: Filter and sort results
            filtered_results = list(unique_cuis.values())
            filtered_results.sort(key=lambda x: x['vector_score'], reverse=True)
            
            # Step 3: Remove concepts that are likely irrelevant
            final_results = []
            for concept in filtered_results:
                # Skip concepts with very long names (typically complex chemicals)
                if len(concept['term']) > 100:
                    continue
                
                # Check for complex chemical names with many special characters
                term = concept['term'].lower()
                sem_type = concept.get('semantic_type', '')
                
                # Filter out complex chemical names unless they're important drug classes
                is_complex_chemical = (
                    re.search(r'[\(\)\[\]\{\}\-\d,]{10,}', term) and 
                    sem_type not in ['T121', 'T109', 'T116']  # Keep medications
                )
                
                if is_complex_chemical:
                    continue
                
                final_results.append(concept)
                
                # Limit to reasonable number of results
                if len(final_results) >= 25:
                    break
            
            logger.info(f"Vector search found {len(final_results)} relevant concepts")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def _keyword_search(self, terms: List[Dict]) -> List[Dict]:
        """Search for concepts using keyword matching"""
        try:
            if not terms:
                return []
                
            all_concepts = []
            
            # Define label mapping for term types
            label_mapping = {
                'disease': ['Disease'],
                'drug': ['Drug', 'Chemical'],
                'symptom': ['Symptom', 'Finding'],
                'anatomy': ['Anatomy', 'Body Part'],
                'procedure': ['Procedure'],
            }
            
            # Process terms by priority
            for priority in [1, 2, 3]:
                priority_terms = [t for t in terms if t.get('priority', 3) == priority]
                
                for term_info in priority_terms:
                    term = term_info['term']
                    term_type = term_info.get('type', '').lower()
                    
                    # Get appropriate labels
                    labels = label_mapping.get(term_type, ['Concept'])
                    
                    # Search with fuzzy matching
                    query = """
                    MATCH (c)
                    WHERE any(label IN $labels WHERE label IN labels(c))
                    AND (
                        toLower(c.term) CONTAINS toLower($term)
                        OR any(syn IN c.synonyms WHERE toLower(syn) CONTAINS toLower($term))
                    )
                    WITH c
                    OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d:Definition)
                    WITH c, d
                    ORDER BY 
                        CASE 
                            WHEN toLower(c.term) = toLower($term) THEN 0
                            WHEN toLower(c.term) STARTS WITH toLower($term) THEN 1
                            WHEN toLower(c.term) CONTAINS toLower($term) THEN 2
                            ELSE 3 
                        END,
                        CASE WHEN d.source = 'UMLS' THEN 0 ELSE 1 END
                    LIMIT 5
                    RETURN DISTINCT
                        c.cui as cui,
                        c.term as term,
                        d.text as definition,
                        c.semantic_type as semantic_type,
                        labels(c) as types,
                        $priority as priority,
                        'keyword' as source
                    """
                    
                    results = self.graph.query(query, {
                        'term': term,
                        'labels': labels,
                        'priority': priority
                    })
                    
                    if results:
                        # Add keyword match score
                        for result in results:
                            # Calculate match score based on term similarity and priority
                            match_score = 0.0
                            
                            # Match quality scores
                            if result['term'].lower() == term.lower():
                                match_score = 1.0  # Exact match
                            elif result['term'].lower().startswith(term.lower()):
                                match_score = 0.9  # Starts with term
                            elif term.lower() in result['term'].lower():
                                match_score = 0.8  # Contains term
                            else:
                                match_score = 0.7  # Synonym match
                            
                            # Adjust by priority
                            priority_factor = 1.0
                            if priority == 1:
                                priority_factor = 1.2
                            elif priority == 2:
                                priority_factor = 1.1
                            
                            # Set final score
                            result['keyword_score'] = match_score * priority_factor
                        
                        all_concepts.extend(results)
            
            return all_concepts
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def _merge_concepts(self, vector_concepts: List[Dict], keyword_concepts: List[Dict]) -> List[Dict]:
        """Merge vector and keyword search results, keeping the best score for each concept"""
        merged = {}
        
        # Process vector concepts first
        for concept in vector_concepts:
            cui = concept.get('cui')
            if cui:
                merged[cui] = concept
        
        # Add or enhance with keyword concepts
        for concept in keyword_concepts:
            cui = concept.get('cui')
            if cui is None:
                continue
                
            if cui not in merged:
                # Add new concept from keyword search
                concept['final_score'] = concept.get('keyword_score', 0.8)
                merged[cui] = concept
            else:
                # Enhance existing concept from vector search
                existing = merged[cui]
                
                # If keyword has definition but vector doesn't, add it
                if 'definition' not in existing and 'definition' in concept:
                    existing['definition'] = concept['definition']
                
                # If found by both methods, mark as hybrid
                existing['source'] = 'hybrid'
                
                # Combine scores for ranking
                vector_score = existing.get('vector_score', 0)
                keyword_score = concept.get('keyword_score', 0)
                
                # Use maximum of both scores, with slight boost for being found by both methods
                existing['final_score'] = max(vector_score, keyword_score) * 1.1
        
        # Ensure all concepts have a final_score
        for cui, concept in merged.items():
            if 'final_score' not in concept:
                if 'vector_score' in concept:
                    concept['final_score'] = concept['vector_score']
                elif 'keyword_score' in concept:
                    concept['final_score'] = concept['keyword_score']
                else:
                    concept['final_score'] = 0.5  # Default score
        
        # Convert to list and sort by final score
        results = list(merged.values())
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return results
    
    def _get_relationships(self, concepts: List[Dict], question_type: str) -> List[Dict]:
        """Get relationships between concepts using both generic and specialized queries"""
        try:
            if not concepts:
                return []
                
            # Extract CUIs for query
            cuis = [c['cui'] for c in concepts if 'cui' in c]
            
            if not cuis:
                return []
            
            # Get generic relationships
            generic_query = """
            // Direct relationships between concepts
            MATCH (c1)-[r]-(c2)
            WHERE c1.cui IN $cuis AND c2.cui IN $cuis AND c1.cui <> c2.cui
            
            // Calculate relevance score based on relationship context
            WITH c1, r, c2,
                 CASE
                     WHEN 'Disease' IN labels(c1) AND 'Drug' IN labels(c2) THEN 5  // Disease-Drug: high relevance
                     WHEN 'Drug' IN labels(c1) AND 'Disease' IN labels(c2) THEN 5  // Drug-Disease: high relevance
                     WHEN 'Disease' IN labels(c1) AND 'Symptom' IN labels(c2) THEN 4  // Disease-Symptom
                     WHEN 'Symptom' IN labels(c1) AND 'Disease' IN labels(c2) THEN 4  // Symptom-Disease
                     WHEN 'Drug' IN labels(c1) AND 'Chemical' IN labels(c2) THEN 4  // Drug-Chemical
                     WHEN 'Drug' IN labels(c1) AND 'Anatomy' IN labels(c2) THEN 3  // Drug-Anatomy
                     WHEN 'Disease' IN labels(c1) AND 'Anatomy' IN labels(c2) THEN 3  // Disease-Anatomy
                     ELSE 2  // Other relationships
                 END AS relevance_score
            
            RETURN 
                type(r) as relationship_type,
                c1.cui as source_cui,
                c1.term as source_name,
                c2.cui as target_cui,
                c2.term as target_name,
                labels(c1) as source_labels,
                labels(c2) as target_labels,
                relevance_score
            ORDER BY relevance_score DESC
            LIMIT 50
            """
            
            generic_relationships = self.graph.query(generic_query, {'cuis': cuis})
            
            # Get specialized relationships based on question type
            specialized_relationships = self._get_specialized_relationships(concepts, question_type)
            
            # Combine and deduplicate
            all_relationships = list(generic_relationships) + list(specialized_relationships)
            
            # Deduplicate relationships
            seen = set()
            unique_relationships = []
            
            for rel in all_relationships:
                key = (rel.get('relationship_type'), rel.get('source_cui'), rel.get('target_cui'))
                if key not in seen:
                    seen.add(key)
                    unique_relationships.append(rel)
            
            # Sort by relevance score
            unique_relationships.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return unique_relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            return []
            
    def _get_specialized_relationships(self, concepts: List[Dict], question_type: str) -> List[Dict]:
        """Get specialized relationships based on question type and available graph structure"""
        try:
            cuis = [c['cui'] for c in concepts if 'cui' in c]
            if not cuis:
                return []
                
            # Build query based on question type
            if question_type.lower() == 'diagnosis':
                query = """
                MATCH (d:Disease)-[:DISEASE_HAS_FINDING|DISEASE_MAY_HAVE_FI]->(s:Symptom)
                WHERE d.cui IN $cuis
                RETURN 'DIAGNOSTIC_SIGN' as relationship_type,
                       d.cui as source_cui, d.term as source_name,
                       s.cui as target_cui, s.term as target_name,
                       labels(d) as source_labels, labels(s) as target_labels,
                       5 as relevance_score
                UNION
                MATCH (d:Disease)-[:MAY_BE_DIAGNOSED_BY]->(p:Procedure)
                WHERE d.cui IN $cuis
                RETURN 'DIAGNOSTIC_PROCEDURE' as relationship_type,
                       d.cui as source_cui, d.term as source_name,
                       p.cui as target_cui, p.term as target_name,
                       labels(d) as source_labels, labels(p) as target_labels,
                       5 as relevance_score
                """
            elif question_type.lower() == 'treatment':
                query = """
                MATCH (d:Disease)-[:MAY_BE_TREATED_BY]->(t)
                WHERE d.cui IN $cuis
                RETURN 'TREATMENT' as relationship_type,
                       d.cui as source_cui, d.term as source_name,
                       t.cui as target_cui, t.term as target_name,
                       labels(d) as source_labels, labels(t) as target_labels,
                       5 as relevance_score
                UNION
                MATCH (drug:Drug)-[:MAY_TREAT]->(d:Disease)
                WHERE d.cui IN $cuis OR drug.cui IN $cuis
                RETURN 'TREATMENT' as relationship_type,
                       drug.cui as source_cui, drug.term as source_name,
                       d.cui as target_cui, d.term as target_name,
                       labels(drug) as source_labels, labels(d) as target_labels,
                       5 as relevance_score
                """
            elif question_type.lower() == 'etiology':
                query = """
                MATCH (d:Disease)-[:HAS_CAUSATIVE_AGENT]->(c)
                WHERE d.cui IN $cuis
                RETURN 'CAUSATIVE_AGENT' as relationship_type,
                       d.cui as source_cui, d.term as source_name,
                       c.cui as target_cui, c.term as target_name,
                       labels(d) as source_labels, labels(c) as target_labels,
                       5 as relevance_score
                UNION
                MATCH (c)-[:CAUSE_OF]->(d:Disease)
                WHERE d.cui IN $cuis
                RETURN 'CAUSE' as relationship_type,
                       c.cui as source_cui, c.term as source_name,
                       d.cui as target_cui, d.term as target_name,
                       labels(c) as source_labels, labels(d) as target_labels,
                       5 as relevance_score
                """
            elif question_type.lower() == 'pharmacology':
                query = """
                MATCH (drug:Drug)-[:HAS_INGREDIENT]->(i)
                WHERE drug.cui IN $cuis
                RETURN 'DRUG_INGREDIENT' as relationship_type,
                       drug.cui as source_cui, drug.term as source_name,
                       i.cui as target_cui, i.term as target_name,
                       labels(drug) as source_labels, labels(i) as target_labels,
                       4 as relevance_score
                UNION
                MATCH (drug:Drug)-[:CONTRAINDICATED_WITH]->(c)
                WHERE drug.cui IN $cuis
                RETURN 'CONTRAINDICATION' as relationship_type,
                       drug.cui as source_cui, drug.term as source_name,
                       c.cui as target_cui, c.term as target_name,
                       labels(drug) as source_labels, labels(c) as target_labels,
                       5 as relevance_score
                UNION
                MATCH (drug:Drug)-[:MAY_TREAT]->(d:Disease)
                WHERE drug.cui IN $cuis
                RETURN 'INDICATION' as relationship_type,
                       drug.cui as source_cui, drug.term as source_name,
                       d.cui as target_cui, d.term as target_name,
                       labels(drug) as source_labels, labels(d) as target_labels,
                       5 as relevance_score
                """
            elif any(term in question_type.lower() for term in ['blood', 'immun', 'hemato']):
                query = """
                MATCH (a:Concept)-[:ASSOCIATED_WITH]->(b:Concept)
                WHERE a.cui IN $cuis OR b.cui IN $cuis
                AND (a.term CONTAINS 'antibod' OR b.term CONTAINS 'antibod' 
                     OR a.term CONTAINS 'immune' OR b.term CONTAINS 'immune'
                     OR a.term CONTAINS 'blood' OR b.term CONTAINS 'blood')
                RETURN 'IMMUNE_RELATION' as relationship_type,
                       a.cui as source_cui, a.term as source_name,
                       b.cui as target_cui, b.term as target_name,
                       labels(a) as source_labels, labels(b) as target_labels,
                       5 as relevance_score
                """
            else:
                # Generic relationship query for other question types
                query = """
                MATCH (c1)-[r]-(c2)
                WHERE c1.cui IN $cuis AND c2.cui IN $cuis AND c1.cui <> c2.cui
                RETURN type(r) as relationship_type,
                       c1.cui as source_cui, c1.term as source_name,
                       c2.cui as target_cui, c2.term as target_name,
                       labels(c1) as source_labels, labels(c2) as target_labels,
                       3 as relevance_score
                """
                
            return self.graph.query(query, {'cuis': cuis})
            
        except Exception as e:
            logger.error(f"Error getting specialized relationships: {str(e)}")
            return []


class HybridMedicalQuestionProcessor:
    """Process medical questions using a hybrid knowledge-based approach"""
    
    def __init__(self, graph: Neo4jGraph, llm_function):
        self.graph = graph
        self.llm = llm_function
        
        # Initialize knowledge search
        self.knowledge_search = MedicalKnowledgeSearch(graph)
        
        # Question type definitions
        self.question_types = {
            "pathophysiology": "Understanding disease mechanisms and processes",
            "diagnosis": "Identifying diseases based on symptoms, test results, and clinical findings",
            "treatment": "Determining appropriate treatments and interventions",
            "pharmacology": "Drug mechanisms, interactions, and clinical applications",
            "basic_science": "Fundamental biological processes and medical science concepts",
            "preventive_medicine": "Preventive measures and public health considerations",
            "ethics": "Ethical considerations in medical practice",
            "etiology": "Causes and origins of diseases and disorders"
        }
        
        logger.info("Hybrid Medical Question Processor initialized")
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a medical question using hybrid knowledge search and reasoning
        
        Args:
            question: The medical question to process
            
        Returns:
            Dict containing the answer and supporting evidence
        """
        try:
            # Step 1: Analyze the question
            question_info = self._analyze_question(question)
            logger.info(f"Analyzed question: {question_info.get('question_type', 'Unknown')}")
            
            # Step 2: Search the knowledge base using hybrid approach
            concepts, relationships = self.knowledge_search.search(question, question_info)
            logger.info(f"Found {len(concepts)} concepts and {len(relationships)} relationships")
            
            # Step 3: Assess knowledge coverage
            knowledge_coverage = self._assess_knowledge_coverage(concepts, relationships, question_info)
            logger.info(f"Knowledge coverage assessment: {knowledge_coverage['coverage_level']}")
            
            # Step 4: Generate answer with knowledge-constrained reasoning
            answer = self._generate_answer(question, concepts, relationships, question_info, knowledge_coverage)
            
            # Return response with all components
            return {
                'answer': answer,
                'question_type': question_info.get('question_type', 'Unknown'),
                'concepts': concepts,
                'relationships': relationships,
                'knowledge_coverage': knowledge_coverage
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_question(self, question: str) -> Dict:
        """
        Analyze the question to extract structure and key elements
        
        Args:
            question: The medical question text
            
        Returns:
            Dict containing question metadata
        """
        prompt = f"""
        Analyze this medical question. Identify:
        1. The question type (pathophysiology, diagnosis, treatment, pharmacology, basic science, preventive medicine, ethics, etiology)
        2. Key medical terms (diseases, drugs, symptoms, anatomy, procedures)
        3. The clinical scenario described
        4. The specific question being asked
        5. If multiple choice, identify each option
        
        Question: {question}
        
        Return ONLY a JSON object with these fields:
        {{
            "question_type": "string",
            "key_terms": [
                {{"term": "string", "type": "disease|drug|symptom|anatomy|procedure", "priority": 1-3}}
            ],
            "clinical_scenario": "string",
            "specific_question": "string",
            "options": [
                {{"label": "A|B|C|D|E", "text": "string"}}
            ],
            "is_multiple_choice": true|false
        }}
        """
        
        response = self.llm(prompt)
        
        try:
            # Process the response to extract JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.split('```')[1]
            if cleaned_response.lower().startswith('json'):
                cleaned_response = cleaned_response.split('json', 1)[1]
            if cleaned_response.startswith('\n'):
                cleaned_response = cleaned_response.strip()
                
            # Parse JSON response
            question_info = json.loads(cleaned_response)
            
            # Validate and set defaults for missing fields
            if 'question_type' not in question_info:
                question_info['question_type'] = 'diagnosis'
            if 'key_terms' not in question_info:
                question_info['key_terms'] = []
            if 'clinical_scenario' not in question_info:
                question_info['clinical_scenario'] = ''
            if 'specific_question' not in question_info:
                question_info['specific_question'] = question
            
            return question_info
            
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            # Return basic info if analysis fails
            return {
                'question_type': 'diagnosis',
                'key_terms': [],
                'clinical_scenario': '',
                'specific_question': question,
                'options': [],
                'is_multiple_choice': False
            }
    
    def _assess_knowledge_coverage(self, concepts: List[Dict], relationships: List[Dict], question_info: Dict) -> Dict:
        """
        Assess how well the knowledge base covers the question topics
        
        Args:
            concepts: List of concepts found in the knowledge base
            relationships: List of relationships between concepts
            question_info: Metadata about the question
            
        Returns:
            Dict with coverage assessment
        """
        try:
            # Score concept coverage
            concept_coverage = 0.0
            if concepts:
                # Check if key terms are found in concepts
                key_terms = [t['term'].lower() for t in question_info.get('key_terms', [])]
                matched_terms = 0
                
                for term in key_terms:
                    # Check if any concept contains this term
                    for concept in concepts:
                        if term in concept.get('term', '').lower():
                            matched_terms += 1
                            break
                
                # Calculate concept coverage score
                if key_terms:
                    term_coverage = matched_terms / len(key_terms)
                    # Weight by number of concepts found, up to a reasonable limit
                    concept_weight = min(len(concepts) / 10, 1.0)
                    concept_coverage = term_coverage * 0.7 + concept_weight * 0.3
            
            # Score relationship coverage
            relationship_coverage = 0.0
            if relationships:
                # Get a score based on number of relationships found
                rel_count_score = min(len(relationships) / 10, 1.0)
                
                # Check if relationships connect key concepts
                if concepts and len(concepts) >= 2:
                    top_cuis = [c['cui'] for c in concepts[:5]]
                    relevant_rels = 0
                    
                    for rel in relationships:
                        if rel.get('source_cui') in top_cuis and rel.get('target_cui') in top_cuis:
                            relevant_rels += 1
                    
                    rel_relevance_score = min(relevant_rels / 3, 1.0)
                    relationship_coverage = rel_count_score * 0.5 + rel_relevance_score * 0.5
                else:
                    relationship_coverage = rel_count_score
            
            # Overall coverage score
            overall_score = concept_coverage * 0.6 + relationship_coverage * 0.4
            
            # Determine coverage level
            if overall_score >= 0.7:
                coverage_level = "high"
                confidence = "The knowledge base has strong coverage of the question topic."
            elif overall_score >= 0.4:
                coverage_level = "medium"
                confidence = "The knowledge base has moderate coverage of the question topic."
            else:
                coverage_level = "low"
                confidence = "The knowledge base has limited coverage of the question topic."
            
            # Get missing key terms
            missing_terms = []
            for term_info in question_info.get('key_terms', []):
                term = term_info['term'].lower()
                found = False
                for concept in concepts:
                    if term in concept.get('term', '').lower():
                        found = True
                        break
                if not found:
                    missing_terms.append(term_info['term'])
            
            return {
                'coverage_level': coverage_level,
                'confidence': confidence,
                'overall_score': overall_score,
                'concept_coverage': concept_coverage,
                'relationship_coverage': relationship_coverage,
                'missing_terms': missing_terms
            }
            
        except Exception as e:
            logger.error(f"Error assessing knowledge coverage: {str(e)}")
            return {
                'coverage_level': 'unknown',
                'confidence': 'Could not assess knowledge coverage due to an error.',
                'overall_score': 0.0,
                'concept_coverage': 0.0,
                'relationship_coverage': 0.0,
                'missing_terms': []
            }
    
    def _generate_answer(self, question: str, concepts: List[Dict], relationships: List[Dict], 
                         question_info: Dict, knowledge_coverage: Dict) -> str:
        """
        Generate an answer to the question based on available knowledge
        
        Args:
            question: The original question text
            concepts: List of relevant concepts
            relationships: List of relevant relationships
            question_info: Metadata about the question
            knowledge_coverage: Assessment of knowledge coverage
            
        Returns:
            Answer string with chain of thought reasoning
        """
        try:
            # Format concepts for the prompt
            concept_text = ""
            for i, concept in enumerate(concepts[:15], 1):
                score = concept.get('vector_score', concept.get('keyword_score', 'N/A'))
                if isinstance(score, float):
                    score = round(score, 2)
                    
                definition = concept.get('definition', 'No definition available')
                if definition:
                    # Truncate long definitions
                    if len(definition) > 200:
                        definition = definition[:200] + "..."
                
                concept_text += f"{i}. {concept.get('term', 'Unknown')} (CUI: {concept.get('cui', 'Unknown')}, Type: {concept.get('semantic_type', 'Unknown')}, Score: {score})\n   Definition: {definition}\n\n"
            
            # Format relationships for the prompt
            relationship_text = ""
            for i, rel in enumerate(relationships[:10], 1):
                rel_type = rel.get('relationship_type', 'Unknown')
                source = rel.get('source_name', 'Unknown')
                target = rel.get('target_name', 'Unknown')
                source_cui = rel.get('source_cui', 'Unknown')
                target_cui = rel.get('target_cui', 'Unknown')
                
                relationship_text += f"{i}. {rel_type}: {source} (CUI: {source_cui}) → {target} (CUI: {target_cui})\n"
            
            # Check if it's a multiple choice question
            is_multiple_choice = question_info.get('is_multiple_choice', False)
            options = question_info.get('options', [])
            options_text = ""
            
            if is_multiple_choice and options:
                options_text = "ANSWER OPTIONS:\n"
                for option in options:
                    options_text += f"{option.get('label', '?')}: {option.get('text', 'Unknown')}\n"
            
            # Determine prompt based on knowledge coverage
            coverage_level = knowledge_coverage.get('coverage_level', 'unknown')
            
            if coverage_level == "high":
                # Comprehensive answer with chain of thought
                reasoning_instruction = """
                Provide a chain of thought reasoning process to arrive at the answer.
                Your reasoning should be based ENTIRELY on the knowledge base data provided above.
                DO NOT use your own medical knowledge beyond what is provided in the concepts and relationships.
                
                Structure your answer as follows:
                1. IDENTIFY KEY FEATURES: Identify the most important clinical features from the question.
                2. ANALYZE CONCEPTS: Analyze the most relevant concepts and their definitions from the knowledge base.
                3. EVALUATE RELATIONSHIPS: Evaluate the relationships between concepts and their significance.
                4. REASONING CHAIN: Build a logical reasoning chain using ONLY the provided knowledge.
                5. CONCLUSION: State your final answer, clearly citing which knowledge graph concepts support it.
                
                If this is a multiple-choice question, analyze each option against the knowledge base data and select the one best supported.
                """
            elif coverage_level == "medium":
                # Partial knowledge with cautious reasoning
                reasoning_instruction = """
                Provide a chain of thought reasoning process using the available knowledge base data.
                Your reasoning should be based ONLY on the provided concepts and relationships.
                DO NOT use your own medical knowledge beyond what is provided.
                
                Structure your answer as follows:
                1. IDENTIFY KEY FEATURES: Identify the most important clinical features from the question.
                2. ANALYZE AVAILABLE CONCEPTS: Analyze the relevant concepts available in the knowledge base.
                3. KNOWLEDGE GAPS: Clearly identify what information appears to be missing from the knowledge base.
                4. LIMITED REASONING: Build a reasoning chain using ONLY the provided knowledge.
                5. TENTATIVE CONCLUSION: State what conclusion is suggested by the available knowledge, if any.
                
                Be explicit about limitations in your answer due to incomplete knowledge coverage.
                """
            else:
                # Limited knowledge with significant gaps
                reasoning_instruction = """
                The knowledge base has limited coverage of this topic. Your answer should:
                1. IDENTIFY THE KEY QUESTION: Clearly restate what the question is asking.
                2. AVAILABLE KNOWLEDGE: Summarize the limited relevant information found in the knowledge base.
                3. SIGNIFICANT KNOWLEDGE GAPS: Clearly identify what critical information is missing from the knowledge base.
                4. LIMITED CONCLUSION: State explicitly that the knowledge base does not contain sufficient information to provide a confident answer.
                
                DO NOT use your own medical knowledge to fill in gaps. Only use the provided concepts and relationships.
                """
            
            # Build the full prompt
            prompt = f"""
            You are a medical question answering system that relies EXCLUSIVELY on a medical knowledge graph.
            You MUST ONLY use the knowledge provided below and cannot use any other medical knowledge.
            
            QUESTION:
            {question}
            
            {options_text}
            
            KNOWLEDGE BASE DATA:
            
            KEY MEDICAL CONCEPTS:
            {concept_text if concept_text else "No relevant medical concepts found in knowledge base."}
            
            KEY CLINICAL RELATIONSHIPS:
            {relationship_text if relationship_text else "No relevant clinical relationships found in knowledge base."}
            
            KNOWLEDGE COVERAGE ASSESSMENT:
            {knowledge_coverage.get('confidence', 'Unknown')}
            
            ANSWER INSTRUCTIONS:
            {reasoning_instruction}
            
            If the knowledge base does not contain enough information to answer confidently, you MUST state this explicitly.
            """
            
            # Generate answer
            response = self.llm(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while generating an answer: {str(e)}"


def initialize_graph():
    """Initialize Neo4j graph connection"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials from environment
        uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        # Connect to Neo4j
        graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password
        )
        
        logger.info(f"Connected to Neo4j at {uri}")
        return graph
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        raise


def initialize_llm():
    """Initialize the LLM for reasoning"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        def llm_function(prompt):
            """Function to call the LLM API"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a medical expert analyzing data from a medical knowledge graph."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logger.error(f"Error calling LLM API: {str(e)}")
                return f"Error generating response: {str(e)}"
        
        logger.info("LLM reasoning engine initialized")
        return llm_function
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise


def process_usmle_question(question: str) -> Dict:
    """
    Process a USMLE-style medical question using the hybrid approach
    
    Args:
        question: The USMLE question text
        
    Returns:
        Dict containing the answer and supporting evidence
    """
    try:
        # Initialize Neo4j graph
        graph = initialize_graph()
        
        # Initialize LLM function
        llm_function = initialize_llm()
        
        # Create processor
        processor = HybridMedicalQuestionProcessor(graph, llm_function)
        
        # Process the question
        result = processor.process_question(question)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing USMLE question: {str(e)}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Check if question is provided as command line argument
    if len(sys.argv) > 1:
        # Get question from command line argument
        question_text = " ".join(sys.argv[1:])
    else:
        # Prompt user for input if no command line arguments
        print("\n=== Medical Knowledge Graph Question Processor ===")
        print("Please enter your medical question (press Enter twice when finished):")
        
        # Collect multi-line input
        lines = []
        while True:
            line = input()
            if not line and lines and not lines[-1]:  # Two consecutive empty lines
                lines.pop()  # Remove the last empty line
                break
            lines.append(line)
        
        question_text = "\n".join(lines)
        
        # If user didn't provide any input, use example question
        if not question_text.strip():
            print("No question provided. Using example question instead.")
            question_text = """
            A 45-year-old man presents with increasing fatigue, joint pain, and a rash on his face that worsens with sun exposure. 
            Laboratory studies show a positive antinuclear antibody (ANA) test and anti-double-stranded DNA antibodies. 
            Which of the following is the most likely diagnosis?
            A) Rheumatoid arthritis
            B) Systemic lupus erythematosus
            C) Psoriatic arthritis
            D) Dermatomyositis
            E) Scleroderma
            """
    
    print("\nProcessing question. This may take a minute...\n")
    
    # Process the question
    result = process_usmle_question(question_text)
    
    # Define output format
    output_mode = "detailed"  # Options: "simple", "detailed", "json"
    
    # Print formatted output
    if output_mode == "json":
        # Full JSON output
        print(json.dumps(result, indent=2))
    
    elif output_mode == "simple":
        # Simple output - just the answer
        print("\n=== ANSWER ===")
        print(result.get('answer', 'No answer generated'))
    
    else:  # detailed output
        # Detailed formatted output
        print("\n" + "="*80)
        print("MEDICAL KNOWLEDGE QUESTION ANALYSIS".center(80))
        print("="*80)
        
        # Print question info
        print("\n📝 QUESTION:")
        print(f"{question_text.strip()}")
        
        print("\n🔍 QUESTION TYPE:")
        print(f"{result.get('question_type', 'Unknown')}")
        
        # Print knowledge coverage
        coverage = result.get('knowledge_coverage', {})
        print("\n📊 KNOWLEDGE COVERAGE:")
        coverage_level = coverage.get('coverage_level', 'unknown')
        if coverage_level == "high":
            level_icon = "🟢"
        elif coverage_level == "medium":
            level_icon = "🟡"
        else:
            level_icon = "🔴"
            
        print(f"{level_icon} Level: {coverage_level}")
        print(f"   Confidence: {coverage.get('confidence', 'Unknown')}")
        
        if 'missing_terms' in coverage and coverage['missing_terms']:
            print(f"   Missing key terms: {', '.join(coverage['missing_terms'])}")
        
        # Print search results
        concepts = result.get('concepts', [])
        vector_concepts = [c for c in concepts if c.get('source') == 'vector']
        keyword_concepts = [c for c in concepts if c.get('source') == 'keyword']
        hybrid_concepts = [c for c in concepts if c.get('source') == 'hybrid']
        
        print("\n🔬 KNOWLEDGE BASE SEARCH RESULTS:")
        print(f"   Total concepts found: {len(concepts)}")
        print(f"   Vector search concepts: {len(vector_concepts)}")
        print(f"   Keyword search concepts: {len(keyword_concepts)}")
        print(f"   Hybrid matched concepts: {len(hybrid_concepts)}")
        
        # Top concepts
        if concepts:
            print("\n📚 TOP MEDICAL CONCEPTS:")
            for i, concept in enumerate(concepts[:5], 1):
                score = concept.get('final_score', 'N/A')
                if isinstance(score, float):
                    score = round(score, 2)
                source = concept.get('source', 'unknown')
                source_icon = "🔍" if source == "keyword" else "🧠" if source == "vector" else "⭐"
                
                print(f"{i}. {source_icon} {concept.get('term', 'Unknown')} (Score: {score})")
                definition = concept.get('definition')
                if definition:
                    # Truncate long definitions
                    if len(definition) > 100:
                        definition = definition[:100] + "..."
                    print(f"   Definition: {definition}")
        
        # Relationships
        relationships = result.get('relationships', [])
        if relationships:
            print("\n🔗 KEY RELATIONSHIPS:")
            for i, rel in enumerate(relationships[:3], 1):
                rel_type = rel.get('relationship_type', 'Unknown')
                source = rel.get('source_name', 'Unknown')
                target = rel.get('target_name', 'Unknown')
                
                print(f"{i}. {source} --[{rel_type}]--> {target}")
        
        # Print answer
        print("\n✅ ANSWER:")
        print(result.get('answer', 'No answer generated'))
        
        print("\n" + "="*80)