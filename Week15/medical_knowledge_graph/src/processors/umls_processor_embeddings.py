import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
from typing import List, Dict, Set
import logging
from datetime import datetime
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings
import sys
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from sentence_transformers import SentenceTransformer
from itertools import groupby

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.processors.base_processor import BaseProcessor, DatabaseMixin  
from src.config.constants import (USMLE_DOMAINS, SEMANTIC_TYPE_TO_LABEL, 
                                 RELATIONSHIP_TIERS, CONCEPT_TIERS, NEO4J_FREE_TIER_LIMITS,RELATION_TYPE_MAPPING,
                                 STEP1_PRIORITY, STEP2_PRIORITY, STEP3_PRIORITY)

logger = logging.getLogger(__name__)

class STEncoder(Embeddings):
    """Custom embeddings class for SentenceTransformer integration with LangChain"""
    def __init__(self, model):
        self.model = model
        
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

class UMLSProcessorEmbeddings(BaseProcessor, DatabaseMixin):
    def __init__(self, graph, max_retries=3, min_wait=1, max_wait=10, target_step="STEP2"):
        super().__init__(graph)
        self.batch_size = 30000
        self.num_workers = max(1, cpu_count() - 1)
        self.node_limit = NEO4J_FREE_TIER_LIMITS['max_nodes']
        self.relationship_limit = NEO4J_FREE_TIER_LIMITS['max_relationships']
        self.processed_concepts = set()
        self.semantic_types_dict = {} 
        self.target_step = target_step
        
        # Initialize statistics tracking
        self.init_stats_tracking()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
        
        # Load concepts for USMLE domains
        self.usmle_domains = USMLE_DOMAINS
        
        print(f"Loaded knowledge graph for {target_step} with {len(self.usmle_domains)} USMLE domains")
        print(f"Node limit: {self.node_limit}, Relationship limit: {self.relationship_limit}")
        
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait

    def init_stats_tracking(self):
        """Initialize statistics tracking"""
        self.stats = {
            "nodes": {
                "total": 0,
                "by_tier": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "other": 0},
                "by_type": {}
            },
            "relationships": {
                "total": 0,
                "by_tier": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "other": 0},
                "by_type": {}
            },
            "skipped": {
                "nodes": 0,
                "relationships": 0
            }
        }

    def _load_semantic_types(self, mrsty_path):
        """Load CUI to TUI mappings from MRSTY.RRF"""
        print("Loading semantic type mappings...")
        for chunk in pd.read_csv(
            mrsty_path, sep='|', header=None, 
            usecols=[0,1],  # Only CUI and TUI
            chunksize=100000,
        ):
            for _, row in chunk.iterrows():
                self.semantic_types_dict[row[0]] = row[1]
        print(f"Loaded {len(self.semantic_types_dict)} CUI-TUI mappings")

    def check_node_count(self):
        """Check current node count in Neo4j database"""
        query = "MATCH (n) RETURN count(n) as node_count"
        result = self.graph.query(query)
        return result[0]["node_count"] if result else 0

    def check_relationship_count(self):
        """Check current relationship count in Neo4j database"""
        query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
        result = self.graph.query(query)
        return result[0]["rel_count"] if result else 0

    def get_concept_tier(self, semantic_type):
        """Determine the tier of a concept based on its semantic type"""
        for tier, concepts in CONCEPT_TIERS.items():
            if semantic_type in concepts:
                return tier
        return "other"

    def get_relationship_tier(self, rel_type):
        """Determine the tier of a relationship based on its type"""
        for tier, rels in RELATIONSHIP_TIERS.items():
            if rel_type in rels:
                return tier
        return "other"

    def should_process_concept(self, semantic_type):
        # Get current node count
        current_node_count = self.check_node_count()
        
        # Get tier for this concept type
        tier = self.get_concept_tier(semantic_type)
        
        # Get step-specific priority if available
        priority_multiplier = 1.0
        
        # For ALL mode, take the maximum priority from any step
        if self.target_step == "ALL":
            if semantic_type in STEP1_PRIORITY:
                priority_multiplier = max(priority_multiplier, STEP1_PRIORITY[semantic_type])
            if semantic_type in STEP2_PRIORITY:
                priority_multiplier = max(priority_multiplier, STEP2_PRIORITY[semantic_type])
            if semantic_type in STEP3_PRIORITY:
                priority_multiplier = max(priority_multiplier, STEP3_PRIORITY[semantic_type])
        # Otherwise use the specific step priority
        elif self.target_step == "STEP1" and semantic_type in STEP1_PRIORITY:
            priority_multiplier = STEP1_PRIORITY[semantic_type]
        elif self.target_step == "STEP2" and semantic_type in STEP2_PRIORITY:
            priority_multiplier = STEP2_PRIORITY[semantic_type]
        elif self.target_step == "STEP3" and semantic_type in STEP3_PRIORITY:
            priority_multiplier = STEP3_PRIORITY[semantic_type]
        
        # For ALL mode, be more generous with capacity thresholds
        tier2_threshold = 0.3
        tier3_threshold = 0.5
        if self.target_step == "ALL":
            tier2_threshold = 0.2  # More generous threshold for tier 2
            tier3_threshold = 0.4  # More generous threshold for tier 3
        
        # Calculate remaining capacity as percentage
        max_nodes = NEO4J_FREE_TIER_LIMITS['max_nodes'] - NEO4J_FREE_TIER_LIMITS['node_buffer']
        remaining_capacity = 1.0 - (current_node_count / max_nodes)
        
        # Decision logic based on tier, priority, and remaining capacity
        if tier == "tier_1":
            # Always process tier 1 concepts
            return True
        elif tier == "tier_2":
            # Process tier 2 if we have enough capacity or it's high priority
            if remaining_capacity > tier2_threshold or priority_multiplier >= 1.5:
                return True
        elif tier == "tier_3":
            # Process tier 3 only if plenty of capacity or very high priority
            if remaining_capacity > tier3_threshold or priority_multiplier >= 1.8:
                return True
        
        # Skip in all other cases
        self.stats["skipped"]["nodes"] += 1
        return False

    def should_process_relationship(self, rel_type):
        # Get current relationship count
        current_rel_count = self.check_relationship_count()
        
        # Get tier for this relationship
        tier = self.get_relationship_tier(rel_type)
        
        # For ALL mode, be more generous with capacity thresholds
        tier2_threshold = 0.3
        tier3_threshold = 0.5
        if self.target_step == "ALL":
            tier2_threshold = 0.2  # More generous threshold for tier 2
            tier3_threshold = 0.4  # More generous threshold for tier 3
        
        # Calculate remaining capacity as percentage
        max_rels = NEO4J_FREE_TIER_LIMITS['max_relationships'] - NEO4J_FREE_TIER_LIMITS['rel_buffer']
        remaining_capacity = 1.0 - (current_rel_count / max_rels)
        
        # Decision logic based on tier and remaining capacity
        if tier == "tier_1":
            # Always process tier 1 relationships
            return True
        elif tier == "tier_2":
            # Process tier 2 if we have enough capacity
            if remaining_capacity > tier2_threshold:
                return True
        elif tier == "tier_3":
            # Process tier 3 only if plenty of capacity
            if remaining_capacity > tier3_threshold:
                return True
        
        # Skip in all other cases
        self.stats["skipped"]["relationships"] += 1
        return False
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, SessionExpired)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying database connection... (attempt {retry_state.attempt_number})"
        )
    )
    def _execute_query_with_retry(self, cypher: str, params: dict = None):
        """Execute Neo4j query with retry logic"""
        try:
            return self.graph.query(cypher, params or {})
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            raise

    def create_indexes(self):
        """Create essential indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.cui)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.term)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Definition) ON (d.def_id)",
            "CREATE INDEX IF NOT EXISTS FOR (st:SemanticType) ON (st.type_id)"
        ]
        
        for index in indexes:
            try:
                self.graph.query(index)
            except Exception as e:
                logger.error(f"Error creating index: {str(e)}")

    def create_vector_index(self):
        """Create vector index for embeddings in Neo4j"""
        try:
            # First check if GDS is available
            check_gds_query = """
            CALL dbms.procedures()
            YIELD name
            WHERE name STARTS WITH 'gds.'
            RETURN count(*) > 0 as gds_available
            """
            
            result = self.graph.query(check_gds_query)
            gds_available = result[0]['gds_available'] if result else False
            
            if not gds_available:
                logger.warning("Graph Data Science library not available. Vector search will not work.")
                return
            
            # Check if index already exists
            check_index_query = """
            SHOW INDEXES
            WHERE type = 'VECTOR' AND name = 'concept_embeddings'
            """
            
            result = self.graph.query(check_index_query)
            if result and len(result) > 0:
                logger.info("Vector index 'concept_embeddings' already exists")
                return
            
            # Get embedding dimension
            test_embedding = self.embedding_model.encode("test")
            embedding_dimension = len(test_embedding)
            
            # Create vector index
            create_index_query = f"""
            CALL db.index.vector.createNodeIndex(
                'concept_embeddings',    // index name
                'Concept',               // node label
                'embedding',             // property name
                {embedding_dimension},   // dimensions
                'cosine'                 // similarity metric
            )
            """
            
            self.graph.query(create_index_query)
            logger.info(f"Created vector index 'concept_embeddings' with dimension {embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            logger.warning("Continuing without vector index. Vector search will not be available.")

    def get_similar_concepts_gds(self, query_term, query_embedding=None, similarity_threshold=0.7, limit=10):
        """
        Find similar concepts using GDS library for similarity calculations
        
        Args:
            query_term: The search term
            query_embedding: Pre-computed embedding vector (optional)
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of similar concepts with similarity scores
        """
        try:
            # Check if GDS is available
            check_gds_query = """
            CALL dbms.procedures()
            YIELD name
            WHERE name STARTS WITH 'gds.'
            RETURN count(*) > 0 as gds_available
            """
            
            result = self.graph.query(check_gds_query)
            gds_available = result[0]['gds_available'] if result else False
            
            if not gds_available:
                logger.warning("GDS not available, falling back to non-GDS similarity")
                return self.get_similar_concepts(query_term, query_embedding, limit)
            
            # Generate embedding if not provided
            if query_embedding is None:
                query_embedding = self.embedding_model.encode(query_term).tolist()
            
            # First try using vector index (fastest approach)
            vector_query = """
            CALL db.index.vector.queryNodes(
                'concept_embeddings',
                $limit,
                $query_vector
            ) YIELD node, score
            RETURN 
                node.cui AS cui,
                node.term AS term,
                node.semantic_type AS semantic_type,
                score AS similarity
            WHERE score >= $threshold
            ORDER BY score DESC
            """
            
            params = {
                "query_vector": query_embedding,
                "limit": limit,
                "threshold": similarity_threshold
            }
            
            result = self.graph.query(vector_query, params)
            
            # If vector index query succeeded, return the results
            if result and len(result) > 0:
                return result
            
            # Fallback to GDS cosine similarity if vector index fails or returns no results
            # This computes similarity on the fly
            gds_query = """
            MATCH (c:Concept)
            WHERE c.embedding IS NOT NULL
            WITH c, gds.similarity.cosine($query_vector, c.embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN 
                c.cui AS cui,
                c.term AS term, 
                c.semantic_type AS semantic_type,
                similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            result = self.graph.query(gds_query, params)
            return result
            
        except Exception as e:
            logger.error(f"Error in GDS similarity search: {str(e)}")
            # Fallback to basic term search
            fallback_query = """
            MATCH (c:Concept)
            WHERE c.term CONTAINS $query_term
            RETURN 
                c.cui AS cui,
                c.term AS term,
                c.semantic_type AS semantic_type,
                1.0 AS similarity
            LIMIT $limit
            """
            return self.graph.query(fallback_query, {"query_term": query_term, "limit": limit})

    def find_similar_concepts_with_paths(self, query_term, max_hops=2, limit=10):
        """
        Find similar concepts and their interconnecting paths using GDS
        
        Args:
            query_term: The search term
            max_hops: Maximum path length (1-3)
            limit: Maximum number of results
            
        Returns:
            Dictionary with similar concepts and connecting paths
        """
        try:
            # Generate embedding for query term
            query_embedding = self.embedding_model.encode(query_term).tolist()
            
            # First find the most similar starting nodes
            similar_nodes = self.get_similar_concepts_gds(
                query_term, 
                query_embedding=query_embedding,
                limit=5  # Start with top 5 similar nodes
            )
            
            if not similar_nodes:
                return {"nodes": [], "paths": []}
            
            # Get CUIs of similar nodes
            similar_cuis = [node["cui"] for node in similar_nodes]
            
            # Using GDS to find paths between similar nodes
            # These paths will represent semantic connections
            paths_query = f"""
            MATCH path = (start:Concept)-[*1..{max_hops}]-(end:Concept)
            WHERE start.cui IN $start_cuis
            AND start <> end
            WITH path, 
                 gds.similarity.cosine(end.embedding, $query_vector) AS end_similarity,
                 gds.similarity.cosine(start.embedding, $query_vector) AS start_similarity
            WHERE end_similarity >= 0.6
            WITH path, 
                 start_similarity + end_similarity AS path_score,
                 end_similarity
            ORDER BY path_score DESC
            LIMIT $limit
            UNWIND nodes(path) AS node
            WITH DISTINCT node, path_score
            RETURN 
                node.cui AS cui,
                node.term AS term,
                node.semantic_type AS semantic_type,
                path_score AS relevance_score
            ORDER BY relevance_score DESC
            LIMIT $limit
            """
            
            path_results = self.graph.query(paths_query, {
                "start_cuis": similar_cuis,
                "query_vector": query_embedding,
                "limit": limit
            })
            
            # Get relationships between these nodes
            if path_results:
                rels_query = """
                MATCH (n:Concept)-[r]-(m:Concept)
                WHERE n.cui IN $node_cuis AND m.cui IN $node_cuis
                RETURN DISTINCT
                    type(r) AS type,
                    startNode(r).cui AS source_cui,
                    endNode(r).cui AS target_cui,
                    startNode(r).term AS source_term,
                    endNode(r).term AS target_term
                LIMIT 100
                """
                
                node_cuis = [node["cui"] for node in path_results]
                rel_results = self.graph.query(rels_query, {"node_cuis": node_cuis})
                
                return {
                    "nodes": path_results,
                    "relationships": rel_results
                }
            
            return {"nodes": similar_nodes, "relationships": []}
            
        except Exception as e:
            logger.error(f"Error in GDS path finding: {str(e)}")
            return {"nodes": [], "relationships": [], "error": str(e)}

    def _process_concept_row(self, row) -> Dict:
        try:
            cui = row[0]       # CUI
            source = row[11]   # SAB column
            term = row[14]     # STR column
            
            # Get TUI from preloaded mappings
            tui = self.semantic_types_dict.get(cui)
            if not tui:
                return None

            # Check if concept exists in USMLE domains
            semantic_type = None
            domain = None  # Initialize domain
            priority = None
            
            # Look through priority levels for current source
            if source in self.usmle_domains:
                for level in ['priority_1', 'priority_2', 'priority_3']:
                    if level in self.usmle_domains[source]:
                        if tui in self.usmle_domains[source][level]:
                            semantic_type = self.usmle_domains[source][level][tui]
                            priority = level
                            domain = semantic_type
                            break
                    if semantic_type:
                        break

            # Skip if not in USMLE domains
            if not semantic_type:
                return None
            
            # Apply tier-based filtering
            if not self.should_process_concept(semantic_type):
                return None
                      
            # Add to stats
            tier = self.get_concept_tier(semantic_type)
            self.stats["nodes"]["total"] += 1
            self.stats["nodes"]["by_tier"][tier] = self.stats["nodes"]["by_tier"].get(tier, 0) + 1
            self.stats["nodes"]["by_type"][semantic_type] = self.stats["nodes"]["by_type"].get(semantic_type, 0) + 1
            
            # Return concept data
            return {
                'cui': cui,
                'term': term,
                'domain': domain,
                'semantic_type': semantic_type,
                'source': source,
                'priority': priority,
                'tier': tier,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error processing concept row: {str(e)}")
            return None

    def _process_mrconso_parallel(self, file_path: str) -> int:
        """Process MRCONSO file using thread-based processing with tiered filtering"""
        try:
            processed = 0
            processed_cuis = self._load_checkpoint()
            skipped = 0
            start_time = time.time()
            
            # Check current node count
            current_node_count = self.check_node_count()
            logger.info(f"Starting with {current_node_count} nodes in database")
            
            # Read data in chunks
            chunks = pd.read_csv(
                file_path,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8'
            )
            
            # Use ThreadPoolExecutor instead of ProcessPool
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for chunk_num, chunk in enumerate(chunks):
                    # Stop if approaching node limit
                    if current_node_count >= (NEO4J_FREE_TIER_LIMITS['max_nodes'] - NEO4J_FREE_TIER_LIMITS['node_buffer']):
                        logger.warning(f"Approaching node limit: {current_node_count}/{NEO4J_FREE_TIER_LIMITS['max_nodes']}")
                        break
                    
                    # Filter English concepts
                    eng_concepts = chunk[chunk[1] == 'ENG']
                    if not eng_concepts.empty:
                        # Process rows using threads
                        futures = [
                            executor.submit(self._process_concept_row, row)
                            for row in eng_concepts.values
                        ]
                        
                        # Collect results
                        valid_concepts = []
                        for future in futures:
                            result = future.result()
                            if result:
                                valid_concepts.append(result)
                        
                        if valid_concepts:
                            # Create nodes in batch
                            self._create_nodes_batch(valid_concepts)
                            processed += len(valid_concepts)
                            current_node_count += len(valid_concepts)
                            
                            # Save checkpoint every 5000 concepts
                            if processed % 5000 == 0:
                                self._save_checkpoint(processed_cuis)
                            
                            # Report progress
                            elapsed = time.time() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            print(f"\rProcessed: {processed:,}/{self.node_limit:,} concepts | "
                                  f"Rate: {rate:.0f} concepts/sec | "
                                  f"Batch: {chunk_num}", end='')
                        
                        skipped += len(eng_concepts) - len(valid_concepts)
            
            # Print final statistics
            print(f"\nCompleted concepts processing: {processed:,} processed, {skipped:,} skipped")
            self.print_stats()
            
            return processed

        except Exception as e:
            logger.error(f"Error in MRCONSO processing: {str(e)}")
            # Save checkpoint on error
            self._save_checkpoint(processed_cuis)
            raise

    def _create_nodes_batch(self, batch: List[Dict], max_retries: int = 3):
        """Create concept nodes in Neo4j using optimized Cypher with retry logic"""
        retries = 0
        while retries < max_retries:
            try:
                # Generate embeddings for terms in batch
                terms = [item['term'] for item in batch]
                
                # Process in smaller batches for embedding generation to avoid memory issues
                embeddings = []
                embedding_batch_size = 300  # Adjust based on your GPU/CPU memory
                
                for i in range(0, len(terms), embedding_batch_size):
                    batch_terms = terms[i:i+embedding_batch_size]
                    batch_embeddings = self.embedding_model.encode(batch_terms).tolist()
                    embeddings.extend(batch_embeddings)
                
                # Add embeddings to batch items
                for i, item in enumerate(batch):
                    item['embedding'] = embeddings[i]
                
                # Create nodes with embeddings
                cypher = """
                UNWIND $batch as item
                MERGE (c:Concept {cui: item.cui})
                ON CREATE SET 
                    c.term = item.term,
                    c.domain = item.domain,
                    c.semantic_type = item.semantic_type,
                    c.source = item.source,
                    c.priority = item.priority,
                    c.tier = item.tier,
                    c.created_at = item.created_at,
                    c.embedding = item.embedding
                """
                self.graph.query(cypher, {'batch': batch})
                return
                
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                logger.warning(f"Batch creation failed (attempt {retries}/{max_retries}). "
                             f"Retrying in {wait_time} seconds... Error: {str(e)}")
                time.sleep(wait_time)
                
        # If we get here, all retries failed
        logger.error(f"Error creating nodes batch after {max_retries} attempts")
        raise

    def process_mrrel(self, file_path: str) -> int:
        """Process MRREL file to create relationships with tiered filtering"""
        try:
            processed = 0
            skipped = 0
            print("\nProcessing relationships...")
            
            # Check current relationship count once at the beginning
            current_rel_count = self.check_relationship_count()
            logger.info(f"Starting with {current_rel_count} relationships in database")
            
            # Pre-calculate relationship limits
            max_rels = NEO4J_FREE_TIER_LIMITS['max_relationships'] - NEO4J_FREE_TIER_LIMITS['rel_buffer']
            
            # Define relationship direction mappings in Python
            # This replaces the complex APOC case statement
            reverse_relationships = set([
                'may_treat', 'may_prevent', 'is_finding_of_disease', 'associated_finding_of',
                'may_be_finding_of_disease', 'associated_etiologic_finding_of',
                'is_not_finding_of_disease', 'clinical_course_of', 'manifestation_of', 
                'location_of', 'is_location_of_biological_process',
                'is_location_of_anatomic_structure', 'mechanism_of_action_of',
                'contraindicated_mechanism_of_action_of', 'cause_of', 
                'pathological_process_of', 'has_contraindicated_drug',
                'finding_site_of', 'direct_procedure_site_of', 'procedure_site_of',
                'may_be_molecular_abnormality_of'
            ])
            
            bidirectional_relationships = set([
                'associated_with', 'related_to', 'gene_associated_with_disease',
                'genetic_biomarker_related_to', 'regulates',
                'negatively_regulates', 'positively_regulates'
            ])
            
            # Increase batch size for more efficient processing
            larger_batch_size = self.batch_size * 3
            
            chunks = pd.read_csv(
                file_path,
                sep='|',
                header=None,
                chunksize=larger_batch_size,
                encoding='utf-8',
                na_filter=False  # Prevent pandas from converting empty strings to NaN
            )
            
            # Process in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for chunk_idx, chunk in enumerate(chunks):
                    # Stop if approaching relationship limit
                    if current_rel_count >= max_rels:
                        logger.warning(f"Approaching relationship limit: {current_rel_count}/{max_rels}")
                        break
                    
                    # Process chunk in parallel
                    futures = []
                    for _, row in chunk.iterrows():
                        futures.append(executor.submit(
                            self._process_relationship_row, 
                            row, 
                            reverse_relationships,
                            bidirectional_relationships
                        ))
                    
                    # Collect valid relationships
                    batch = []
                    for future in futures:
                        result = future.result()
                        if result:
                            batch.append(result)
                            # Update stats
                            self.stats["relationships"]["total"] += 1
                            tier = result['tier']
                            rel_type = result['standardized_rel']
                            self.stats["relationships"]["by_tier"][tier] = self.stats["relationships"]["by_tier"].get(tier, 0) + 1
                            self.stats["relationships"]["by_type"][rel_type] = self.stats["relationships"]["by_type"].get(rel_type, 0) + 1
                        else:
                            skipped += 1
                    
                    if batch:
                        # Create relationships in optimized batch
                        created = self._create_relationships_batch_optimized(batch)
                        processed += created
                        current_rel_count += created
                        
                        # Report progress
                        print(f"\rProcessed {processed:,} relationships, skipped {skipped:,}, current count: {current_rel_count}/{max_rels} (chunk {chunk_idx})", end='')

                print(f"\nCompleted relationship processing: {processed:,} relationships created, {skipped:,} skipped")
                return processed

        except Exception as e:
            logger.error(f"Error in relationships processing: {str(e)}")
            raise

    def _process_relationship_row(self, row, reverse_relationships, bidirectional_relationships):
        """Process a single relationship row with optimized filtering"""
        try:
            # MRREL columns: CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|...
            cui1 = row[0]
            rel_type = row[3]  # REL field
            rel_attr = row[7]  # RELA field
            cui2 = row[4]
            source = row[10]  # SAB field
            
            # Skip if either concept doesn't exist (fast Python check)
            if cui1 not in self.processed_concepts or cui2 not in self.processed_concepts:
                return None
            
            # Map to standardized relationship type
            rel_attr_lower = rel_attr.lower() if rel_attr else ''
            rel_type_lower = rel_type.lower()
            standardized_rel = RELATION_TYPE_MAPPING.get(rel_attr_lower) or RELATION_TYPE_MAPPING.get(rel_type_lower)
            
            # Skip if not in our mapping
            if not standardized_rel:
                return None
            
            # Get tier for this relationship (no database query)
            tier = self.get_relationship_tier(standardized_rel)
            
            # Apply tier-based filtering logic in Python (no database query)
            # This is a simplified version that doesn't query the database each time
            if tier == "tier_1":
                # Always process tier 1 relationships
                pass
            elif tier == "tier_2" or tier == "tier_3":
                # Skip lower tier relationships if we're approaching capacity
                # This is now a static check based on the current count
                current_rel_count = self.stats["relationships"]["total"]
                max_rels = NEO4J_FREE_TIER_LIMITS['max_relationships'] - NEO4J_FREE_TIER_LIMITS['rel_buffer']
                remaining_capacity = 1.0 - (current_rel_count / max_rels)
                
                if (tier == "tier_2" and remaining_capacity < 0.2) or (tier == "tier_3" and remaining_capacity < 0.4):
                    self.stats["skipped"]["relationships"] += 1
                    return None
            else:
                # Skip other tiers
                self.stats["skipped"]["relationships"] += 1
                return None
            
            # Determine relationship direction in Python
            is_reverse = rel_attr_lower in reverse_relationships
            is_bidirectional = rel_attr_lower in bidirectional_relationships
            
            # Return processed relationship data
            return {
                'cui1': cui1,
                'cui2': cui2,
                'rel_type': rel_type,
                'standardized_rel': standardized_rel,
                'tier': tier,
                'source': source,
                'is_reverse': is_reverse,
                'is_bidirectional': is_bidirectional,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.warning(f"Skipping MRREL row due to error: {str(e)}")
            return None

    def _create_relationships_batch_optimized(self, batch: List[Dict]):
        """Create relationships with optimized Cypher (no APOC)"""
        try:
            # Split batch by direction for more efficient processing
            forward_batch = []
            reverse_batch = []
            bidirectional_batch = []
            
            for item in batch:
                if item['is_bidirectional']:
                    bidirectional_batch.append(item)
                elif item['is_reverse']:
                    reverse_batch.append(item)
                else:
                    forward_batch.append(item)
            
            created_count = 0
            
            # Process forward relationships
            if forward_batch:
                # Group by relationship type for more efficient processing
                for rel_type, items in groupby(forward_batch, key=lambda x: x['standardized_rel']):
                    items_list = list(items)
                    
                    # Include the relationship type directly in the query
                    cypher_forward = f"""
                    UNWIND $batch as item
                    MATCH (c1:Concept {{cui: item.cui1}})
                    MATCH (c2:Concept {{cui: item.cui2}})
                    CREATE (c1)-[r:{rel_type} {{
                        source: item.source,
                        tier: item.tier,
                        created_at: item.created_at
                    }}]->(c2)
                    RETURN count(r) as created
                    """
                    
                    result = self.graph.query(cypher_forward, {
                        'batch': items_list
                    })
                    created_count += result[0]['created'] if result else 0
            
            # Process reverse relationships
            if reverse_batch:
                # Group by relationship type
                for rel_type, items in groupby(reverse_batch, key=lambda x: x['standardized_rel']):
                    items_list = list(items)
                    
                    # Include the relationship type directly in the query
                    cypher_reverse = f"""
                    UNWIND $batch as item
                    MATCH (c1:Concept {{cui: item.cui1}})
                    MATCH (c2:Concept {{cui: item.cui2}})
                    CREATE (c2)-[r:{rel_type} {{
                        source: item.source,
                        tier: item.tier,
                        created_at: item.created_at
                    }}]->(c1)
                    RETURN count(r) as created
                    """
                    
                    result = self.graph.query(cypher_reverse, {
                        'batch': items_list
                    })
                    created_count += result[0]['created'] if result else 0
            
            # Process bidirectional relationships (create in both directions)
            if bidirectional_batch:
                # Group by relationship type
                for rel_type, items in groupby(bidirectional_batch, key=lambda x: x['standardized_rel']):
                    items_list = list(items)
                    
                    # Include the relationship type directly in the query
                    cypher_bidirectional = f"""
                    UNWIND $batch as item
                    MATCH (c1:Concept {{cui: item.cui1}})
                    MATCH (c2:Concept {{cui: item.cui2}})
                    CREATE (c1)-[r1:{rel_type} {{
                        source: item.source,
                        tier: item.tier,
                        created_at: item.created_at,
                        bidirectional: true
                    }}]->(c2)
                    CREATE (c2)-[r2:{rel_type} {{
                        source: item.source,
                        tier: item.tier,
                        created_at: item.created_at,
                        bidirectional: true
                    }}]->(c1)
                    RETURN count(r1) + count(r2) as created
                    """
                    
                    result = self.graph.query(cypher_bidirectional, {
                        'batch': items_list
                    })
                    created_count += result[0]['created'] if result else 0
            
            return created_count
            
        except Exception as e:
            logger.error(f"Error creating relationships batch: {str(e)}")
            # Try with smaller batches if we get an error
            if len(batch) > 50:
                mid = len(batch) // 2
                count1 = self._create_relationships_batch_optimized(batch[:mid])
                count2 = self._create_relationships_batch_optimized(batch[mid:])
                return count1 + count2
            raise

    def print_stats(self):
        """Print processing statistics"""
        print("\n=== Knowledge Graph Statistics ===")
        print(f"Total nodes: {self.stats['nodes']['total']}")
        print(f"Total relationships: {self.stats['relationships']['total']}")
        
        print("\nNodes by tier:")
        for tier, count in self.stats["nodes"]["by_tier"].items():
            print(f"  {tier}: {count}")
        
        print("\nRelationships by tier:")
        for tier, count in self.stats["relationships"]["by_tier"].items():
            print(f"  {tier}: {count}")
        
        print("\nTop concept types:")
        sorted_types = sorted(self.stats["nodes"]["by_type"].items(), key=lambda x: x[1], reverse=True)[:10]
        for type_name, count in sorted_types:
            print(f"  {type_name}: {count}")
        
        print("\nTop relationship types:")
        sorted_rels = sorted(self.stats["relationships"]["by_type"].items(), key=lambda x: x[1], reverse=True)[:10]
        for rel_name, count in sorted_rels:
            print(f"  {rel_name}: {count}")
        
        print(f"\nSkipped nodes: {self.stats['skipped']['nodes']}")
        print(f"Skipped relationships: {self.stats['skipped']['relationships']}")

    def add_concept_labels(self):
        """Add specific labels to concepts based on their semantic types"""
        try:
            print("\nAdding labels to concepts based on semantic types...")
            
            cypher = """
            MATCH (c:Concept)-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            WHERE st.type_id IN keys($type_mapping)
            WITH c, st, $type_mapping[st.type_id] as new_label
            CALL apoc.create.addLabels(c, [new_label]) YIELD node
            RETURN COUNT(DISTINCT node) as labeled_concepts
            """
            
            result = self._execute_query_with_retry(cypher, {
                'type_mapping': SEMANTIC_TYPE_TO_LABEL
            })
            
            print(f"Added labels to {result[0]['labeled_concepts']} concepts")
            
            # Log the results
            logger.info(f"Added labels to {result[0]['labeled_concepts']} concepts based on semantic types")
            
            return result[0]['labeled_concepts']
        
        except Exception as e:
            logger.error(f"Error adding concept labels: {str(e)}")
            raise

    def add_embeddings_to_existing_concepts(self, batch_size=500):
        """Add embeddings to existing concepts that don't have them"""
        try:
            # Count concepts without embeddings
            count_query = """
            MATCH (c:Concept)
            WHERE c.term IS NOT NULL AND c.embedding IS NULL
            RETURN count(c) as count
            """
            
            result = self.graph.query(count_query)
            total_concepts = result[0]['count'] if result else 0
            
            if total_concepts == 0:
                logger.info("No concepts found without embeddings")
                return 0
            
            logger.info(f"Adding embeddings to {total_concepts} existing concepts")
            processed = 0
            start_time = time.time()
            
            while processed < total_concepts:
                # Get batch of concepts without embeddings
                batch_query = """
                MATCH (c:Concept)
                WHERE c.term IS NOT NULL AND IS NULL(c.embedding)
                RETURN c.cui as cui, c.term as term
                LIMIT $batch_size
                """
                
                batch = self.graph.query(batch_query, {'batch_size': batch_size})
                
                if not batch:
                    break
                
                # Generate embeddings
                terms = [record['term'] for record in batch]
                embeddings = self.embedding_model.encode(terms).tolist()
                
                # Update concepts with embeddings
                updates = []
                for i, record in enumerate(batch):
                    updates.append({
                        'cui': record['cui'],
                        'embedding': embeddings[i]
                    })
                
                # Update batch
                update_query = """
                UNWIND $updates as item
                MATCH (c:Concept {cui: item.cui})
                SET c.embedding = item.embedding
                """
                
                self.graph.query(update_query, {'updates': updates})
                
                # Update progress
                processed += len(batch)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                
                print(f"\rProcessed: {processed}/{total_concepts} concepts | "
                      f"Rate: {rate:.0f} concepts/sec | "
                      f"{(processed/total_concepts)*100:.1f}% complete", end='')
            
            print(f"\nCompleted embedding generation for {processed} concepts")
            return processed
            
        except Exception as e:
            logger.error(f"Error adding embeddings to existing concepts: {str(e)}")
            raise

    def _save_checkpoint(self, processed_cuis: set):
        """Save processing checkpoint"""
        try:
            with open('processing_checkpoint.txt', 'w') as f:
                f.write('\n'.join(processed_cuis))
            logger.info(f"Saved checkpoint with {len(processed_cuis)} concepts")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def _load_checkpoint(self) -> set:
        """Load processing checkpoint"""
        try:
            if os.path.exists('processing_checkpoint.txt'):
                with open('processing_checkpoint.txt', 'r') as f:
                    cuis = set(f.read().splitlines())
                logger.info(f"Loaded checkpoint with {len(cuis)} concepts")
                return cuis
            return set()
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return set()

    def process_mrsty(self, file_path: str) -> int:
        """Process MRSTY file using thread-based processing with tier filtering"""
        try:
            processed = 0
            skipped = 0
            start_time = time.time()
            
            print("\nProcessing semantic types...")
            
            chunks = pd.read_csv(
                file_path,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8'
            )
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for chunk_num, chunk in enumerate(chunks):
                    futures = [
                        executor.submit(self._process_sty_row, row)
                        for row in chunk.values
                    ]
                    
                    valid_types = []
                    for future in futures:
                        result = future.result()
                        if result:
                            valid_types.append(result)
                    
                    if valid_types:
                        self._create_semantic_types_batch(valid_types)
                        processed += len(valid_types)
                        
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"\rProcessed: {processed:,} semantic types | "
                              f"Rate: {rate:.0f} types/sec | "
                              f"Batch: {chunk_num}", end='')
                    
                    skipped += len(chunk) - len(valid_types)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in MRSTY processing: {str(e)}")
            raise

    def _process_sty_row(self, row) -> Dict:
        """Process a single semantic type row with tier filtering"""
        try:
            cui = row[0]              # CUI
            semantic_type_id = row[1]  # TUI
            
            # Check if concept exists
            if cui in self.processed_concepts:
                # Get semantic type name from SEMANTIC_TYPES if available
                from src.config.constants import SEMANTIC_TYPES
                semantic_type_name = SEMANTIC_TYPES.get(semantic_type_id, row[3])
                
                # Determine tier for this semantic type
                tier = "other"
                for t, types in CONCEPT_TIERS.items():
                    if semantic_type_name in types:
                        tier = t
                        break
                
                return {
                    'cui': cui,                  # Concept CUI
                    'type_id': semantic_type_id, # Type ID
                    'semantic_type': semantic_type_name,
                    'name': row[3],             # STY
                    'tree_number': row[2],      # STN
                    'tier': tier,
                    'source': 'UMLS',
                    'status': 'active',
                    'version': '2024.1',
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            return None
            
        except Exception as e:
            logger.error(f"Error processing semantic type row: {str(e)}")
            return None

    def _create_semantic_types_batch(self, batch: List[Dict]):
        """Create semantic type nodes and relationships in Neo4j"""
        try:
            # First create semantic type nodes with all fields
            cypher_nodes = """
            UNWIND $batch as item
            MERGE (st:SemanticType {type_id: item.type_id})
            ON CREATE SET 
                st.semantic_type = item.semantic_type,
                st.name = item.name,
                st.tree_number = item.tree_number,
                st.tier = item.tier,
                st.source = item.source,
                st.status = item.status,
                st.version = item.version,
                st.created_at = item.created_at
            """
            self.graph.query(cypher_nodes, {'batch': batch})
            
            # Then create relationships and update concepts
            cypher_rels = """
            UNWIND $batch as item
            MATCH (c:Concept {cui: item.cui})
            MATCH (st:SemanticType {type_id: item.type_id})
            MERGE (c)-[r:HAS_SEMANTIC_TYPE]->(st)
            ON CREATE SET 
                r.created_at = item.created_at,
                r.source = item.source
            SET c.semantic_type = item.semantic_type,
                c.semantic_type_id = item.type_id
            """
            self.graph.query(cypher_rels, {'batch': batch})
            
        except Exception as e:
            logger.error(f"Error creating semantic types batch: {str(e)}")
            
            # Try processing in smaller batches if large batch fails
            if len(batch) > 100:
                mid = len(batch) // 2
                self._create_semantic_types_batch(batch[:mid])
                self._create_semantic_types_batch(batch[mid:])
            else:
                raise

    def _process_mrdef_parallel(self, file_path: str) -> int:
        """Process MRDEF file using thread-based processing"""
        try:
            processed = 0
            skipped = 0
            start_time = time.time()
            
            print("\nProcessing definitions...")
            
            chunks = pd.read_csv(
                file_path,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8'
            )
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for chunk_num, chunk in enumerate(chunks):
                    futures = [
                        executor.submit(self._process_def_row, row)
                        for row in chunk.values
                    ]
                    
                    valid_defs = []
                    for future in futures:
                        result = future.result()
                        if result:
                            valid_defs.append(result)
                    
                    if valid_defs:
                        self._create_definitions_batch(valid_defs)
                        processed += len(valid_defs)
                        
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"\rProcessed: {processed:,} definitions | "
                              f"Rate: {rate:.0f} defs/sec | "
                              f"Batch: {chunk_num}", end='')
                    
                    skipped += len(chunk) - len(valid_defs)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in MRDEF processing: {str(e)}")
            raise

    def _process_def_row(self, row) -> Dict:
        """Process a single definition row"""
        try:
            cui = row[0]      # CUI
            source = row[4]   # SAB
            text = row[5]     # DEF
            suppress = row[6]  # SUPPRESS
            
            # Check if concept exists and definition is not suppressed
            if (cui in self.processed_concepts and 
                suppress != 'Y' and 
                pd.notna(text)):
                
                return {
                    'cui': cui,
                    'def_id': f"DEF_{cui}_{source}",
                    'text': text.strip(),
                    'source': source,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            return None
            
        except Exception as e:
            logger.error(f"Error processing definition row: {str(e)}")
            return None

    def _create_definitions_batch(self, batch: List[Dict]):
        """Create definition nodes and relationships in Neo4j"""
        try:
            # First create definition nodes
            cypher_nodes = """
            UNWIND $batch as def
            MERGE (d:Definition {def_id: def.def_id})
            ON CREATE SET 
                d.text = def.text,
                d.source = def.source,
                d.created_at = def.created_at
            """
            self.graph.query(cypher_nodes, {'batch': batch})
            
            # Then create relationships
            cypher_rels = """
            UNWIND $batch as def
            MATCH (c:Concept {cui: def.cui})
            MATCH (d:Definition {def_id: def.def_id})
            MERGE (c)-[r:HAS_DEFINITION]->(d)
            ON CREATE SET 
                r.created_at = def.created_at
            """
            self.graph.query(cypher_rels, {'batch': batch})
            
        except Exception as e:
            logger.error(f"Error creating definitions batch: {str(e)}")
            
            # Try processing in smaller batches if large batch fails
            if len(batch) > 100:
                mid = len(batch) // 2
                self._create_definitions_batch(batch[:mid])
                self._create_definitions_batch(batch[mid:])
            else:
                raise


    def cleanup_duplicate_relationships(self):
        """Clean up duplicate relationships in the database"""
        try:
            logger.info("Starting cleanup of duplicate relationships...")
            
            # First, get a count of duplicates
            count_query = """
            MATCH (n)-[r]->(m)
            WITH n, m, type(r) as relType, count(r) as relCount
            WHERE relCount > 1
            RETURN sum(relCount - 1) as totalDuplicates
            """
            
            result = self._execute_query_with_retry(count_query)
            duplicate_count = result[0]['totalDuplicates'] if result else 0
            
            if duplicate_count == 0:
                logger.info("No duplicate relationships found.")
                return 0
                
            logger.info(f"Found {duplicate_count} duplicate relationships to clean up")
            
            # Clean up duplicates for each relationship type
            cleanup_query = """
            MATCH (n)-[r]->(m)
            WITH n, m, type(r) as relType, collect(r) as rels
            WHERE size(rels) > 1
            WITH n, m, relType, rels[0] as keepRel, rels[1..] as duplicateRels
            FOREACH (rel IN duplicateRels | DELETE rel)
            RETURN count(duplicateRels) as deletedCount
            """
            
            result = self._execute_query_with_retry(cleanup_query)
            deleted_count = result[0]['deletedCount'] if result else 0
            
            logger.info(f"Successfully deleted {deleted_count} duplicate relationships")
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up duplicate relationships: {str(e)}")
            raise



    def process_dataset(self, files):
        """
        Process the entire dataset.
        
        Args:
            files: Dictionary of file paths
            
        Returns:
            Processing results
        """
        logger.info("Processing dataset")
        
        # Create indexes
        self.create_indexes()
        self.create_vector_index()
        
        # Process concepts
        concepts_processed = self._process_mrconso_parallel(files['mrconso'])
        
        # Process semantic types
        semantic_types_processed = self.process_mrsty(files['mrsty'])
        
        # Process relationships
        relationships_processed = self.process_mrrel(files['mrrel'])
        
        # Process definitions
        definitions_processed = self._process_mrdef_parallel(files['mrdef'])
        
        # Add labels based on semantic types
        self.add_concept_labels()
        
        return {
            "concepts_processed": concepts_processed,
            "semantic_types_processed": semantic_types_processed,
            "relationships_processed": relationships_processed,
            "definitions_processed": definitions_processed
        }
