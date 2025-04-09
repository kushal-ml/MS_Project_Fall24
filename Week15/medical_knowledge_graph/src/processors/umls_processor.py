import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
from typing import List, Dict, Set
import logging
from datetime import datetime
from src.processors.base_processor import BaseProcessor, DatabaseMixin
from src.config.constants import IMPORTANT_RELATIONS, USMLE_DOMAINS, IMPORTANT_SEMANTIC_TYPE, SEMANTIC_TYPE_TO_LABEL, RELATION_TYPE_MAPPING,  HIER_TYPE_MAPPING
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class STEncoder(Embeddings):
    """Sentence Transformer encoder for medical concepts"""
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        """Embed documents/texts"""
        return self.model.embed_documents(texts)
    
    def embed_query(self, text):
        """Embed a query"""
        return self.model.embed_query(text)

class UMLSProcessor(BaseProcessor, DatabaseMixin):
    def __init__(self, graph, max_retries=3, min_wait=1, max_wait=10):
        super().__init__(graph)
        self.batch_size = 10000
        self.num_workers = max(1, cpu_count() - 1)
        self.node_limit = 50000
        self.relationship_limit = 75000
        self.processed_concepts = set()
        
        # Include both priority 1 and 2 for semantic types
        self.important_semantic_types = {
            **IMPORTANT_SEMANTIC_TYPE.get('priority_1', {}),
            **IMPORTANT_SEMANTIC_TYPE.get('priority_2', {})
        }
        
        # Include both priority 1 and 2 for relations
        self.important_relations = {
            **IMPORTANT_RELATIONS.get('priority_1', {}),
            **IMPORTANT_RELATIONS.get('priority_2', {})
        }
        
        # Include both priority 1 and 2 for USMLE domains
        self.usmle_domains = {
            k: {
                **v.get('priority_1', {}),
                **v.get('priority_2', {})
            }
            for k, v in USMLE_DOMAINS.items()
        }
        
        print(f"Loaded {len(self.important_semantic_types)} semantic types")
        print(f"Loaded {len(self.important_relations)} relations")
        print(f"Loaded {len(self.usmle_domains)} USMLE domains")
        
        self._load_existing_data()
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait

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

    def _load_existing_data(self):
        """Load existing concepts, relationships, and semantic types from the graph"""
        try:
            # Load existing concepts
            cypher_concepts = """
            MATCH (c:Concept) 
            RETURN c.code as code, c.source as source, c.priority as priority
            """
            existing_concepts = self.graph.query(cypher_concepts)
            self.existing_concepts = {
                (row['code'], row['source'], row['priority']) 
                for row in existing_concepts
            }

            # Load existing relationships
            cypher_rels = """
            MATCH (c1:Concept)-[r]->(c2:Concept)
            RETURN type(r) as rel_type, c1.code as source_code, c2.code as target_code
            """
            existing_rels = self.graph.query(cypher_rels)
            self.existing_relationships = {
                (row['rel_type'], row['source_code'], row['target_code']) 
                for row in existing_rels
            }

            # Load existing semantic types
            cypher_sem = """
            MATCH (c:Concept)-[:HAS_SEMANTIC_TYPE]->(s:SemanticType)
            RETURN c.code as code, s.category as category
            """
            existing_sem = self.graph.query(cypher_sem)
            self.existing_semantic_types = {
                (row['code'], row['category']) 
                for row in existing_sem
            }

            logger.info(f"Loaded {len(self.existing_concepts)} existing concepts, "
                       f"{len(self.existing_relationships)} relationships, "
                       f"{len(self.existing_semantic_types)} semantic type mappings")

        except Exception as e:
            logger.error(f"Error loading existing data: {str(e)}")
            raise

    def validate_data(self, data: Dict) -> bool:
        """Validate UMLS data files"""
        try:
            # Check required files exist
            required_files = ['mrconso', 'mrrel', 'mrsty', 'mrdef']
            for file in required_files:
                if file not in data:
                    logger.error(f"Missing required file: {file}")
                    return False
                if not os.path.exists(data[file]):
                    logger.error(f"File not found: {data[file]}")
                    return False
            
            # Validate file formats
            for file_name, file_path in data.items():
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line or '|' not in first_line:
                        logger.error(f"Invalid file format for {file_name}: {file_path}")
                        return False
            
            logger.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return False

    def preprocess_data(self, data: Dict) -> Dict:
        """Preprocess UMLS data files"""
        try:
            # Validate files first
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Create indexes if needed
            self.create_indexes()
            
            logger.info("Data preprocessing completed successfully")
            return data

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def process_dataset(self, data: Dict):
        """Process UMLS dataset using parallel processing"""
        try:
            print(f"\n=== Starting Parallel Processing (CPUs: {self.num_workers}) ===")
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Process MRCONSO first
                mrconso_future = executor.submit(
                    self._process_mrconso_parallel, 
                    data['mrconso']
                )
                
                # Wait for concepts to be processed
                concepts_processed = mrconso_future.result()
                print(f"✓ Processed {concepts_processed:,} concepts")
                
                # Load processed concepts for filtering
                self._load_processed_concepts()
                
                # Process remaining files in parallel
                futures = {
       
                    'semantic_types': executor.submit(
                        self.process_mrsty, 
                        data['mrsty']
                    ),
                    'definitions': executor.submit(
                        self._process_mrdef_parallel, 
                        data['mrdef']
                    ),
                    'relationships': executor.submit(
                        self.process_mrrel, 
                        data['mrrel']
                    ),
                }
                
                # Collect results
                results = {}
                for name, future in futures.items():
                    try:
                        results[name] = future.result()
                        print(f"✓ Completed {name} processing")
                    except Exception as e:
                        logger.error(f"Error processing {name}: {str(e)}")
                        results[name] = 0
            
            total_time = time.time() - start_time
            print(f"\n=== Processing Complete ({total_time:.1f}s) ===")
            print(f"Concepts: {concepts_processed:,}")
            print(f"Relationships: {results.get('relationships', 0):,}")
            print(f"Semantic Types: {results.get('semantic_types', 0):,}")
            print(f"Definitions: {results.get('definitions', 0):,}")
            print(f"Hierarchical Relationships: {results.get('hierarchies', 0):,}")
            
            # # Verify hierarchical relationships
            # print("\nVerifying hierarchical relationships...")
            # self.verify_hierarchy_relationships()
            
        except Exception as e:
            logger.error(f"Error in dataset processing: {str(e)}")
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

    def _process_mrconso_parallel(self, file_path: str) -> int:
        """Process MRCONSO file using thread-based processing instead of multiprocessing"""
        try:
            processed = 0
            processed_cuis = self._load_checkpoint()
            skipped = 0
            start_time = time.time()
            
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
                    if processed >= self.node_limit:
                        print(f"\n✓ Node limit reached: {processed:,}/{self.node_limit:,}")
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
            
            return processed

        except Exception as e:
            logger.error(f"Error in MRCONSO processing: {str(e)}")
            # Save checkpoint on error
            self._save_checkpoint(processed_cuis)
            raise

    def _process_concept_row(self, row) -> Dict:
        """Process a single concept row"""
        try:
            source = row[11]  # SAB column
            tty = row[12]     # TTY column
            
            if (source in self.usmle_domains and 
                tty in self.usmle_domains[source]):
                return {
                    'cui': row[0],      # CUI
                    'term': row[14],    # STR
                    'domain': self.usmle_domains[source][tty],
                    'source': source,
                    'priority': 'priority_1',
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            return None
        except Exception as e:
            logger.error(f"Error processing concept row: {str(e)}")
            return None

    def _create_nodes_batch(self, batch: List[Dict], max_retries: int = 3):
        """Create concept nodes in Neo4j using optimized Cypher with retry logic"""
        retries = 0
        
        # Initialize the embedding model if not already done
        if not hasattr(self, 'embedding_model'):
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            logger.info(f"Initializing embedding model: {model_name}")
            
            # Determine if CUDA is available for GPU acceleration
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.embedding_model = STEncoder(model)
        
        # Generate embeddings for all terms in the batch
        terms = [item['term'] for item in batch if 'term' in item and item['term']]
        if terms:
            try:
                embeddings = self.embedding_model.embed_documents(terms)
                
                # Add embeddings to the batch items
                embedding_idx = 0
                for item in batch:
                    if 'term' in item and item['term']:
                        item['embedding'] = embeddings[embedding_idx]
                        embedding_idx += 1
            except Exception as e:
                logger.warning(f"Error generating embeddings: {str(e)}")
                # Continue without embeddings if there's an error
        
        while retries < max_retries:
            try:
                cypher = """
                UNWIND $batch as item
                MERGE (c:Concept {cui: item.cui})
                ON CREATE SET 
                    c.term = item.term,
                    c.domain = item.domain,
                    c.source = item.source,
                    c.priority = item.priority,
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

    def _load_processed_concepts(self):
        """Load processed concepts from Neo4j for filtering"""
        try:
            cypher = """
            MATCH (c:Concept)
            RETURN c.cui as cui
            """
            results = self.graph.query(cypher)
            self.processed_concepts = {row['cui'] for row in results}
            print(f"Loaded {len(self.processed_concepts):,} processed concepts")
            
        except Exception as e:
            logger.error(f"Error loading processed concepts: {str(e)}")
            raise

    def process_mrrel(self, file_path: str) -> int:
        """Process MRREL file to create relationships"""
        try:
            processed = 0
            print("\nProcessing relationships...")
            
            chunks = pd.read_csv(
                file_path,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8',
                na_filter=False  # Prevent pandas from converting empty strings to NaN
            )
            
            for chunk in chunks:
                batch = []
                for _, row in chunk.iterrows():
                    try:
                        # MRREL columns: CUI1|AUI1|RELA|REL|CUI2|AUI2|RELA|REL_ATTR|...
                        cui1 = row[0]
                        rel_type = row[3]  # REL field
                        rel_attr = row[7]  # REL_ATTR field - important for treats/cause_of
                        cui2 = row[4]
                        source = row[10]  # Adjust index based on your file structure
                        
                        if cui1 in self.processed_concepts and cui2 in self.processed_concepts:
                            batch.append({
                                'cui1': cui1,
                                'cui2': cui2,
                                'rel_type': rel_type,
                                'rel_attr': rel_attr,
                                'source': source,
                                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                    except Exception as e:
                        logger.warning(f"Skipping MRREL row due to error: {str(e)}")
                        continue

                if batch:
                    self._create_relationships_batch(batch)
                    processed += len(batch)
                    print(f"\rProcessed {processed:,} relationships", end='')

            print(f"\nCompleted relationship processing: {processed:,} relationships created")
            return processed

        except Exception as e:
            logger.error(f"Error in relationships processing: {str(e)}")
            raise

    def _create_relationships_batch(self, batch: List[Dict]):
        """Create specific clinical relationships with correct directions"""
        try:
            cypher = """
            UNWIND $batch as item
            MATCH (c1:Concept {cui: item.cui1})
            MATCH (c2:Concept {cui: item.cui2})
            WITH c1, c2, item
            CALL apoc.do.case(
                [
                    item.rel_attr IN ['may_treat', 'may_prevent',
                        'is_finding_of_disease', 'associated_finding_of',
                        'may_be_finding_of_disease', 'associated_etiologic_finding_of',
                        'is_not_finding_of_disease', 'clinical_course_of',
                        'manifestation_of', 'location_of',
                        'is_location_of_biological_process',
                        'is_location_of_anatomic_structure',
                        'mechanism_of_action_of',
                        'contraindicated_mechanism_of_action_of',
                        'cause_of'],
                    "RETURN {type: $rel_mapping[item.rel_attr], reverse: true} AS rel_info",
                    
                    item.rel_attr IN ['may_be_treated_by',
                        'disease_has_accepted_treatment_with_regimen',
                        'disease_has_finding', 'disease_may_have_finding',
                        'has_causative_agent', 'occurs_in', 'has_location',
                        'has_ingredient', 'has_precise_ingredient',
                        'has_mechanism_of_action',
                        'chemical_or_drug_has_mechanism_of_action',
                        'chemical_or_drug_affects_gene_product',
                        'contraindicated_with_disease', 'occurs_before',
                        'regulates', 'negatively_regulates',
                        'positively_regulates', 'develops_into',
                        'has_course', 'part_of', 'drains_into',
                        'may_be_diagnosed_by'],
                    "RETURN {type: $rel_mapping[item.rel_attr], reverse: false} AS rel_info",
                    
                    item.rel_attr IN ['associated_with',
                        'related_to', 'gene_associated_with_disease'],
                    "RETURN {type: $rel_mapping[item.rel_attr], reverse: false} AS rel_info"
                ],
                "RETURN {type: $rel_mapping[item.rel_attr], reverse: false} AS rel_info",  // elseQuery as a string
                {c1: c1, c2: c2, item: item, rel_mapping: $rel_mapping}  // parameters
            ) YIELD value
            WITH c1, c2, item, value.rel_info AS rel_info
            WHERE rel_info.type IS NOT NULL
            CALL apoc.merge.relationship(
                CASE WHEN rel_info.reverse THEN c2 ELSE c1 END,
                rel_info.type,
                {
                    source: item.source,
                    rel_attr: item.rel_attr,
                    rel_type: item.rel_type,
                    created_at: item.created_at
                },
                {},
                CASE WHEN rel_info.reverse THEN c1 ELSE c2 END
            )
            YIELD rel
            RETURN count(rel) as created
            """
            
            result = self.graph.query(cypher, {
                'batch': batch,
                'rel_mapping': RELATION_TYPE_MAPPING
            })
            created = result[0]['created'] if result else 0
            
            return created
            
        except Exception as e:
            logger.error(f"Error creating relationships batch: {str(e)}")
            raise

    def process_mrsty(self, file_path: str) -> int:
        """Process MRSTY file using thread-based processing"""
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
        """Process a single semantic type row"""
        try:
            cui = row[0]              # CUI
            semantic_type_id = row[1]  # TUI
            
            # Check if concept exists and semantic type is important
            if (cui in self.processed_concepts and 
                semantic_type_id in self.important_semantic_types):
                
                category = 'priority_1' if semantic_type_id in IMPORTANT_SEMANTIC_TYPE['priority_1'] else 'priority_2'
                
                return {
                    'cui': cui,                  # Concept CUI
                    'type_id': semantic_type_id, # Aligned with new naming
                    'semantic_type': self.important_semantic_types[semantic_type_id],
                    'name': row[3],             # STY
                    'tree_number': row[2],      # STN
                    'category': category,
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
                st.category = item.category,
                st.source = item.source,
                st.status = item.status,
                st.description = item.description,
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

    def create_semantic_type_hierarchy(self):
        """Create hierarchical relationships between semantic types"""
        try:
            cypher = """
            MATCH (st1:SemanticType), (st2:SemanticType)
            WHERE st1.tree_number STARTS WITH st2.tree_number
            AND st1.tree_number <> st2.tree_number
            MERGE (st1)-[r:IS_A]->(st2)
            RETURN count(r) as hierarchy_count
            """
            result = self.graph.query(cypher)
            count = result[0]['hierarchy_count'] if result else 0
            logger.info(f"Created {count} semantic type hierarchical relationships")

        except Exception as e:
            logger.error(f"Error creating semantic type hierarchy: {str(e)}")

    def get_semantic_type_stats(self) -> Dict:
        """Get statistics about semantic types"""
        try:
            cypher = """
            MATCH (st:SemanticType)
            WITH count(st) as total_types
            MATCH (c:Concept)-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            RETURN 
                total_types,
                count(DISTINCT c) as concepts_with_types,
                count(DISTINCT st) as types_in_use
            """
            result = self.graph.query(cypher)
            return dict(result[0]) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting semantic type stats: {str(e)}")
            return {}
        
    
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

    def get_concept_definitions(self, cui: str) -> List[Dict]:
        """Get all definitions for a concept"""
        try:
            cypher = """
            MATCH (c:Concept {cui: $cui})-[r:HAS_DEFINITION]->(d:Definition)
            RETURN d.def_id as def_id,
                   d.text as text,
                   d.source as source,
                   d.created_at as created_at
            ORDER BY d.created_at DESC
            """
            result = self.graph.query(cypher, {'cui': cui})
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            raise
    def get_definition_stats(self) -> Dict:
        """Get statistics about definitions"""
        try:
            cypher = """
            MATCH (d:Definition)
            WITH count(d) as total_defs
            MATCH (c:Concept)-[:HAS_DEFINITION]->(d:Definition)
            RETURN total_defs,
                   count(DISTINCT c) as concepts_with_defs,
                   avg(size(d.text)) as avg_def_length
            """
            result = self.graph.query(cypher)
            return dict(result[0]) if result else {}
                    
        except Exception as e:
            logger.error(f"Error getting definition stats: {str(e)}")
            return {}
        

    def get_concept_summary(self, cui: str) -> Dict:
        """Get comprehensive summary of a concept"""
        try:
            cypher = """
            MATCH (c:Concept {cui: $cui})
            OPTIONAL MATCH (c)-[:HAS_DEFINITION]->(d:Definition)
            OPTIONAL MATCH (c)-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            OPTIONAL MATCH (c)-[r]->(related:Concept)
            RETURN c.cui as cui,
                   c.term as term,
                   c.domain as domain,
                   c.source as source,
                   collect(DISTINCT d.text) as definitions,
                   collect(DISTINCT st.name) as semantic_types,
                   count(DISTINCT r) as relationship_count,
                   collect(DISTINCT {
                       term: related.term,
                       type: type(r)
                   }) as related_concepts
            """
            result = self.graph.query(cypher, {'cui': cui})
            return dict(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting concept summary: {str(e)}")
            return None

    def search_concepts(self, term: str, domain: str = None, semantic_type: str = None, limit: int = 10) -> List[Dict]:
        """Search concepts with filters"""
        try:
            cypher = """
            MATCH (c:Concept)
            WHERE c.term =~ $term_pattern
            AND ($domain IS NULL OR c.domain = $domain)
            AND ($semantic_type IS NULL OR c.semantic_type = $semantic_type)
            RETURN c.cui as cui,
                   c.term as term,
                   c.domain as domain,
                   c.semantic_type as semantic_type
            ORDER BY size(c.term)
            LIMIT $limit
            """
            params = {
                'term_pattern': f'(?i).*{term}.*',
                'domain': domain,
                'semantic_type': semantic_type,
                'limit': limit
            }
            result = self.graph.query(cypher, params)
            return [dict(record) for record in result]
        
        except Exception as e:
            logger.error(f"Error searching concepts: {str(e)}")
            return []

    def find_shortest_path(self, start_cui: str, end_cui: str, max_depth: int = 3) -> List[Dict]:
        """Find shortest path between two concepts"""
        try:
            cypher = """
            MATCH path = shortestPath((start:Concept {cui: $start_cui})-[*..%d]->(end:Concept {cui: $end_cui}))
            RETURN [node in nodes(path) | node.name] as concepts,
                   [type(r) in relationships(path)] as relationships,
                   length(path) as path_length
            """ % max_depth
            
            result = self.graph.query(cypher, {
                'start_cui': start_cui,
                'end_cui': end_cui
            })
            
            if not result:
                return []
            
            return [{
                'concepts': row['concepts'],
                'relationships': row['relationships'],
                'path_length': row['path_length']
            } for row in result]
            
        except Exception as e:
            logger.error(f"Error finding shortest path: {str(e)}")
            return []

    def get_domain_summary(self, domain: str = None) -> Dict:
        """Get summary statistics for a domain"""
        try:
            cypher = """
            MATCH (c:Concept)
            WHERE $domain IS NULL OR c.domain = $domain
            WITH collect(c) as concepts
            RETURN size(concepts) as concept_count,
                   count(DISTINCT c.semantic_type) as semantic_type_count,
                   count(DISTINCT c.source) as source_count,
                   avg(size((c)-[:HAS_DEFINITION]->(:Definition))) as avg_definitions,
                   avg(size((c)-[]->(:Concept))) as avg_relationships
            """
            result = self.graph.query(cypher, {'domain': domain})
            return dict(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting domain summary: {str(e)}")
            return None

    def export_subgraph(self, cui: str, depth: int = 2) -> Dict:
        """Export a subgraph around a concept"""
        try:
            cypher = """
            MATCH path = (c:Concept {cui: $cui})-[*..${depth}]-(related)
            WITH collect(path) as paths
            RETURN {
                nodes: [node in nodes(paths) | {
                    cui: node.cui,
                    term: node.term,
                    type: labels(node)[0]
                }],
                relationships: [rel in relationships(paths) | {
                    source: startNode(rel).cui,
                    target: endNode(rel).cui,
                    type: type(rel)
                }]
            } as graph
            """
            result = self.graph.query(cypher, {'cui': cui, 'depth': depth})
            return dict(result[0]['graph']) if result else None
        except Exception as e:
            logger.error(f"Error exporting subgraph: {str(e)}")
            return None

    def validate_data_quality(self) -> Dict:
        """Validate data quality and return report"""
        try:
            report = {
                'missing_properties': 0,
                'orphaned_nodes': 0,
                'invalid_relationships': 0,
                'duplicate_definitions': 0,
                'issues': []
            }
            
            # Check for missing required properties
            cypher = """
            MATCH (c:Concept)
            WHERE NOT EXISTS(c.cui) OR NOT EXISTS(c.term)
            RETURN count(c) as count
            """
            result = self.graph.query(cypher)
            report['missing_properties'] = result[0]['count'] if result else 0
            
            # Check for orphaned nodes
            cypher = """
            MATCH (c:Concept)
            WHERE NOT (c)--()
            RETURN count(c) as count
            """
            result = self.graph.query(cypher)
            report['orphaned_nodes'] = result[0]['count'] if result else 0
            
            # Check for duplicate definitions
            cypher = """
            MATCH (c:Concept)-[:HAS_DEFINITION]->(d:Definition)
            WITH c, d.text as text, count(*) as count
            WHERE count > 1
            RETURN sum(count) as total
            """
            result = self.graph.query(cypher)
            report['duplicate_definitions'] = result[0]['total'] if result else 0
            
            return report
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return None

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
        
    def get_semantic_types_for_concept(self, cui: str) -> List[Dict]:
        """Retrieve semantic types for a given concept CUI"""
        try:
            cypher = """
            MATCH (c:Concept {cui: $cui})-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            RETURN st.type_id as type_id, st.name as semantic_type
            """
            result = self.graph.query(cypher, {'cui': cui})
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error retrieving semantic types for CUI {cui}: {str(e)}")
            return []
        

    def get_definitions_for_concept(self, cui: str) -> List[Dict]:
        """Retrieve definitions for a given concept CUI"""
        try:
            cypher = """
            MATCH (c:Concept {cui: $cui})-[:HAS_DEFINITION]->(d:Definition)
            RETURN d.text as text
            """
            result = self.graph.query(cypher, {'cui': cui})
            return [{'text': record['text']} for record in result]
        except Exception as e:
            logger.error(f"Error retrieving definitions for CUI {cui}: {str(e)}")
            return []

    def get_synonyms_for_concept(self, cui: str) -> List[str]:
        """Retrieve synonyms for a given concept CUI"""
        try:
            cypher = """
            MATCH (c:Concept {cui: $cui})-[:HAS_SYNONYM]->(s:Synonym)
            RETURN s.name as synonym
            """
            result = self.graph.query(cypher, {'cui': cui})
            return [record['synonym'] for record in result]
        except Exception as e:
            logger.error(f"Error retrieving synonyms for CUI {cui}: {str(e)}")
            return []

    def _process_new_concepts_parallel(self):
        """Process and load new concepts using parallel processing"""
        try:
            new_concepts = []
            processed = 0
            
            # Collect only new concepts that don't exist in the graph
            for source, priorities in USMLE_DOMAINS.items():
                for priority, concepts in priorities.items():
                    batch = [
                        {
                            'cui': code,         # Using code as CUI
                            'term': name,        # Term from concepts
                            'domain': concepts[code],  # Domain from concepts
                            'source': source,
                            'priority': priority,
                            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        for code, name in concepts.items()
                        if (code, source, priority) not in self.existing_concepts
                    ]
                    new_concepts.extend(batch)

            # Load new concepts in batches
            if new_concepts:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    for i in range(0, len(new_concepts), self.batch_size):
                        batch = new_concepts[i:i + self.batch_size]
                        future = executor.submit(self._load_concepts_batch, batch)
                        processed += future.result()
                        
                        progress = (i + len(batch)) / len(new_concepts) * 100
                        print(f"\rLoading new concepts: {progress:.1f}% complete | {processed} loaded", end='')

            print(f"\nCompleted loading {processed} new concepts")
            return processed

        except Exception as e:
            logger.error(f"Error in parallel concept processing and loading: {str(e)}")
            raise

    def _load_concepts_batch(self, batch):
        """Load a batch of new concepts into Neo4j"""
        try:
            cypher = """
            UNWIND $batch as concept
            MERGE (c:Concept {cui: concept.cui})
            SET 
                c.term = concept.term,
                c.domain = concept.domain,
                c.source = concept.source,
                c.priority = concept.priority,
                c.created_at = concept.created_at
            """
            self.graph.query(cypher, {'batch': batch})
            return len(batch)
        except Exception as e:
            logger.error(f"Error loading concepts batch: {str(e)}")
            raise

    def get_processing_stats(self):
        """Get statistics about the processed data"""
        try:
            stats = {}
            
            # Get concept counts by source and priority
            cypher_concepts = """
            MATCH (c:Concept)
            RETURN 
                c.source as source,
                c.priority as priority,
                count(c) as count
            ORDER BY source, priority
            """
            stats['concepts'] = self.graph.query(cypher_concepts)
            
            # Get relationship counts
            cypher_rels = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """
            stats['relationships'] = self.graph.query(cypher_rels)
            
            # Get semantic type counts
            cypher_sem = """
            MATCH (s:SemanticType)<-[:HAS_SEMANTIC_TYPE]-(c:Concept)
            RETURN 
                s.category as category,
                count(DISTINCT c) as concept_count
            ORDER BY concept_count DESC
            """
            stats['semantic_type'] = self.graph.query(cypher_sem)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            raise

    

    def _verify_semantic_type_updates(self):
        """Verify semantic type updates"""
        try:
            stats = {}
            
            # Check SemanticType nodes
            cypher_nodes = """
            MATCH (st:SemanticType)
            RETURN 
                count(st) as total_nodes,
                count(st.cui) as has_cui,
                count(st.type_id) as has_type_id,
                count(st.semantic_type) as has_semantic_type,
                count(st.name) as has_name,
                collect(DISTINCT st.semantic_type) as types
            """
            node_stats = self.graph.query(cypher_nodes)
            stats['nodes'] = dict(node_stats[0])
            
            # Check relationships and concept properties
            cypher_rels = """
            MATCH (c:Concept)-[r:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            RETURN 
                count(DISTINCT c) as concepts_with_type,
                count(c.semantic_type) as concepts_with_type_property,
                count(c.semantic_type_id) as concepts_with_type_id,
                count(r) as total_relationships
            """
            rel_stats = self.graph.query(cypher_rels)
            stats['relationships'] = dict(rel_stats[0])
            
            # Check for missing properties
            cypher_missing = """
            MATCH (st:SemanticType)
            WHERE st.cui IS NULL 
               OR st.type_id IS NULL 
               OR st.semantic_type IS NULL
            RETURN count(st) as missing_properties
            """
            missing_stats = self.graph.query(cypher_missing)
            stats['missing'] = dict(missing_stats[0])
            
            print("\n=== Semantic Type Update Verification ===")
            print(f"Total SemanticType nodes: {stats['nodes']['total_nodes']}")
            print(f"Nodes with CUI: {stats['nodes']['has_cui']}")
            print(f"Nodes with type_id: {stats['nodes']['has_type_id']}")
            print(f"Nodes with semantic_type: {stats['nodes']['has_semantic_type']}")
            print(f"Nodes with missing properties: {stats['missing']['missing_properties']}")
            print(f"\nConcepts with semantic type: {stats['relationships']['concepts_with_type']}")
            print(f"Total HAS_SEMANTIC_TYPE relationships: {stats['relationships']['total_relationships']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error verifying semantic type updates: {str(e)}")
            raise

    def print_semantic_type_mapping(self):
        """Print the semantic type mapping from constants"""
        print("\nPriority 1 Semantic Types:")
        for type_id, name in IMPORTANT_SEMANTIC_TYPE['priority_1'].items():
            print(f"{type_id}: {name}")
            
        print("\nPriority 2 Semantic Types:")
        for type_id, name in IMPORTANT_SEMANTIC_TYPE['priority_2'].items():
            print(f"{type_id}: {name}")

    def check_semantic_type_nodes(self):
        """Check current state of semantic type nodes"""
        try:
            cypher = """
            MATCH (st:SemanticType)
            RETURN 
                st.name as name,
                st.type_id as type_id,
                st.semantic_type as semantic_type,
                st.cui as cui,
                st.tree_number as tree_number
            ORDER BY st.name
            """
            
            results = self.graph.query(cypher)
            print("\n=== Current Semantic Type Nodes ===")
            for row in results:
                print(f"Name: {row['name']}")
                print(f"Type ID: {row['type_id']}")
                print(f"Semantic Type: {row['semantic_type']}")
                print(f"CUI: {row['cui']}")
                print(f"Tree Number: {row['tree_number']}")
                print("-" * 50)
            
            return results

        except Exception as e:
            logger.error(f"Error checking semantic type nodes: {str(e)}")
            raise

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
            
            result = self.graph.query(cypher, {
                'type_mapping': SEMANTIC_TYPE_TO_LABEL
            })
            
            print(f"Added labels to {result[0]['labeled_concepts']} concepts")
        
        except Exception as e:
            logger.error(f"Error adding concept labels: {str(e)}")
            raise

   
    # def verify_concepts_exist(self, cuis: List[str]) -> Dict:
    #     """Verify if concepts exist and return missing ones"""
    #     try:
    #         cypher = """
    #         UNWIND $cuis as cui
    #         OPTIONAL MATCH (c:Concept {cui: cui})
    #         RETURN cui, 
    #                CASE WHEN c IS NULL THEN false ELSE true END as exists
    #         """
            
    #         results = self.graph.query(cypher, {'cuis': cuis})
    #         existing = []
    #         missing = []
            
    #         for record in results:
    #             if record['exists']:
    #                 existing.append(record['cui'])
    #             else:
    #                 missing.append(record['cui'])
                
    #         return {
    #             'existing': existing,
    #             'missing': missing,
    #             'total_checked': len(cuis),
    #             'total_existing': len(existing),
    #             'total_missing': len(missing)
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error verifying concepts: {str(e)}")
    #         raise

    # def create_missing_concepts(self, mrconso_file: str, cuis: List[str]):
    #     """Create missing concepts from MRCONSO file"""
    #     try:
    #         # Read MRCONSO for the specific CUIs
    #         chunks = pd.read_csv(
    #             mrconso_file,
    #             sep='|',
    #             header=None,
    #             chunksize=self.batch_size,
    #             encoding='utf-8'
    #         )
            
    #         concepts_to_create = []
    #         for chunk in chunks:
    #             relevant_rows = chunk[
    #                 (chunk[0].isin(cuis)) &  # CUI
    #                 (chunk[1] == 'ENG')      # English only
    #             ]
                
    #             for _, row in relevant_rows.iterrows():
    #                 concepts_to_create.append({
    #                     'cui': row[0],
    #                     'term': row[14],    # STR
    #                     'source': row[11],   # SAB
    #                     'domain': self.usmle_domains.get(row[11], {}).get(row[12], 'unknown'),
    #                     'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #                 })
            
    #         if concepts_to_create:
    #             cypher = """
    #             UNWIND $batch as item
    #             MERGE (c:Concept {cui: item.cui})
    #             ON CREATE SET 
    #                 c.term = item.term,
    #                 c.source = item.source,
    #                 c.domain = item.domain,
    #                 c.created_at = item.created_at
    #             """
                
    #             self.graph.query(cypher, {'batch': concepts_to_create})
    #             logger.info(f"Created {len(concepts_to_create)} new concepts")
                
    #         return len(concepts_to_create)
            
    #     except Exception as e:
    #         logger.error(f"Error creating missing concepts: {str(e)}")
    #         raise


    def diagnose_missing_relationships(self, mrrel_file: str):
        """Optimized function to diagnose missing concepts with retry logic"""
        try:
            # Use all relationships from RELATION_TYPE_MAPPING
            target_relations = set(RELATION_TYPE_MAPPING.values())
            
            # Collect CUIs and relationships from MRREL
            all_cuis = set()
            relationship_pairs = {rel: [] for rel in target_relations}
            total_rows = 0
            
            logger.info("Reading MRREL file...")
            chunks = pd.read_csv(
                mrrel_file,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8',
                na_filter=False
            )
            
            for chunk in chunks:
                # Use the mapped relationship types
                relevant_rows = chunk[chunk[7].isin(target_relations)]

                total_rows += len(relevant_rows)
                
                for _, row in relevant_rows.iterrows():
                    rel_type = row[7]
                    cui1, cui2 = row[0], row[4]
                    all_cuis.add(cui1)
                    all_cuis.add(cui2)
                    relationship_pairs[rel_type].append((cui1, cui2))
            
            logger.info(f"Found {len(all_cuis)} unique CUIs in {total_rows} relationships")
            
            # Check existing CUIs in batches with retry
            existing_cuis = set()
            cui_batches = [list(all_cuis)[i:i + 1000] for i in range(0, len(all_cuis), 1000)]
            
            for batch_num, cui_batch in enumerate(cui_batches, 1):
                logger.info(f"Checking CUI batch {batch_num}/{len(cui_batches)}")
                
                cypher = """
                UNWIND $cuis as cui
                MATCH (c:Concept {cui: cui})
                RETURN c.cui as cui
                """
                
                try:
                    result = self._execute_query_with_retry(cypher, {'cuis': cui_batch})
                    existing_cuis.update(r['cui'] for r in result)
                except Exception as e:
                    logger.error(f"Error checking CUI batch {batch_num}: {str(e)}")
                    continue
            
            # Analyze relationships
            analysis = {rel: {'total': 0, 'missing_pairs': []} for rel in target_relations}
            
            for rel_type, pairs in relationship_pairs.items():
                analysis[rel_type]['total'] = len(pairs)
                
                for cui1, cui2 in pairs:
                    cui1_exists = cui1 in existing_cuis
                    cui2_exists = cui2 in existing_cuis
                    
                    if not (cui1_exists and cui2_exists):
                        analysis[rel_type]['missing_pairs'].append({
                            'cui1': cui1,
                            'cui2': cui2,
                            'cui1_exists': cui1_exists,
                            'cui2_exists': cui2_exists
                        })
            
            # Generate summary
            summary = {
                'total_relationships': total_rows,
                'unique_cuis': len(all_cuis),
                'existing_cuis': len(existing_cuis),
                'missing_cuis': len(all_cuis - existing_cuis),
                'relationship_stats': {}
            }
            
            for rel_type, data in analysis.items():
                if data['total'] > 0:
                    missing_count = len(data['missing_pairs'])
                    success_rate = ((data['total'] - missing_count) / data['total']) * 100
                    
                    summary['relationship_stats'][rel_type] = {
                        'total': data['total'],
                        'missing': missing_count,
                        'success_rate': f"{success_rate:.2f}%"
                    }
            
            # Log summary
            self._log_analysis_summary(summary)
            
            return {
                'analysis': analysis,
                'summary': summary,
                'missing_cuis': list(all_cuis - existing_cuis)
            }
            
        except Exception as e:
            logger.error(f"Error diagnosing relationships: {str(e)}")
            raise

    def _log_analysis_summary(self, summary: dict):
        """Log analysis summary with proper formatting"""
        logger.info("\n=== Relationship Analysis Summary ===")
        logger.info(f"Total Relationships: {summary['total_relationships']:,}")
        logger.info(f"Unique CUIs: {summary['unique_cuis']:,}")
        logger.info(f"Existing CUIs: {summary['existing_cuis']:,}")
        logger.info(f"Missing CUIs: {summary['missing_cuis']:,}")
        
        logger.info("\nRelationship Statistics:")
        for rel_type, stats in summary['relationship_stats'].items():
            logger.info(f"\n{rel_type}:")
            logger.info(f"  Total: {stats['total']:,}")
            logger.info(f"  Missing: {stats['missing']:,}")
            logger.info(f"  Success Rate: {stats['success_rate']}")

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
            
            # Verify cleanup
            verify_query = """
            MATCH (n)-[r]->(m)
            WITH n, m, type(r) as relType, count(r) as relCount
            WHERE relCount > 1
            RETURN count(*) as remainingDuplicates
            """
            
            result = self._execute_query_with_retry(verify_query)
            remaining_duplicates = result[0]['remainingDuplicates'] if result else 0
            
            if remaining_duplicates > 0:
                logger.warning(f"Found {remaining_duplicates} remaining duplicate relationships after cleanup")
            else:
                logger.info("All duplicate relationships have been cleaned up")
                
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up duplicate relationships: {str(e)}")
            raise

    def create_missing_concepts_and_relationships(self, mrrel_file: str, mrconso_file: str, mrsty_file: str):
        """Create missing concepts and their relationships with error handling"""
        try:
            logger.info("Starting creation of missing concepts and relationships...")
            
            # Single cleanup operation at the start
            deleted_count = self.cleanup_duplicate_relationships()
            logger.info(f"Cleaned up {deleted_count} duplicate relationships")
            
            # Get initial relationship count and capacity
            count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            current_count = self._execute_query_with_retry(count_query)[0]['count']
            remaining_capacity = 400000 - current_count  # Neo4j Free tier limit
            
            if remaining_capacity <= 0:
                logger.warning("No remaining relationship capacity")
                return {
                    'concepts_created': 0,
                    'relationships_created': 0,
                    'duplicates_removed': deleted_count > 0
                }
            
            # Process missing concepts and relationships
            missing_data = self.diagnose_missing_relationships(mrrel_file)
            if not missing_data or not missing_data.get('missing_cuis'):
                logger.info("No missing concepts found.")
                return {
                    'concepts_created': 0,
                    'relationships_created': 0,
                    'duplicates_removed': deleted_count > 0
                }
                
            missing_cuis = set(missing_data.get('missing_cuis', []))
            
            # Read concept data from MRCONSO for missing CUIs
            concepts_data = {}
            mrconso_chunks = pd.read_csv(
                mrconso_file,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8',
                na_filter=False
            )
            
            for chunk in mrconso_chunks:
                relevant_rows = chunk[
                    (chunk[0].isin(missing_cuis)) &  # CUI
                    (chunk[1] == 'ENG') &            # English
                    (chunk[12].isin(['PT', 'PN']))   # Preferred terms
                ]
                
                for _, row in relevant_rows.iterrows():
                    cui = row[0]
                    if cui not in concepts_data:
                        concepts_data[cui] = {
                            'cui': cui,
                            'term': row[14],  # String
                            'source': row[11],  # SAB
                            'semantic_types': []
                        }
            
            # Get semantic types for missing concepts
            mrsty_chunks = pd.read_csv(
                mrsty_file,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8',
                na_filter=False
            )
            
            for chunk in mrsty_chunks:
                relevant_rows = chunk[chunk[0].isin(missing_cuis)]
                for _, row in relevant_rows.iterrows():
                    cui = row[0]
                    if cui in concepts_data:
                        concepts_data[cui]['semantic_types'].append(row[1])  # TUI
            
            # Create concepts in batches
            concept_batches = [list(concepts_data.values())[i:i + 1000] 
                            for i in range(0, len(concepts_data), 1000)]
            
            created_concepts = 0
            for batch_num, concept_batch in enumerate(concept_batches, 1):
                logger.info(f"Creating concept batch {batch_num}/{len(concept_batches)}")
                
                cypher = """
                UNWIND $batch as item
                MERGE (c:Concept {cui: item.cui})
                SET c.term = item.term,
                    c.source = item.source,
                    c.semantic_types = item.semantic_types,
                    c.created_at = datetime()
                WITH c, item
                
                // Set labels based on semantic types (TUIs)
                FOREACH (tui IN CASE WHEN any(x IN item.semantic_types WHERE x IN ['T047','T048','T191']) 
                        THEN [1] ELSE [] END | SET c:Disease)
                FOREACH (tui IN CASE WHEN any(x IN item.semantic_types WHERE x IN ['T121','T200']) 
                        THEN [1] ELSE [] END | SET c:Drug)
                FOREACH (tui IN CASE WHEN any(x IN item.semantic_types WHERE x IN ['T023','T190']) 
                        THEN [1] ELSE [] END | SET c:Anatomy)
                FOREACH (tui IN CASE WHEN any(x IN item.semantic_types WHERE x IN ['T184']) 
                        THEN [1] ELSE [] END | SET c:Symptom)
                RETURN count(c) as created
                """
                
                result = self._execute_query_with_retry(cypher, {'batch': concept_batch})
                created_concepts += result[0]['created'] if result else 0
            
            logger.info(f"Created {created_concepts} concepts")
            
            # Create relationships with proper directionality
            relationship_count = 0
            for rel_type, data in missing_data['analysis'].items():
                if data['missing_pairs'] and relationship_count < remaining_capacity:
                    pairs_batch = [
                        {
                            'cui1': pair['cui1'],
                            'cui2': pair['cui2'],
                            'rel_type': rel_type,
                            'rel_attr': pair.get('rel_attr', ''),
                            'source': pair.get('source', ''),
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        for pair in data['missing_pairs']
                    ]
                    
                    batch_size = min(100, remaining_capacity - relationship_count)
                    
                    for i in range(0, len(pairs_batch), batch_size):
                        if relationship_count >= remaining_capacity:
                            break
                            
                        current_batch = pairs_batch[i:i + batch_size]
                        try:
                            cypher = """
                            UNWIND $batch as item
                            MATCH (c1:Concept {cui: item.cui1})
                            MATCH (c2:Concept {cui: item.cui2})
                            WITH c1, c2, item
                            CALL apoc.do.case([
                                // Reverse direction relationships (c2->c1)
                                item.rel_attr IN ['may_treat', 'may_prevent',
                                    'is_finding_of_disease', 'associated_finding_of',
                                    'may_be_finding_of_disease', 'associated_etiologic_finding_of',
                                    'is_not_finding_of_disease', 'clinical_course_of',
                                    'manifestation_of', 'location_of',
                                    'is_location_of_biological_process',
                                    'is_location_of_anatomic_structure',
                                    'mechanism_of_action_of',
                                    'contraindicated_mechanism_of_action_of',
                                    'cause_of'],
                                "MERGE (c2)-[r:$rel_type]->(c1) 
                                ON CREATE SET r += $props 
                                RETURN r",
                                
                                // Forward direction relationships (c1->c2)
                                item.rel_attr IN ['may_be_treated_by',
                                    'disease_has_accepted_treatment_with_regimen',
                                    'disease_has_finding', 'disease_may_have_finding',
                                    'has_causative_agent', 'occurs_in', 'has_location',
                                    'has_ingredient', 'has_precise_ingredient',
                                    'has_mechanism_of_action',
                                    'chemical_or_drug_has_mechanism_of_action',
                                    'chemical_or_drug_affects_gene_product',
                                    'contraindicated_with_disease', 'occurs_before',
                                    'regulates', 'negatively_regulates',
                                    'positively_regulates', 'develops_into',
                                    'has_course', 'part_of', 'drains_into',
                                    'may_be_diagnosed_by'],
                                "MERGE (c1)-[r:$rel_type]->(c2) 
                                ON CREATE SET r += $props 
                                RETURN r",
                                
                                // Bidirectional/symmetric relationships
                                item.rel_attr IN ['associated_with',
                                    'related_to', 'gene_associated_with_disease'],
                                "MERGE (c1)-[r:$rel_type]->(c2) 
                                ON CREATE SET r += $props 
                                RETURN r"
                            ],
                            "MERGE (c1)-[r:$rel_type]->(c2) 
                            ON CREATE SET r += $props 
                            RETURN r",
                            {
                                rel_type: item.rel_type,
                                props: {
                                    source: item.source,
                                    rel_attr: item.rel_attr,
                                    created_at: item.created_at
                                }
                            }) YIELD value
                            RETURN count(value) as created
                            """
                            
                            result = self._execute_query_with_retry(cypher, {
                                'batch': current_batch,
                                'rel_type': rel_type
                            })
                            
                            if result:
                                created = result[0]['created']
                                relationship_count += created
                                logger.info(f"Created {created} relationships of type {rel_type}")
            
                        except Exception as e:
                            if "exceeded the logical size limit" in str(e):
                                logger.warning("Relationship limit reached")
                                break
                            else:
                                logger.error(f"Error creating relationships batch: {str(e)}")
            
            return {
                'concepts_created': created_concepts,
                'relationships_created': relationship_count,
                'duplicates_removed': deleted_count > 0
            }
            
        except Exception as e:
            logger.error(f"Error creating concepts and relationships: {str(e)}")
            return {
                'concepts_created': 0,
                'relationships_created': 0,
                'duplicates_removed': False,
                'error': str(e)
            }

    def create_vector_index(self):
        """Create a vector index for concept embeddings"""
        try:
            logger.info("Creating vector index for concept embeddings...")
            
            # Check if Neo4j version supports vector indexes
            version_query = "CALL dbms.components() YIELD name, versions RETURN versions[0] as version"
            result = self._execute_query_with_retry(version_query)
            neo4j_version = result[0]['version'] if result else ""
            
            logger.info(f"Neo4j version: {neo4j_version}")
            
            # Try standard method first
            try:
                index_query = """
                CALL db.index.vector.createNodeIndex(
                    'concept_embeddings',
                    'Concept',
                    'embedding',
                    768,
                    'cosine'
                )
                """
                self._execute_query_with_retry(index_query)
                logger.info("Vector index 'concept_embeddings' created successfully")
                return True
            except Exception as e:
                logger.warning(f"Standard vector index creation failed: {str(e)}")
                
                # Try alternative method with Aura
                try:
                    index_query = """
                    CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
                    FOR (c:Concept)
                    ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                    """
                    self._execute_query_with_retry(index_query)
                    logger.info("Vector index created using alternative method")
                    return True
                except Exception as e2:
                    logger.error(f"Alternative vector index creation failed: {str(e2)}")
                    return False
        
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            return False

    def add_embeddings_to_existing_concepts(self, batch_size=200):
        """Add embeddings to existing concepts that don't have them"""
        try:
            # Initialize HuggingFace model for embeddings
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
            encoder = STEncoder(model)
            
            # Count concepts without embeddings
            count_query = """
                MATCH (c:Concept)
                WHERE c.term IS NOT NULL AND c.embedding IS NULL
                RETURN count(c) as count
            """
            result = self.graph.query(count_query)
            total_concepts = result[0]['count'] if result else 0
            
            if total_concepts == 0:
                logger.info("No concepts found that need embeddings")
                return 0
            
            logger.info(f"Found {total_concepts} concepts that need embeddings")
            
            # Process in batches
            processed = 0
            batch_num = 0
            
            while processed < total_concepts:
                batch_num += 1
                
                # Fetch batch
                batch_query = """
                    MATCH (c:Concept)
                    WHERE c.term IS NOT NULL AND c.embedding IS NULL
                    RETURN c.cui as cui, c.term as term
                    LIMIT $batch_size
                """
                
                batch = self.graph.query(batch_query, {'batch_size': batch_size})
                
                if not batch:
                    break
                    
                # Create embeddings
                terms = [item['term'] for item in batch]
                cuis = [item['cui'] for item in batch]
                
                logger.info(f"Generating embeddings for batch {batch_num} ({len(terms)} concepts)")
                embeddings = encoder.embed_documents(terms)
                
                # Update with embeddings
                update_query = """
                    UNWIND $data as item
                    MATCH (c:Concept {cui: item.cui})
                    SET c.embedding = item.embedding
                """
                
                data = [
                    {'cui': cui, 'embedding': embedding}
                    for cui, embedding in zip(cuis, embeddings)
                ]
                
                self.graph.query(update_query, {'data': data})
                
                processed += len(batch)
                logger.info(f"Progress: {processed}/{total_concepts} concepts processed")
                
                # Prevent overloading the server
                time.sleep(0.5)
            
            logger.info(f"Completed adding embeddings to {processed} concepts")
            return processed
            
        except Exception as e:
            logger.error(f"Error adding embeddings to existing concepts: {str(e)}")
            raise

    def vector_similarity_search(self, query_text, limit=10, score_threshold=0.6):
        """Search for concepts by vector similarity"""
        try:
            # Initialize the embedding model
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            encoder = STEncoder(model)
            
            # Get query embedding
            query_embedding = encoder.embed_query(query_text)
            
            # Execute vector search
            cypher = """
            CALL db.index.vector.queryNodes(
                'concept_embeddings',
                $limit,
                $embedding
            ) YIELD node, score
            WHERE score >= $threshold
            RETURN 
                node.cui as cui,
                node.term as term,
                node.semantic_type as semantic_type,
                node.domain as domain,
                score
            ORDER BY score DESC
            """
            
            try:
                results = self.graph.query(cypher, {
                    'embedding': query_embedding,
                    'limit': limit,
                    'threshold': score_threshold
                })
                
                return [dict(record) for record in results]
                
            except Exception as e:
                logger.warning(f"Vector index search failed: {str(e)}")
                
                # Fallback to manual similarity search for older Neo4j versions
                logger.info("Falling back to manual similarity calculation")
                
                fallback_query = """
                MATCH (c:Concept)
                WHERE c.embedding IS NOT NULL AND c.term IS NOT NULL
                RETURN c.cui as cui, c.term as term, c.embedding as embedding,
                       c.semantic_type as semantic_type, c.domain as domain
                LIMIT 1000
                """
                
                candidates = self.graph.query(fallback_query)
                
                # Calculate cosine similarity manually
                results = []
                for c in candidates:
                    if c['embedding'] is None:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, c['embedding'])
                    
                    if similarity >= score_threshold:
                        results.append({
                            'cui': c['cui'],
                            'term': c['term'],
                            'semantic_type': c['semantic_type'],
                            'domain': c['domain'],
                            'score': similarity
                        })
                
                # Sort by score and limit results
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:limit]
        
        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return []

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Convert to numpy arrays if they're lists
        if isinstance(vec1, list):
            vec1 = np.array(vec1)
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)

    def hybrid_search(self, query_text, keyword_limit=5, vector_limit=10, score_threshold=0.6):
        """Perform both keyword and vector search and combine results"""
        try:
            # Get keyword results
            keyword_results = self.search_concepts(query_text, limit=keyword_limit)
            
            # Get vector results
            vector_results = self.vector_similarity_search(
                query_text, 
                limit=vector_limit,
                score_threshold=score_threshold
            )
            
            # Combine results, preserving uniqueness by CUI
            combined = {}
            
            # Add keyword results with default score
            for item in keyword_results:
                cui = item['cui']
                combined[cui] = {
                    **item,
                    'score': 1.0,
                    'source': 'keyword'
                }
            
            # Add vector results
            for item in vector_results:
                cui = item['cui']
                
                if cui not in combined:
                    combined[cui] = {
                        **item,
                        'source': 'vector'
                    }
                else:
                    # If already exists from keyword search, keep the higher score
                    combined[cui] = {
                        **combined[cui],
                        'score': max(combined[cui].get('score', 0), item.get('score', 0)),
                        'source': 'hybrid'
                    }
            
            # Convert to list and sort by score
            results = list(combined.values())
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:vector_limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []