import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
from typing import List, Dict
import logging
from src.processors.base_processor import BaseProcessor, DatabaseMixin
from src.config.constants import IMPORTANT_RELATIONS, USMLE_DOMAINS, IMPORTANT_SEMANTIC_TYPE

logger = logging.getLogger(__name__)

class UMLSProcessor(BaseProcessor, DatabaseMixin):
    def __init__(self, graph):
        super().__init__(graph)
        self.batch_size = 10000
        self.num_workers = max(1, cpu_count() - 1)
        self.node_limit = 50000
        self.relationship_limit = 75000
        self.processed_concepts = set()
        
        # Configuration mappings
        self.important_relations = IMPORTANT_RELATIONS.get('priority_1', {})
        self.usmle_domains = {k: v.get('priority_1', {}) for k, v in USMLE_DOMAINS.items()}
        self.important_semantic_types = IMPORTANT_SEMANTIC_TYPE.get('priority_1', {})

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
            
            # Process files in parallel using ThreadPoolExecutor
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
                    'relationships': executor.submit(
                        self.process_mrrel, 
                        data['mrrel']
                    ),
                    'semantic_types': executor.submit(
                        self.process_mrsty, 
                        data['mrsty']
                    ),
                    'definitions': executor.submit(
                        self._process_mrdef_parallel, 
                        data['mrdef']
                    )
                }
                
                # Collect results
                results = {}
                for name, future in futures.items():
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing {name}: {str(e)}")
                        results[name] = 0
            
            total_time = time.time() - start_time
            print(f"\n=== Processing Complete ({total_time:.1f}s) ===")
            print(f"Concepts: {concepts_processed:,}")
            print(f"Relationships: {results.get('relationships', 0):,}")
            print(f"Semantic Types: {results.get('semantic_types', 0):,}")
            print(f"Definitions: {results.get('definitions', 0):,}")
            
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
                    c.created_at = item.created_at
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
        """Load processed concepts for filtering"""
        try:
            result = self.graph.query("MATCH (c:Concept) RETURN c.cui")
            self.processed_concepts = {record['c.cui'] for record in result}
            logger.info(f"Loaded {len(self.processed_concepts):,} processed concepts")
        except Exception as e:
            logger.error(f"Error loading processed concepts: {str(e)}")
            raise

    def process_mrrel(self, file_path: str) -> int:
        """Process MRREL file using thread-based processing"""
        try:
            processed = 0
            skipped = 0
            start_time = time.time()
            remaining_relations = self.relationship_limit - self._get_relationship_count()
            
            if remaining_relations <= 0:
                logger.warning("Relationship limit already reached")
                return 0
            
            print(f"\nProcessing relationships (limit: {remaining_relations:,})...")
            
            chunks = pd.read_csv(
                file_path,
                sep='|',
                header=None,
                chunksize=self.batch_size,
                encoding='utf-8'
            )
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for chunk_num, chunk in enumerate(chunks):
                    if processed >= remaining_relations:
                        break
                        
                    futures = [
                        executor.submit(self._process_rel_row, row)
                        for row in chunk.values
                    ]
                    
                    valid_relations = []
                    for future in futures:
                        result = future.result()
                        if result:
                            valid_relations.append(result)
                    
                    if valid_relations:
                        if processed + len(valid_relations) > remaining_relations:
                            valid_relations = valid_relations[:remaining_relations - processed]
                        
                        self._create_relationships_batch(valid_relations)
                        processed += len(valid_relations)
                        
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"\rProcessed: {processed:,}/{remaining_relations:,} relationships | "
                              f"Rate: {rate:.0f} rels/sec | "
                              f"Batch: {chunk_num}", end='')
                    
                    skipped += len(chunk) - len(valid_relations)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in MRREL processing: {str(e)}")
            raise

    def _process_rel_row(self, row) -> Dict:
        """Process a single relationship row"""
        try:
            cui1 = row[0]     # CUI1
            rel_type = row[3]  # REL
            rela = row[7]     # RELA
            cui2 = row[4]     # CUI2
            source = row[10]   # SAB
            
            # Check if both concepts exist and relation is important
            if (cui1 in self.processed_concepts and 
                cui2 in self.processed_concepts and 
                rel_type in self.important_relations):
                
                return {
                    'cui1': cui1,
                    'cui2': cui2,
                    'rel_type': self.important_relations[rel_type],
                    'rel_attribute': rela if pd.notna(rela) else None,
                    'source': source,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            return None
            
        except Exception as e:
            logger.error(f"Error processing relationship row: {str(e)}")
            return None

    def _create_relationships_batch(self, batch: List[Dict]):
        """Create relationships in Neo4j using optimized Cypher"""
        try:
            cypher = """
            UNWIND $batch as rel
            MATCH (c1:Concept {cui: rel.cui1})
            MATCH (c2:Concept {cui: rel.cui2})
            CALL {
                WITH c1, c2, rel
                MERGE (c1)-[r:RELATES_TO {type: rel.rel_type}]->(c2)
                ON CREATE SET 
                    r.rel_attribute = rel.rel_attribute,
                    r.source = rel.source,
                    r.created_at = rel.created_at
            } IN TRANSACTIONS OF 1000 ROWS
            """
            self.graph.query(cypher, {'batch': batch})
            
        except Exception as e:
            logger.error(f"Error creating relationships batch: {str(e)}")
            
            # Try processing in smaller batches if large batch fails
            if len(batch) > 100:
                mid = len(batch) // 2
                self._create_relationships_batch(batch[:mid])
                self._create_relationships_batch(batch[mid:])
            else:
                raise

    def _get_relationship_count(self) -> int:
        """Get current count of relationships in the database"""
        try:
            cypher = "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
            result = self.graph.query(cypher)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting relationship count: {str(e)}")
            return 0
        
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
                
                return {
                    'cui': cui,
                    'semantic_type_id': semantic_type_id,
                    'semantic_type': self.important_semantic_types[semantic_type_id],
                    'tree_number': row[2],  # STN
                    'name': row[3],         # STY
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            return None
            
        except Exception as e:
            logger.error(f"Error processing semantic type row: {str(e)}")
            return None

    def _create_semantic_types_batch(self, batch: List[Dict]):
        """Create semantic type nodes and relationships in Neo4j"""
        try:
            # First create semantic type nodes
            cypher_nodes = """
            UNWIND $batch as item
            MERGE (st:SemanticType {type_id: item.semantic_type_id})
            ON CREATE SET 
                st.name = item.semantic_type,
                st.tree_number = item.tree_number,
                st.created_at = item.created_at
            """
            self.graph.query(cypher_nodes, {'batch': batch})
            
            # Then create relationships and update concepts
            cypher_rels = """
            UNWIND $batch as item
            MATCH (c:Concept {cui: item.cui})
            MATCH (st:SemanticType {type_id: item.semantic_type_id})
            MERGE (c)-[r:HAS_SEMANTIC_TYPE]->(st)
            ON CREATE SET 
                r.created_at = item.created_at
            SET c.semantic_type = item.semantic_type,
                c.semantic_type_id = item.semantic_type_id
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
            logger.error(f"Error getting concept definitions: {str(e)}")
            return []

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
            MATCH path = shortestPath(
                (start:Concept {cui: $start_cui})-[*..${max_depth}]->(end:Concept {cui: $end_cui})
            )
            RETURN [node in nodes(path) | {
                cui: node.cui,
                term: node.term
            }] as nodes,
            [rel in relationships(path) | type(rel)] as relationships
            """
            params = {
                'start_cui': start_cui,
                'end_cui': end_cui,
                'max_depth': max_depth
            }
            result = self.graph.query(cypher, params)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error finding path: {str(e)}")
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