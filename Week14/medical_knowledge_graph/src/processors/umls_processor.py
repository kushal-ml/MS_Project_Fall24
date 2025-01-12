import logging
from typing import Dict
from src.processors.base_processor import BaseProcessor, DatabaseMixin
from src.config.constants import IMPORTANT_RELATIONS, USMLE_DOMAINS,IMPORTANT_SEMANTIC_TYPE
logger = logging.getLogger(__name__)


class UMLSProcessor(BaseProcessor,DatabaseMixin):
    def __init__(self, graph):
        super().__init__(graph)
        self.node_limit = 100000  # Reduced limit
        self.relationship_limit = 150000  # Reduced limit
        self.batch_size = 50  # Smaller batch size for better memory management
        self.processed_concepts = set()
        self.important_relations = IMPORTANT_RELATIONS
        self.usmle_domains = USMLE_DOMAINS
        self.important_semantic_types = IMPORTANT_SEMANTIC_TYPE

    def create_indexes(self):
        """Create essential indexes only"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.cui)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.term)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.semantic_type)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.semantic_type_id)",
            "CREATE INDEX IF NOT EXISTS FOR (st:SemanticType) ON (st.type_id)",
            "CREATE INDEX IF NOT EXISTS FOR (st:SemanticType) ON (st.name)"
        ]
        for index in indexes:
            self.graph.query(index)

    def process_dataset(self, file_paths):
        """Process core UMLS files"""
        try:
            if self._get_node_count() >= self.node_limit:
                logger.warning(f"Node limit reached: {self._get_node_count()} nodes")
                return

            self.process_mrconso(file_paths['mrconso'])
            self._load_processed_concepts()
            self.process_mrrel(file_paths['mrrel'])
            self.process_mrdef(file_paths['mrdef'])
            self.process_mrsty(file_paths['mrsty'])
            
        except Exception as e:
            logger.error(f"Error processing UMLS dataset: {str(e)}")
            raise

    def process_mrconso(self, file_path):
        """Process MRCONSO.RRF with strict filtering"""
        try:
            batch = []
            processed = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    concept = self._parse_mrconso_line(line)
                    if concept:
                        batch.append(concept)
                        processed += 1

                        if len(batch) >= self.batch_size:
                            self._process_batch(batch, self._get_concept_cypher())
                            logger.info(f"Processed {processed} concepts")
                            batch = []

                if batch:
                    self._process_batch(batch, self._get_concept_cypher())

            logger.info(f"Total concepts processed: {processed}")

        except Exception as e:
            logger.error(f"Error processing MRCONSO: {str(e)}")
            raise

    def process_mrrel(self, file_path):
        """Process MRREL with strict relationship filtering"""
        try:
            current_count = self._get_relationship_count()
            remaining = self.relationship_limit - current_count
            
            if remaining <= 0:
                logger.warning("Relationship limit reached")
                return
                
            batch = []
            processed = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if processed >= remaining:
                        break

                    fields = line.strip().split('|')
                    cui1, cui2, rel_type = fields[0], fields[4], fields[3]

                    if (cui1 not in self.processed_concepts or 
                        cui2 not in self.processed_concepts or
                        rel_type not in self.important_relations):
                        continue

                    relationship = {
                        'cui1': cui1,
                        'cui2': cui2,
                        'rel_type': self.important_relations[rel_type],
                        'source': fields[10]
                    }
                    batch.append(relationship)
                    processed += 1
                    
                    if len(batch) >= self.batch_size:
                        self._create_relationships_batch(batch)
                        batch = []

                if batch:
                    self._create_relationships_batch(batch)
                        
            logger.info(f"Total relationships processed: {processed}")
                        
        except Exception as e:
            logger.error(f"Error processing MRREL: {str(e)}")
            raise

    def process_mrdef(self, file_path):
        """Process MRDEF selectively"""
        try:
            batch = []
            processed = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    fields = line.strip().split('|')
                    cui = fields[0]
                    
                    if cui not in self.processed_concepts:
                        continue
                        
                    definition = {
                        'cui': cui,
                        'text': fields[5],
                        'def_id': f"DEF_{cui}"
                    }
                    batch.append(definition)
                    processed += 1

                    if len(batch) >= self.batch_size:
                        self._process_batch(batch, self._get_definition_cypher())
                        batch = []

                if batch:
                    self._process_batch(batch, self._get_definition_cypher())

            logger.info(f"Total definitions processed: {processed}")

        except Exception as e:
            logger.error(f"Error processing MRDEF: {str(e)}")
            raise

    def process_mrsty(self, file_path):
        """Process MRSTY.RRF to add semantic types to concepts"""
        try:
            batch = []
            processed = 0
            skipped = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    fields = line.strip().split('|')
                    cui = fields[0]
                    semantic_type_id = fields[1]  # TUI field
                    semantic_type_name = fields[3]  # STY field
                    
                    # Only process if concept exists and semantic type is important
                    if cui in self.processed_concepts:
                        if semantic_type_id in self.important_semantic_types:
                            batch.append({
                                'cui': cui,
                                'semantic_type': self.important_semantic_types[semantic_type_id],
                                'semantic_type_id': semantic_type_id,
                                'original_name': semantic_type_name
                            })
                            processed += 1
                        else:
                            skipped += 1

                        if len(batch) >= self.batch_size:
                            self._update_semantic_types_batch(batch)
                            logger.info(f"Processed {processed} semantic types")
                            batch = []

                if batch:
                    self._update_semantic_types_batch(batch)

            logger.info(f"Semantic types processing complete - Added: {processed}, Skipped: {skipped}")

        except Exception as e:
            logger.error(f"Error processing MRSTY: {str(e)}")
            raise
        
    def _update_semantic_types_batch(self, batch):
        """Update concepts with semantic types in batch"""
        try:
            # Update existing concepts with semantic type information
            cypher = """
            UNWIND $batch AS item
            MATCH (c:Concept {cui: item.cui})
            SET c.semantic_type = item.semantic_type,
                c.semantic_type_id = item.semantic_type_id,
                c.original_semantic_name = item.original_name
            """
            self.graph.query(cypher, {'batch': batch})
            
            # Optionally create semantic type nodes and relationships
            cypher_semantic_nodes = """
            UNWIND $batch AS item
            MERGE (st:SemanticType {type_id: item.semantic_type_id})
            SET st.name = item.semantic_type
            WITH st, item
            MATCH (c:Concept {cui: item.cui})
            MERGE (c)-[r:HAS_SEMANTIC_TYPE]->(st)
            """
            self.graph.query(cypher_semantic_nodes, {'batch': batch})
            
        except Exception as e:
            logger.error(f"Error updating semantic types batch: {str(e)}")
            # Try processing one by one if batch fails
            for item in batch:
                try:
                    self.graph.query("""
                    MATCH (c:Concept {cui: $cui})
                    SET c.semantic_type = $semantic_type
                    """, item)
                except Exception as inner_e:
                    logger.error(f"Error updating individual semantic type: {str(inner_e)}")

    def _parse_mrconso_line(self, line):
        """Parse MRCONSO line with strict filtering"""
        fields = line.strip().split('|')
        
        if fields[1] != "ENG":  # English only
            return None

        source = fields[11]
        tty = fields[12]
        
        domain_info = self._get_domain_priority(source, tty)
        if not domain_info:
            return None
            
        return {
            'cui': fields[0],
            'term': fields[14],
            'source': source,
            'domain': domain_info['domain'],
            'priority': domain_info['priority']
        }

    def _create_relationships_batch(self, batch):
        """Create relationships in batch"""
        cypher = """
        UNWIND $batch AS rel
        MATCH (c1:Concept {cui: rel.cui1})
        MATCH (c2:Concept {cui: rel.cui2})
        MERGE (c1)-[r:RELATES_TO {type: rel.rel_type}]->(c2)
        """
        self.graph.query(cypher, {'batch': batch})

    def _get_concept_cypher(self):
        """Get Cypher query for concept creation"""
        return """
        UNWIND $batch as concept
        MERGE (c:Concept {cui: concept.cui})
        SET c.term = concept.term,
            c.domain = concept.domain,
            c.priority = concept.priority
        """

    def _get_definition_cypher(self):
        """Get Cypher query for definition creation"""
        return """
        UNWIND $batch as def
        MATCH (c:Concept {cui: def.cui})
        MERGE (d:Definition {id: def.def_id})
        SET d.text = def.text
        MERGE (c)-[r:HAS_DEFINITION]->(d)
        """

    def _load_processed_concepts(self):
        """Load processed concepts from graph"""
        cypher = "MATCH (c:Concept) RETURN c.cui"
        result = self.graph.query(cypher)
        self.processed_concepts = {record['c.cui'] for record in result}
        logger.info(f"Loaded {len(self.processed_concepts)} processed concepts")

    def _get_domain_priority(self, source, tty):
        """Get domain and priority info"""
        if source not in self.usmle_domains:
            return None

        for priority in ['priority_1']:  # Only check priority_1
            if tty in self.usmle_domains[source].get(priority, {}):
                return {
                    'domain': self.usmle_domains[source][priority][tty],
                    'priority': priority
                }
        return None
    


    def _get_relationship_count(self):
        """Get current count of all relationships in the database"""
        try:
            # Count all relationships regardless of type
            cypher = """
            MATCH ()-[r]->() 
            RETURN count(r) as count
            """
            result = self.graph.query(cypher)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting relationship count: {str(e)}")
            return 0

    def _get_node_count(self):
        """Get current count of concept nodes in the database"""
        try:
            # First check if Concept label exists
            check_label = """
            CALL db.labels() 
            YIELD label 
            RETURN count(label) as count 
            WHERE label = 'Concept'
            """
            label_exists = self.graph.query(check_label)
            
            if not label_exists or label_exists[0]['count'] == 0:
                return 0
                
            # If label exists, count nodes
            cypher = """
            MATCH (c:Concept) 
            RETURN count(c) as count
            """
            result = self.graph.query(cypher)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting node count: {str(e)}")
            return 0

    # Change this method
    def _create_relationships_batch(self, batch):
        """Create relationships in batch"""
        try:
            # Instead of creating generic RELATES_TO relationships,
            # create relationships with their specific types
            cypher = """
            UNWIND $batch AS rel
            MATCH (c1:Concept {cui: rel.cui1})
            MATCH (c2:Concept {cui: rel.cui2})
            CALL apoc.merge.relationship(c1, rel.rel_type, {}, {}, c2)
            YIELD rel as created
            RETURN count(created) as count
            """
            self.graph.query(cypher, {'batch': batch})
        except Exception as e:
            # Fallback without APOC
            cypher = """
            UNWIND $batch AS rel
            MATCH (c1:Concept {cui: rel.cui1})
            MATCH (c2:Concept {cui: rel.cui2})
            MERGE (c1)-[r:`${rel.rel_type}`]->(c2)
            """
            self.graph.query(cypher, {'batch': batch})


    def get_concepts_by_semantic_type(self, semantic_type: str, limit: int = 100):
        """Get concepts of a specific semantic type"""
        try:
            cypher = """
            MATCH (c:Concept)-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            WHERE st.name = $semantic_type
            RETURN c.cui as cui, c.term as term, c.semantic_type as type
            LIMIT $limit
            """
            results = self.graph.query(cypher, {
                'semantic_type': semantic_type,
                'limit': limit
            })
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Error getting concepts by semantic type: {str(e)}")
            return []

    def get_semantic_types_for_concept(self, cui: str):
        """Get all semantic types for a specific concept"""
        try:
            cypher = """
            MATCH (c:Concept {cui: $cui})-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            RETURN st.name as semantic_type, st.type_id as type_id
            """
            results = self.graph.query(cypher, {'cui': cui})
            return [{'semantic_type': result['semantic_type']} for result in results]
        except Exception as e:
            logger.error(f"Error getting semantic types for concept {cui}: {str(e)}")
            return []

    def get_all_semantic_types(self):
        """Get list of all semantic types in the database"""
        try:
            cypher = """
            MATCH (st:SemanticType)
            RETURN st.name as name, st.type_id as type_id,
                   count((st)<-[:HAS_SEMANTIC_TYPE]-()) as concept_count
            ORDER BY concept_count DESC
            """
            results = self.graph.query(cypher)
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Error getting all semantic types: {str(e)}")
            return []

    def get_related_concepts_by_semantic_type(self, cui: str, semantic_type: str, limit: int = 10):
        """Get related concepts of a specific semantic type"""
        try:
            cypher = """
            MATCH (c1:Concept {cui: $cui})-[r]->(c2:Concept)-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            WHERE st.name = $semantic_type
            RETURN c2.term as related_term, 
                   c2.cui as related_cui,
                   type(r) as relationship_type
            LIMIT $limit
            """
            results = self.graph.query(cypher, {
                'cui': cui,
                'semantic_type': semantic_type,
                'limit': limit
            })
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Error getting related concepts by semantic type: {str(e)}")
            return []

    def get_semantic_type_statistics(self):
        """Get statistics about semantic types usage"""
        try:
            cypher = """
            MATCH (st:SemanticType)
            OPTIONAL MATCH (st)<-[:HAS_SEMANTIC_TYPE]-(c:Concept)
            WITH st.name as semantic_type, 
                 count(c) as concept_count,
                 count(DISTINCT (c)-[]->()) as relationship_count
            RETURN semantic_type, concept_count, relationship_count
            ORDER BY concept_count DESC
            """
            results = self.graph.query(cypher)
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Error getting semantic type statistics: {str(e)}")
            return []

    def find_concepts_by_semantic_type_and_term(self, semantic_type: str, term_pattern: str, limit: int = 10):
        """Find concepts of a specific semantic type matching a term pattern"""
        try:
            cypher = """
            MATCH (c:Concept)-[:HAS_SEMANTIC_TYPE]->(st:SemanticType)
            WHERE st.name = $semantic_type
            AND toLower(c.term) CONTAINS toLower($term_pattern)
            RETURN c.cui as cui, 
                   c.term as term,
                   c.semantic_type as type
            LIMIT $limit
            """
            results = self.graph.query(cypher, {
                'semantic_type': semantic_type,
                'term_pattern': term_pattern,
                'limit': limit
            })
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Error finding concepts by semantic type and term: {str(e)}")
            return []

    def get_semantic_type_hierarchy(self):
        """Get hierarchical relationships between semantic types"""
        try:
            cypher = """
            MATCH (st1:SemanticType)
            OPTIONAL MATCH (st1)<-[:HAS_SEMANTIC_TYPE]-(c:Concept)-[:broader_than|parent_of]->(c2)-[:HAS_SEMANTIC_TYPE]->(st2:SemanticType)
            WHERE st1 <> st2
            WITH st1.name as semantic_type, 
                 st2.name as broader_type,
                 count(DISTINCT c) as concept_count
            WHERE broader_type IS NOT NULL
            RETURN semantic_type, broader_type, concept_count
            ORDER BY concept_count DESC
            """
            results = self.graph.query(cypher)
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Error getting semantic type hierarchy: {str(e)}")
            return []