import logging
from typing import Dict
from src.processors.base_processor import BaseProcessor, DatabaseMixin
from src.config.constants import IMPORTANT_RELATIONS, USMLE_DOMAINS
logger = logging.getLogger(__name__)


class UMLSProcessor(BaseProcessor,DatabaseMixin):
    def __init__(self, graph):
        super().__init__(graph)
        self.node_limit = 190000
        self.relationship_limit = 200000  # Added relationship limit
        self.batch_size = 50  # Reduced batch size
        self.processed_concepts = set()
        self.processed_stns = set()
        self.important_relations = IMPORTANT_RELATIONS
        self.usmle_domains = USMLE_DOMAINS
        
    def _get_relationship_count(self):
        """Get current relationship count"""
        cypher = "MATCH ()-[r]->() RETURN count(r) as count"
        result = self.graph.query(cypher)
        return result[0]['count']

    def create_indexes(self):
        """Create necessary indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.cui)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.priority)",
            "CREATE INDEX IF NOT EXISTS FOR (st:SemanticType) ON (st.tui)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Definition) ON (d.id)"
        ]
        for index in indexes:
            self.graph.query(index)

    def process_dataset(self, file_paths):
        """Process all UMLS files"""
        try:
            # Check node limit before processing
            if self._get_node_count() >= self.node_limit:
                logger.warning(f"Node limit reached: {self._get_node_count()} nodes")
                return

            # Process each file in order
            self.process_mrconso(file_paths['mrconso'])
            self._load_processed_concepts()
            self.process_mrrel(file_paths['mrrel'])
            self.process_mrdef(file_paths['mrdef'])
            self.process_mrsty(file_paths['mrsty'])
            
        except Exception as e:
            logger.error(f"Error processing UMLS dataset: {str(e)}")
            raise

    def process_mrconso(self, file_path):
        """Process MRCONSO.RRF with USMLE focus"""
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

            self.processed_items.update([item['cui'] for item in batch])
            logger.info(f"Total concepts processed: {processed}")

        except Exception as e:
            logger.error(f"Error processing MRCONSO: {str(e)}")
            raise

    def process_mrrel(self, file_path):
        """Process MRREL with relationship limit"""
        logger.info("Processing MRREL.RRF...")
        try:
            # Check current relationship count
            current_count = self._get_relationship_count()
            remaining_relationships = self.relationship_limit - current_count
            
            if remaining_relationships <= 0:
                logger.warning("Relationship limit already reached")
                return
                
            logger.info(f"Current relationships: {current_count}")
            logger.info(f"Remaining relationships: {remaining_relationships}")
            
            # Load processed concepts
            self._load_processed_concepts()
            
            batch = []
            processed = 0
            skipped = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if processed >= remaining_relationships:
                        logger.warning(f"Reached relationship limit of {self.relationship_limit}")
                        break

                    fields = line.strip().split('|')
                    cui1 = fields[0]
                    cui2 = fields[4]
                    rel_type = fields[3]

                    # Skip if concepts not in processed set or non-important relationships
                    if (cui1 not in self.processed_concepts or 
                        cui2 not in self.processed_concepts or
                        rel_type not in self.important_relations):
                        skipped += 1
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
                        try:
                            # Check if adding batch would exceed limit
                            if (current_count + len(batch)) > self.relationship_limit:
                                # Process individually up to limit
                                remaining = self.relationship_limit - current_count
                                for i in range(remaining):
                                    self._create_single_relationship(batch[i])
                                logger.info(f"Reached relationship limit during batch processing")
                                return
                                
                            self._create_relationships_batch(batch)
                            current_count += len(batch)
                            logger.info(f"Processed {processed} relationships")
                            batch = []
                        except Exception as e:
                            logger.error(f"Batch processing error: {str(e)}")
                            batch = []
                            
            if batch:
                # Check final batch
                if (current_count + len(batch)) <= self.relationship_limit:
                    self._create_relationships_batch(batch)
                
            logger.info(f"Total relationships processed: {processed}")
            logger.info(f"Skipped relationships: {skipped}")
                        
        except Exception as e:
            logger.error(f"Error processing MRREL: {str(e)}")
            raise

    def process_mrdef(self, file_path):
        """Process MRDEF with definitions"""
        try:
            batch = []
            processed = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    definition = self._parse_mrdef_line(line)
                    if definition:
                        batch.append(definition)
                        processed += 1

                        if len(batch) >= self.batch_size:
                            self._process_batch(batch, self._get_definition_cypher())
                            logger.info(f"Processed {processed} definitions")
                            batch = []

                if batch:
                    self._process_batch(batch, self._get_definition_cypher())

            logger.info(f"Total definitions processed: {processed}")

        except Exception as e:
            logger.error(f"Error processing MRDEF: {str(e)}")
            raise

    def process_mrsty(self, file_path):
        """Process MRSTY.RRF for semantic types"""
        try:
            batch = []
            processed = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    semantic_type = self._parse_mrsty_line(line)
                    if semantic_type and semantic_type['stn'] not in self.processed_stns:
                        batch.append(semantic_type)
                        processed += 1

                        if len(batch) >= self.batch_size:
                            self._process_batch(batch, self._get_semantic_type_cypher())
                            self.processed_stns.update(item['stn'] for item in batch)
                            logger.info(f"Processed {processed} semantic types")
                            batch = []

                if batch:
                    self._process_batch(batch, self._get_semantic_type_cypher())
                    self.processed_stns.update(item['stn'] for item in batch)

            logger.info(f"Total semantic types processed: {processed}")

        except Exception as e:
            logger.error(f"Error processing MRSTY: {str(e)}")
            raise

    # Helper methods for parsing lines
    def _parse_mrconso_line(self, line):
        """Parse MRCONSO line into concept dictionary"""
        fields = line.strip().split('|')
        
        if fields[1] != "ENG":  # Process only English terms
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

    def _parse_mrrel_line(self, line):
        """Parse MRREL line into relationship dictionary"""
        fields = line.strip().split('|')
        cui1, cui2, rel_type = fields[0], fields[4], fields[3]
        
        if (cui1 not in self.processed_concepts or 
            cui2 not in self.processed_concepts or
            rel_type not in self.important_relations):
            return None
            
        return {
            'cui1': cui1,
            'cui2': cui2,
            'rel_type': self.important_relations[rel_type],
            'source': fields[10]
        }

    def _parse_mrdef_line(self, line):
        """Parse MRDEF line into definition dictionary"""
        fields = line.strip().split('|')
        cui = fields[0]
        
        if cui not in self.processed_concepts:
            return None
            
        return {
            'cui': cui,
            'source': fields[4],
            'text': fields[5],
            'def_id': f"DEF_{cui}_{fields[4]}"
        }

    def _parse_mrsty_line(self, line):
        """Parse MRSTY line into semantic type dictionary"""
        fields = line.strip().split('|')
        return {
            'cui': fields[0],
            'tui': fields[1],
            'stn': fields[2],
            'sty': fields[3]
        }

    def _create_single_relationship(self, rel):
        """Create a single relationship"""
        cypher = """
        MATCH (c1:Concept {cui: $cui1})
        MATCH (c2:Concept {cui: $cui2})
        MERGE (c1)-[r:RELATES_TO {
            type: $rel_type,
            source: $source
        }]->(c2)
        """
        self.graph.query(cypher, rel)

    def _create_relationships_batch(self, batch):
        """Create relationships in batch"""
        try:
            cypher = """
            UNWIND $batch AS rel
            MATCH (c1:Concept {cui: rel.cui1})
            MATCH (c2:Concept {cui: rel.cui2})
            MERGE (c1)-[r:RELATES_TO {type: rel.rel_type, source: rel.source}]->(c2)
            """
            self.graph.query(cypher, {'batch': batch})
        except Exception as e:
            logger.error(f"Error in batch processing relationships: {str(e)}")
            raise
    
    # Cypher query methods

    def _get_concept_info(self, term: str) -> dict:
        """Get basic concept information for a term"""
        cypher = """
        MATCH (c:Concept)
        WHERE toLower(c.term) CONTAINS toLower($term)
        OR toLower($term) CONTAINS toLower(c.term)
        RETURN c.term as term, c.cui as cui, c.domain as domain
        LIMIT 5
        """
        results = self.graph.query(cypher, {'term': term})
        return [{'term': r['term'], 'cui': r['cui'], 'domain': r['domain']} for r in results]

    def _get_definitions(self, cui: str) -> list:
        """Get definitions for a concept"""
        cypher = """
        MATCH (c:Concept {cui: $cui})-[:HAS_DEFINITION]->(d:Definition)
        RETURN d.text as definition
        """
        results = self.graph.query(cypher, {'cui': cui})
        return [r['definition'] for r in results]

    def _get_top_relationships(self, cui: str, limit: int = 10) -> list:
        """Get top relationships for a concept"""
        cypher = """
        MATCH (c:Concept {cui: $cui})-[r:RELATES_TO]->(c2:Concept)
        RETURN r.type as type, c2.term as related_term
        LIMIT $limit
        """
        results = self.graph.query(cypher, {'cui': cui, 'limit': limit})
        return [{'type': r['type'], 'related_term': r['related_term']} for r in results]


    def _get_concept_cypher(self):
        """Get Cypher query for concept creation"""
        return """
        UNWIND $batch as concept
        MERGE (c:Concept {cui: concept.cui})
        SET c.term = concept.term,
            c.source = concept.source,
            c.domain = concept.domain,
            c.priority = concept.priority
        """

    def _get_relationship_cypher(self):
        """Get Cypher query for relationship creation"""
        return """
        UNWIND $batch as rel
        MATCH (c1:Concept {cui: rel.cui1})
        MATCH (c2:Concept {cui: rel.cui2})
        MERGE (c1)-[r:RELATES_TO {type: rel.rel_type, source: rel.source}]->(c2)
        """

    def _get_definition_cypher(self):
        """Get Cypher query for definition creation"""
        return """
        UNWIND $batch as def
        MATCH (c:Concept {cui: def.cui})
        MERGE (d:Definition {id: def.def_id})
        SET d.text = def.text,
            d.source = def.source
        MERGE (c)-[r:HAS_DEFINITION]->(d)
        """

    def _get_semantic_type_cypher(self):
        """Get Cypher query for semantic type creation"""
        return """
        UNWIND $batch as item
        MERGE (st:SemanticType {tui: item.tui})
        SET st.name = item.sty,
            st.tree_number = item.stn
        MERGE (c:Concept {cui: item.cui})
        MERGE (c)-[r:HAS_SEMANTIC_TYPE]->(st)
        """

    def _load_processed_concepts(self):
        """Load processed concepts from the graph"""
        try:
            cypher = "MATCH (c:Concept) RETURN c.cui"
            result = self.graph.query(cypher)
            self.processed_concepts = {record['c.cui'] for record in result}
            logger.info(f"Loaded {len(self.processed_concepts)} processed concepts")
        except Exception as e:
            logger.error(f"Error loading processed concepts: {str(e)}")
            raise

    def _get_domain_priority(self, source, tty):
        """Determine domain and priority for a concept"""
        if source not in self.usmle_domains:
            return None

        for priority in ['priority_1', 'priority_2']:
            if tty in self.usmle_domains[source].get(priority, {}):
                return {
                    'domain': self.usmle_domains[source][priority][tty],
                    'priority': priority
                }
        return None

    def get_statistics(self) -> Dict[str, int]:
        """Get extended statistics including UMLS-specific metrics"""
        base_stats = super().get_statistics()
        return {
            **base_stats,
            'concepts': len(self.processed_concepts),
            'semantic_types': len(self.processed_stns),
            'relationships': self._get_relationship_count(),
            'available_nodes': self.node_limit - self._get_node_count()
        }