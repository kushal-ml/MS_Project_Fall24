from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.batch_size = 500
        self.processed_items = set()
        
    @abstractmethod
    def create_indexes(self):
        """Create necessary database indexes"""
        pass
    
    @abstractmethod
    def process_dataset(self, file_path: str):
        """Process the main dataset file"""
        pass
        
    def preprocess_data(self, data):
        """Default implementation of preprocess_data"""
        return data
    
    def process_dataset(self, files):
        """Default implementation of process_dataset"""
        return {}
    
    def validate_data(self, data):
        """Default implementation of validate_data"""
        return True


    def _process_batch(self, batch: List[Dict], cypher_query: str):
        """Generic batch processing method"""
        try:
            self.graph.query(cypher_query, {'batch': batch})
            logger.info(f"Processed batch of size {len(batch)}")
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            self._process_items_individually(batch, cypher_query)

    def _process_items_individually(self, items: List[Dict], cypher_query: str):
        """Process items one by one when batch fails"""
        for item in items:
            try:
                self.graph.query(cypher_query, item)
            except Exception as e:
                logger.error(f"Error processing individual item: {str(e)}")

    def _get_node_count(self, label: str = None) -> int:
        """Get count of nodes with optional label filter"""
        cypher = "MATCH (n{}) RETURN count(n) as count".format(
            f":{label}" if label else ""
        )
        result = self.graph.query(cypher)
        return result[0]['count']

    def _check_node_exists(self, label: str, property_name: str, property_value: Any) -> bool:
        """Check if a node exists with given label and property"""
        cypher = f"""
        MATCH (n:{label} {{{property_name}: $value}})
        RETURN count(n) as count
        """
        result = self.graph.query(cypher, {'value': property_value})
        return result[0]['count'] > 0

    def _create_relationship(self, from_label: str, to_label: str, 
                           from_prop: Dict, to_prop: Dict, 
                           rel_type: str, rel_props: Dict = None):
        """Create relationship between nodes"""
        cypher = f"""
        MATCH (a:{from_label})
        WHERE {' AND '.join(f'a.{k} = ${k}' for k in from_prop.keys())}
        MATCH (b:{to_label})
        WHERE {' AND '.join(f'b.{k} = ${k}2' for k in to_prop.keys())}
        MERGE (a)-[r:{rel_type}]->(b)
        """
        if rel_props:
            cypher += f"\nSET r += $rel_props"
        
        params = {
            **from_prop,
            **{f'{k}2': v for k, v in to_prop.items()},
            'rel_props': rel_props or {}
        }
        
        self.graph.query(cypher, params)

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {
            'processed_count': len(self.processed_items),
            'total_nodes': self._get_node_count(),
            'batch_size': self.batch_size
        }

    # @abstractmethod
    # def process_dataset(self, file_paths: Dict):
    #     """Process the dataset"""
    #     pass

    @abstractmethod
    def create_indexes(self):
        """Create database indexes"""
        pass

    # @abstractmethod
    # def validate_data(self, data):
    #     """Validate input data"""
    #     pass

    # @abstractmethod
    # def preprocess_data(self, data):
    #     """Preprocess data before insertion"""
    #     pass

class DatabaseMixin:
    """Shared database operations for processors"""
    
    def _get_concept_info(self, term: str) -> dict:
        """Get UMLS concept information for a term"""
        cypher = """
        MATCH (c:Concept)
        WHERE c.term CONTAINS $term
        RETURN c
        LIMIT 1
        """
        try:
            result = self.graph.query(cypher, {'term': term})
            if result:
                return result[0]['c']
            return None
        except Exception as e:
            logger.error(f"Error getting concept info for {term}: {str(e)}")
            return None

    def _get_top_relationships(self, cui: str, limit: int = 25) -> list:
        """Get top N relationships for a concept"""
        cypher = """
        MATCH (c1:Concept {cui: $cui})-[r]->(c2:Concept)
        WHERE type(r) IN ['broader_than', 'child_of', 'HAS_DEFINITION', 'narrower_than', 'parent_of', 'synonym_of']
        RETURN c2.term as related_term, type(r) as type
        LIMIT $limit
        """
        try:
            results = self.graph.query(cypher, {'cui': cui, 'limit': limit})
            return [{'related_term': result['related_term'], 
                    'type': result['type']} 
                    for result in results]
        except Exception as e:
            logger.error(f"Error getting relationships for {cui}: {str(e)}")
            return []

    def _get_definitions(self, cui: str) -> list:
        """Get definitions for a concept"""
        cypher = """
        MATCH (c:Concept {cui: $cui})-[:HAS_DEFINITION]->(d:Definition)
        RETURN d
        """
        try:
            return self.graph.query(cypher, {'cui': cui})
        except Exception as e:
            logger.error(f"Error getting definitions for {cui}: {str(e)}")
            return []

    @abstractmethod
    def _process_batch(self, batch, cypher_query):
        """Process a batch of data"""
        pass

    @abstractmethod
    def _handle_failed_batch(self, batch):
        """Handle failed batch processing"""
        pass