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