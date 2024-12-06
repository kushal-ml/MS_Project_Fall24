from neo4j import GraphDatabase
import os
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UMLSProcessor:
    def __init__(self, graph):
        self.graph = graph
        self.node_limit = 190000  
        self.processed_count = 0
        self.processed_concepts = set()
        self.processed_stns = set()  # Track processed STNs

        # Relationship types
        self.important_relations = {
            'RO': 'has_occurrence',
            'RN': 'narrower_than',
            'RB': 'broader_than',
            'PAR': 'parent_of',
            'CHD': 'child_of',
            'SIB': 'sibling_of',
            'AQ': 'allowed_qualifier',
            'QB': 'can_be_qualified_by',
            'MOA': 'mechanism_of_action',
            'COS': 'causes',
            'TRE': 'treats',
            'PRE': 'presents_with'
        }

        self.usmle_domains = {
            'SNOMED_CT': {
                'priority_1': {
                    'DISO': 'Disease',
                    'FIND': 'Clinical_Finding',
                    'PROC': 'Procedure',
                    'SYMP': 'Symptom',
                    'DGNS': 'Diagnosis',
                    'PATH': 'Pathology',
                    'TREAT': 'Treatment',
                    'CRIT': 'Critical_Care',
                    'EMER': 'Emergency'
                },
                'priority_2': {
                    'ANAT': 'Anatomy',
                    'CHEM': 'Biochemistry',
                    'PHYS': 'Physiology',
                    'MICR': 'Microbiology',
                    'IMMUN': 'Immunology',
                    'PHARM': 'Pharmacology',
                    'GENE': 'Genetics',
                    'LAB': 'Laboratory',
                    'RAD': 'Radiology',
                    'CELL': 'Cell_Biology',
                    'MOLC': 'Molecular_Biology',
                    'NUTRI': 'Nutrition'
                }
            },
            'MSH': {
                'priority_1': {
                    'D27': 'Drug_Classes',
                    'C': 'Diseases',
                    'F03': 'Mental_Disorders',
                    'C23': 'Pathological_Conditions',
                    'E01': 'Diagnostic_Techniques',
                    'G02': 'Biological_Phenomena',
                    'D26': 'Pharmaceutical_Preparations',
                    'C13': 'Female_Urogenital_Diseases',
                    'C12': 'Male_Urogenital_Diseases',
                    'C14': 'Cardiovascular_Diseases',
                    'C08': 'Respiratory_Diseases',
                    'C06': 'Digestive_Diseases',
                    'C10': 'Nervous_System_Diseases',
                    'C19': 'Endocrine_Diseases',
                    'C15': 'Hemic_Diseases',
                    'C20': 'Immune_Diseases'
                },
                'priority_2': {
                    'A': 'Anatomy',
                    'G': 'Biological_Sciences',
                    'B': 'Organisms',
                    'D': 'Chemicals_Drugs',
                    'E02': 'Therapeutics',
                    'F02': 'Psychological_Phenomena',
                    'N02': 'Facilities_Services',
                    'E03': 'Anesthesia_Analgesia',
                    'E04': 'Surgical_Procedures',
                    'G03': 'Metabolism',
                    'G04': 'Cell_Physiology',
                    'G05': 'Genetic_Processes'
                }
            },
            'RXNORM': {
                'priority_1': {
                    'IN': 'Ingredient',
                    'PIN': 'Precise_Ingredient',
                    'BN': 'Brand_Name',
                    'SCDC': 'Clinical_Drug',
                    'DF': 'Dose_Form',
                    'DFG': 'Dose_Form_Group',
                    'PSN': 'Prescribable_Name',
                    'SBD': 'Semantic_Branded_Drug',
                    'SCD': 'Semantic_Clinical_Drug'
                },
                'priority_2': {
                    'BPCK': 'Brand_Pack',
                    'GPCK': 'Generic_Pack',
                    'SY': 'Synonym',
                    'TMSY': 'Tall_Man_Lettering',
                    'ET': 'Entry_Term'
                }
            },
            'LOINC': {
                'priority_1': {
                    'LP': 'Laboratory_Procedure',
                    'LPRO': 'Laboratory_Protocol',
                    'LG': 'Laboratory_Group',
                    'LX': 'Laboratory_Index',
                    'LB': 'Laboratory_Battery',
                    'LC': 'Laboratory_Class'
                },
                'priority_2': {
                    'PANEL': 'Laboratory_Panel',
                    'COMP': 'Component',
                    'PROC': 'Procedure',
                    'SPEC': 'Specimen'
                }
            }
        }

    
    def _create_semantic_type_batch(self, batch):
        """Create Semantic Types and Tree Numbers in the graph"""
        try:
            cypher = """
            UNWIND $batch AS item
            MERGE (st:SemanticType {tui: item.tui})
            SET st.name = item.sty,
                st.tree_number = item.stn
            MERGE (c:Concept {cui: item.cui})
            MERGE (c)-[r:HAS_SEMANTIC_TYPE]->(st)
            """
            self.graph.query(cypher, {'batch': batch})
            # Mark STNs as processed
            self.processed_stns.update(item['stn'] for item in batch)
        except Exception as e:
            logger.error(f"Error creating semantic type batch: {str(e)}")
            raise


    def _load_processed_concepts(self):
        """Load processed concepts from the graph into a set."""
        try:
            cypher = "MATCH (c:Concept) RETURN c.cui"
            result = self.graph.query(cypher)
            self.processed_concepts = {record['c.cui'] for record in result}
            logger.info(f"Loaded {len(self.processed_concepts)} processed concepts.")
        except Exception as e:
            logger.error(f"Error loading processed concepts: {str(e)}")
            raise    

    def _create_relationships_batch(self, batch):
        """Create relationships in batches."""
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
      
    def _create_definitions_batch(self, batch):
        """Create definitions in batches."""
        try:
            cypher = """
            UNWIND $batch AS def_item
            MATCH (c:Concept {cui: def_item.cui})
            MERGE (d:Definition {id: def_item.def_id})
            SET d.text = def_item.text,
                d.source = def_item.source
            MERGE (c)-[r:HAS_DEFINITION]->(d)
            """
            self.graph.query(cypher, {'batch': batch})
        except Exception as e:
            logger.error(f"Error creating definitions batch: {str(e)}")
            raise

    def process_mrconso(self, file_path):
        """Process MRCONSO.RRF with USMLE focus"""
        try:
            current_count = self._get_node_count()
            if current_count >= self.node_limit:
                logger.warning(f"Node limit reached: {current_count} nodes")
                return

            remaining_nodes = self.node_limit - current_count
            logger.info(f"Available nodes: {remaining_nodes}")

            batch = []
            processed = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if processed >= remaining_nodes:
                        logger.warning("Node limit would be exceeded. Stopping.")
                        break

                    fields = line.strip().split('|')
                    
                    # Process only English terms
                    if fields[1] != "ENG":
                        continue

                    source = fields[11]  # SAB field
                    tty = fields[12]     # Term Type
                    
                    # Check domain priority
                    domain_info = self._get_domain_priority(source, tty)
                    if not domain_info:
                        continue

                    concept = {
                        'cui': fields[0],
                        'term': fields[14],
                        'source': source,
                        'domain': domain_info['domain'],
                        'priority': domain_info['priority']
                    }
                    
                    # Only process priority 1 if space is limited
                    if remaining_nodes < 50000 and domain_info['priority'] != 'priority_1':
                        continue

                    batch.append(concept)
                    processed += 1

                    if len(batch) >= 500:  # Smaller batch size
                        self._create_concepts_batch(batch)
                        logger.info(f"Processed {processed} concepts")
                        batch = []

                if batch:
                    self._create_concepts_batch(batch)

            logger.info(f"Total concepts processed: {processed}")

        except Exception as e:
            logger.error(f"Error processing MRCONSO: {str(e)}")
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

    def _create_concepts_batch(self, batch):
        """Create concepts in batch"""
        cypher = """
        UNWIND $batch as concept
        MERGE (c:Concept {cui: concept.cui})
        SET c.term = concept.term,
            c.source = concept.source,
            c.domain = concept.domain,
            c.priority = concept.priority
        """
        self.graph.query(cypher, {'batch': batch})

    def _get_node_count(self):
        """Get current node count"""
        cypher = "MATCH (n) RETURN count(n) as count"
        result = self.graph.query(cypher)
        return result[0]['count']

    def create_indexes(self):
        """Create necessary indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.cui)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.priority)"
        ]
        for index in indexes:
            self.graph.query(index)


    def process_mrrel(self, file_path):
      """Process MRREL with smaller batches and transaction control"""
      logger.info("Processing MRREL.RRF...")
      try:
          # Load processed concepts
          self._load_processed_concepts()
          
          batch = []
          processed = 0
          skipped = 0
          batch_size = 50  # Reduced batch size
          max_relationships = 200000  # Limit total relationships
          
          with open(file_path, 'r', encoding='utf-8') as f:
              for line in f:
                  # Check if we've hit the relationship limit
                  if processed >= max_relationships:
                      logger.warning(f"Reached maximum relationship limit of {max_relationships}")
                      break

                  fields = line.strip().split('|')
                  cui1 = fields[0]
                  cui2 = fields[4]
                  rel_type = fields[3]

                  # Skip if concepts not in our processed set
                  if cui1 not in self.processed_concepts or \
                    cui2 not in self.processed_concepts:
                      skipped += 1
                      continue

                  # Skip non-important relationships
                  if rel_type not in self.important_relations:
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
                  
                  if len(batch) >= batch_size:
                      try:
                          self._create_relationships_batch(batch)
                          logger.info(f"Processed {processed} relationships")
                          batch = []
                      except Exception as e:
                          logger.error(f"Batch processing error: {str(e)}")
                          # Process one by one if batch fails
                          self._process_relationships_individually(batch)
                          batch = []
                          
              if batch:
                  try:
                      self._create_relationships_batch(batch)
                  except Exception as e:
                      logger.error(f"Final batch processing error: {str(e)}")
                      self._process_relationships_individually(batch)
                      
          logger.info(f"Total relationships processed: {processed}")
          logger.info(f"Skipped relationships: {skipped}")
                      
      except Exception as e:
          logger.error(f"Error processing MRREL: {str(e)}")
          raise

    def _process_relationships_individually(self, relationships):
      """Process relationships one at a time when batch fails"""
      for rel in relationships:
          try:
              cypher = """
              MATCH (c1:Concept {cui: $cui1})
              MATCH (c2:Concept {cui: $cui2})
              MERGE (c1)-[r:RELATES_TO {
                  type: $rel_type,
                  source: $source
              }]->(c2)
              """
              self.graph.query(cypher, rel)
          except Exception as e:
              logger.error(f"Error processing individual relationship {rel['cui1']}->{rel['cui2']}: {str(e)}")

    def process_mrdef(self, file_path):
      """Process MRDEF with controlled batch size"""
      logger.info("Processing MRDEF.RRF...")
      try:
          batch = []
          processed = 0
          batch_size = 50  # Reduced batch size
          max_definitions = 50000  # Limit total definitions
          
          with open(file_path, 'r', encoding='utf-8') as f:
              for line in f:
                  # Check if we've hit the definition limit
                  if processed >= max_definitions:
                      logger.warning(f"Reached maximum definition limit of {max_definitions}")
                      break

                  fields = line.strip().split('|')
                  cui = fields[0]
                  
                  # Skip if concept not in our processed set
                  if cui not in self.processed_concepts:
                      continue
                          
                  definition = {
                      'cui': cui,
                      'source': fields[4],
                      'text': fields[5],
                      'def_id': f"DEF_{cui}_{fields[4]}"
                  }
                  batch.append(definition)
                  processed += 1
                  
                  if len(batch) >= batch_size:
                      try:
                          self._create_definitions_batch(batch)
                          logger.info(f"Processed {processed} definitions")
                          batch = []
      
                      except Exception as e:
                          logger.error(f"Batch processing error: {str(e)}")
                          # Process one by one if batch fails
                          self._process_definitions_individually(batch)
                          batch = []
                          
              if batch:
                  try:
                      self._create_definitions_batch(batch)
                  except Exception as e:
                      logger.error(f"Final batch processing error: {str(e)}")
                      self._process_definitions_individually(batch)
                      
          logger.info(f"Total definitions processed: {processed}")
                      
      except Exception as e:
          logger.error(f"Error processing MRDEF: {str(e)}")
          raise

    def _process_definitions_individually(self, definitions):
      """Process definitions one at a time when batch fails"""
      for def_item in definitions:
          try:
              cypher = """
              MATCH (c:Concept {cui: $cui})
              MERGE (d:Definition {id: $def_id})
              SET d.text = $text,
                  d.source = $source
              MERGE (c)-[r:HAS_DEFINITION]->(d)
              """
              self.graph.query(cypher, def_item)
          except Exception as e:
              logger.error(f"Error processing individual definition for {def_item['cui']}: {str(e)}")
    
    def process_mrsty(self, file_path):
        """Process MRSTY.RRF to extract Semantic Tree Numbers and Types"""
        logger.info("Processing MRSTY.RRF...")
        try:
            batch = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    fields = line.strip().split('|')
                    cui = fields[0]  # Concept Unique Identifier
                    tui = fields[1]  # Type Unique Identifier
                    stn = fields[2]  # Semantic Tree Number
                    sty = fields[3]  # Semantic Type Name
                    
                    if stn in self.processed_stns:
                        continue  # Skip already processed STNs

                    semantic_type = {
                        'cui': cui,
                        'tui': tui,
                        'stn': stn,
                        'sty': sty
                    }
                    batch.append(semantic_type)

                    # Process in batches
                    if len(batch) >= 500:
                        self._create_semantic_type_batch(batch)
                        batch = []

                if batch:
                    self._create_semantic_type_batch(batch)

            logger.info("Finished processing MRSTY.RRF.")
        except Exception as e:
            logger.error(f"Error processing MRSTY.RRF: {str(e)}")
            raise


def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Neo4j connection
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    # Initialize processor
    processor = UMLSProcessor(graph)
    
    # Define file paths
    base_path = Path("data")
    files = {
        'mrconso': base_path / "MRCONSO.RRF",
        'mrrel': base_path / "MRREL.RRF",
        'mrdef': base_path / "MRDEF.RRF",
        'mrsty': base_path / "MRSTY.RRF"
    }
    
    # Process files
    try:
        processor.process_mrconso(files['mrconso'])
        processor.process_mrrel(files['mrrel'])
        processor.process_mrdef(files['mrdef'])
        processor.process_mrsty(files['mrsty']) 
        logger.info("UMLS processing completed successfully")
    except Exception as e:
        logger.error(f"Error during UMLS processing: {str(e)}")

if __name__ == "__main__":
    main()
