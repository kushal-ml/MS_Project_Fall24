import requests
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UMLSSearcher:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("UMLS API key not found in environment variables")
            
        self.base_uri = 'https://uts-ws.nlm.nih.gov/rest'
        self.version = 'current'
        
        # Default vocabularies and term types
        self.default_sabs = 'MSH,SNOMEDCT_US,RXNORM'  # Medical Subject Headings, SNOMED CT, RxNorm
        self.default_ttys = 'PT,SY,IN'  # Preferred terms, Synonyms, Instance
        
    async def search_terms(self, 
                         terms: List[str], 
                         sabs: str = None, 
                         ttys: str = None) -> Dict[str, List[Dict]]:
        """
        Search UMLS for a list of medical terms
        
        Args:
            terms: List of medical terms to search
            sabs: Comma-separated list of source vocabularies
            ttys: Comma-separated list of term types
        """
        try:
            results = {}
            
            for term in terms:
                term_results = []
                page = 0
                
                while True:
                    page += 1
                    path = f'/search/{self.version}'
                    
                    query = {
                        'string': term,
                        'apiKey': self.api_key,
                        'rootSource': sabs or self.default_sabs,
                        'termType': ttys or self.default_ttys,
                        'pageNumber': page
                    }
                    
                    response = requests.get(
                        self.base_uri + path,
                        params=query
                    )
                    response.encoding = 'utf-8'
                    
                    if response.status_code != 200:
                        logger.error(f"Error searching term '{term}': {response.status_code}")
                        break
                        
                    data = response.json()
                    current_results = data['result']['results']
                    
                    if not current_results:
                        if page == 1:
                            logger.warning(f"No results found for term: {term}")
                        break
                        
                    for item in current_results:
                        term_results.append({
                            'ui': item['ui'],
                            'name': item['name'],
                            'uri': item['uri'],
                            'source': item['rootSource'],
                            'term_type': item.get('termType', ''),
                            'score': item.get('score', 0)
                        })
                
                results[term] = term_results
                
            return results
            
        except Exception as e:
            logger.error(f"Error in UMLS search: {str(e)}")
            raise

    async def get_concept_info(self, cui: str) -> Dict[str, Any]:
        """Get detailed information about a UMLS concept"""
        try:
            path = f'/content/{self.version}/CUI/{cui}'
            response = requests.get(
                self.base_uri + path,
                params={'apiKey': self.api_key}
            )
            
            if response.status_code != 200:
                logger.error(f"Error getting concept info for CUI {cui}: {response.status_code}")
                return None
                
            return response.json()['result']
            
        except Exception as e:
            logger.error(f"Error getting concept info: {str(e)}")
            return None

    async def get_complete_concept_info(self, cui: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a concept including:
        - Basic concept info
        - Definitions
        - Relationships
        - Semantic types
        """
        try:
            concept_info = {
                'basic_info': await self.get_concept_info(cui),
                'definitions': await self.get_concept_definitions(cui),
                'relationships': await self.get_concept_relationships(cui),
                'semantic_types': await self.get_semantic_types(cui),
                'atoms': await self.get_concept_atoms(cui)
            }
            return concept_info
            
        except Exception as e:
            logger.error(f"Error getting complete concept info for CUI {cui}: {str(e)}")
            return None

    async def get_concept_definitions(self, cui: str) -> List[Dict[str, Any]]:
        """Get all definitions for a concept"""
        try:
            path = f'/content/{self.version}/CUI/{cui}/definitions'
            response = requests.get(
                self.base_uri + path,
                params={'apiKey': self.api_key}
            )
            
            if response.status_code != 200:
                logger.error(f"Error getting definitions for CUI {cui}: {response.status_code}")
                return []
                
            results = response.json()['result']
            return [
                {
                    'value': definition.get('value', ''),
                    'source': definition.get('rootSource', ''),
                    'context': definition.get('context', '')
                }
                for definition in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting definitions: {str(e)}")
            return []

    async def get_concept_relationships(self, cui: str) -> List[Dict[str, Any]]:
        """Get all relationships for a concept"""
        try:
            # Updated REST API endpoint with apiKey as query parameter
            path = f'/content/{self.version}/CUI/{cui}/relations'
            logger.info(f"Fetching relationships from: {self.base_uri + path}")
            
            # Simplified request with apiKey in query parameters only
            params = {
                'apiKey': self.api_key,  # API key as query parameter
                'pageSize': 100  # Get more results per page
            }
            
            response = requests.get(
                self.base_uri + path,
                params=params  # Pass API key in query parameters
            )
            
            # Debug response
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Error getting relationships for CUI {cui}: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return []
            
            json_response = response.json()
            if not json_response.get('result'):
                logger.warning(f"No relationships found for CUI {cui}")
                return []
            
            results = json_response['result']
            relationships = []
            
            for rel in results:
                # Extract related concept details
                related_concept = rel.get('relatedConcept', {})
                relationship_label = rel.get('relationLabel', '')
                additional_label = rel.get('additionalRelationLabel', '')
                
                # Skip empty relationships
                if not related_concept or not relationship_label:
                    continue
                
                # Build relationship dictionary with more detailed information
                relationship = {
                    'relationship_type': relationship_label,
                    'related_concept': {
                        'cui': related_concept.get('ui', ''),
                        'name': related_concept.get('name', ''),
                        'source': related_concept.get('rootSource', '')
                    },
                    'source_of_relationship': rel.get('rootSource', ''),
                    'additional_type': additional_label,
                    'rela': rel.get('rela', '')
                }
                
                # Only add relationships that have both type and related concept
                if relationship['relationship_type'] and relationship['related_concept']['cui']:
                    relationships.append(relationship)
            
            # Sort relationships by type for better organization
            relationships.sort(key=lambda x: (x['relationship_type'], x['related_concept']['name']))
            
            logger.info(f"Successfully retrieved {len(relationships)} relationships for CUI {cui}")
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            logger.error(f"Full URL: {self.base_uri + path}")
            return []

    async def get_semantic_types(self, cui: str) -> List[Dict[str, Any]]:
        """Get semantic types for a concept"""
        try:
            path = f'/semantic-network/current/TUI/T109'
            logger.info(f"Attempting to access: {self.base_uri + path}")
            
            response = requests.get(
                self.base_uri + path,
                params={'apiKey': self.api_key}
            )
            
            if response.status_code != 200:
                logger.error(f"Error getting semantic types for CUI {cui}: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return []
            
            # Debug the response
            logger.info(f"Response content: {response.text}")
            
            # Parse JSON response
            json_response = response.json()
            logger.info(f"JSON Response type: {type(json_response)}")
            logger.info(f"JSON Response content: {json_response}")
            
            # Check if response is a string (error case)
            if isinstance(json_response, str):
                logger.error(f"Received string response instead of JSON object: {json_response}")
                return []
            
            # Get results safely
            results = json_response.get('result', [])
            if not results:
                logger.warning("No results found in response")
                return []
            
            return [
                {
                    'semantic_type': st.get('name', ''),
                    'type_ui': st.get('ui', ''),
                    'uri': st.get('uri', '')
                }
                for st in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting semantic types: {str(e)}")
            logger.error(f"Full URL: {self.base_uri + path}")
            return []

    async def get_concept_atoms(self, cui: str) -> List[Dict[str, Any]]:
        """Get atoms (terms and codes) for a concept - English only"""
        try:
            path = f'/content/{self.version}/CUI/{cui}/atoms'
            response = requests.get(
                self.base_uri + path,
                params={
                    'apiKey': self.api_key,
                    'language': 'ENG'  # Filter for English language
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error getting atoms for CUI {cui}: {response.status_code}")
                return []
                
            results = response.json()['result']
            
            # Filter for English terms and preferred sources
            preferred_sources = {'SNOMEDCT_US', 'MSH', 'RXNORM'}
            english_atoms = [
                {
                    'name': atom.get('name', ''),
                    'term_type': atom.get('termType', ''),
                    'source': atom.get('rootSource', ''),
                    'code': atom.get('code', ''),
                    'language': atom.get('language', ''),
                    'is_preferred': atom.get('termType') == 'PF'  # Preferred term flag
                }
                for atom in results
                if atom.get('language') == 'ENG'  # Only English terms
            ]
            
            # Sort by preferred terms and sources
            english_atoms.sort(
                key=lambda x: (
                    x['is_preferred'],
                    x['source'] in preferred_sources,
                    x['source'],
                    x['name']
                ),
                reverse=True
            )
            
            return english_atoms
            
        except Exception as e:
            logger.error(f"Error getting atoms: {str(e)}")
            return []

    async def search_and_get_info(self, term: str) -> List[Dict[str, Any]]:
        """
        Search for a term and get complete information for all matching concepts
        """
        try:
            # First search for the term to get CUIs
            path = f'/search/{self.version}'
            query = {
                'string': term,
                'apiKey': self.api_key,
                'rootSource': self.default_sabs,
                'termType': self.default_ttys,
                'pageNumber': 1,
                'searchType': 'exact'  # Add exact matching
            }
            
            # Debug search request
            logger.info(f"Searching term: {term}")
            logger.info(f"Search URL: {self.base_uri + path}")
            
            response = requests.get(
                self.base_uri + path,
                params=query
            )
            
            if response.status_code != 200:
                logger.error(f"Error searching term '{term}': {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return []
                
            search_results = response.json()['result']['results']
            
            if not search_results:
                logger.warning(f"No results found for term: {term}")
                return []
                
            # Debug search results
            logger.info(f"Found {len(search_results)} results")
            for result in search_results[:3]:  # Show first 3 results
                logger.info(f"CUI: {result['ui']}, Name: {result['name']}, Score: {result.get('score')}")
            
            # Get complete information for each matching concept
            concept_results = []
            for result in search_results[:5]:  # Limit to top 5 matches
                cui = result['ui']
                concept_info = await self.get_complete_concept_info(cui)
                if concept_info:
                    concept_info['search_score'] = result.get('score', 0)
                    concept_results.append(concept_info)
                    
            return concept_results
            
        except Exception as e:
            logger.error(f"Error in search_and_get_info: {str(e)}")
            return []

def _print_relationships(relationships: List[Dict[str, Any]]):
    """Helper function to print relationships in a more readable format"""
    if not relationships:
        print("No relationships found")
        return
        
    # Group relationships by type
    grouped_rels = {}
    for rel in relationships:
        rel_type = rel['relationship_type']
        if rel_type not in grouped_rels:
            grouped_rels[rel_type] = []
        grouped_rels[rel_type].append(rel)
    
    # Print relationships in a structured format
    print("\nRelationships:")
    print("-" * 50)
    
    # Define relationship type descriptions
    rel_descriptions = {
        'RB': 'Broader',
        'RN': 'Narrower',
        'RO': 'Other',
        'PAR': 'Parent',
        'CHD': 'Child',
        'SIB': 'Sibling',
        'RQ': 'Related, Possibly Synonymous',
        'SY': 'Synonym',
        'QB': 'Can Be Qualified By',
        'AQ': 'Allowed Qualifier'
    }
    
    for rel_type, rels in grouped_rels.items():
        # Get description for relationship type
        description = rel_descriptions.get(rel_type, rel_type)
        print(f"\n{description} Relationships:")
        
        for rel in rels:
            related = rel['related_concept']
            # Add RELA information if available
            rela = f" [{rel['rela']}]" if rel['rela'] else ""
            # Add source information
            source = f" ({rel['source_of_relationship']})" if rel['source_of_relationship'] else ""
            
            print(f"  • {related['name']}")
            print(f"    CUI: {related['cui']}{rela}{source}")

def _print_concept_info(concept_info: Dict[str, Any], index: int = None):
    """Helper function to print concept information in a structured format"""
    try:
        basic_info = concept_info['basic_info']
        prefix = f"Concept {index + 1}: " if index is not None else "Concept: "
        
        print(f"\n{prefix}{basic_info.get('name', '')}")
        print("=" * 50)
        print(f"CUI: {basic_info.get('ui', '')}")
        
        # Definitions
        if concept_info['definitions']:
            print("\nDefinitions:")
            print("-" * 50)
            for def_info in concept_info['definitions'][:2]:  # Show top 2 definitions
                source = def_info['source']
                print(f"[{source}]")
                print(f"{def_info['value']}\n")
        
        # Semantic Types
        if concept_info['semantic_types']:
            print("Semantic Types:")
            print("-" * 50)
            for st in concept_info['semantic_types']:
                print(f"• {st['semantic_type']}")
        
        # Relationships
        if concept_info['relationships']:
            _print_relationships(concept_info['relationships'])
            
        # Terms and Codes
        if concept_info['atoms']:
            print("\nAlternative Terms and Codes:")
            print("-" * 50)
            shown_terms = set()
            count = 0
            
            for atom in concept_info['atoms']:
                term = atom['name']
                if (term not in shown_terms and 
                    count < 5 and  # Show top 5 terms
                    atom['source'] in {'SNOMEDCT_US', 'MSH', 'RXNORM'}):
                    print(f"• {term}")
                    print(f"  {atom['source']}: {atom['code']}")
                    shown_terms.add(term)
                    count += 1
                    
    except Exception as e:
        logger.error(f"Error printing concept info: {str(e)}")

# Update main function to use term-based search
def main():
    """Test the UMLS searcher with term-based search"""
    try:
        searcher = UMLSSearcher()
        
        while True:
            # Get search term from user
            term = input("\nEnter a medical term to search (or 'quit' to exit): ").strip()
            
            if term.lower() == 'quit':
                break
                
            if not term:
                print("Please enter a valid term")
                continue
                
            print(f"\nSearching for: {term}")
            print("-" * 50)
            
            # Get and display results
            import asyncio
            results = asyncio.run(searcher.search_and_get_info(term))
            
            if not results:
                print("No results found")
                continue
                
            # Print each concept's information
            for i, concept_info in enumerate(results):
                _print_concept_info(concept_info, i)
                
            print("\n" + "="*50 + "\n")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        print("\nSearch session ended")

if __name__ == "__main__":
    main() 