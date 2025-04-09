import httpx
import logging
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
logger = logging.getLogger(__name__)


class UMLSAPIProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('API_KEY')
        self.base_uri = 'https://uts-ws.nlm.nih.gov'
        self.version = 'current'
        self.client = httpx.AsyncClient(timeout=30.0)
        self.RELATION_MAP = {
        'RO': 'related_to',
        'CHD': 'child_of',
        'PAR': 'parent_of',
        'SY': 'synonym',
        'RB': 'broader_than',
        'RN': 'narrower_than',
        'causes': 'causes',  # From rela parameter
        'treats': 'treats'
    }
        if not self.api_key:
            raise ValueError("UMLS API key is required")
        
        logger.info("UMLS API processor initialized")

    async def search_and_get_info(self, term: str) -> List[Dict[str, Any]]:
        """Search UMLS for a term and get complete information, including concepts."""
        try:
            search_results = await self._search_term(term)
            if not search_results:
                return []

            detailed_results = []
            for result in search_results[:3]:  # Limit to top 3 matches
                cui = result.get('ui')
                if cui:
                    # Fetch concept details, definitions, semantic types, and relationships
                    concept_details, definitions, semantic_types, relationships = await asyncio.gather(
                        self._get_concept_details(cui),  # Fetch concept details
                        self._get_definitions(cui),
                        self._get_semantic_types(cui),
                        self._get_concept_relationships(cui)
                    )
                    
                    # Filter definitions for English sources
                    english_sources = {
                        'NCI', 'MSH', 'SNOMEDCT_US', 'CSP', 'MTH', 'NIC', 
                        'NOC', 'NDFRT', 'AOD', 'CHV', 'MEDLINEPLUS', 'HPO'
                    }
                    english_definitions = [
                        d for d in definitions 
                        if d.get('rootSource') in english_sources
                    ]
                    
                    # Combine all information into a single concept info dictionary
                    concept_info = {
                        'concept_details': concept_details,  # Add concept details
                        'basic_info': result,
                        'definitions': english_definitions,
                        'semantic_types': semantic_types,
                        'relationships': relationships
                    }
                    detailed_results.append(concept_info)
                    logger.info(f"Retrieved detailed info for concept: {cui}")

            return detailed_results

        except Exception as e:
            logger.error(f"Error in search_and_get_info: {str(e)}")
            return []

    async def _search_term(self, term: str) -> List[Dict]:
        """Search for a term in UMLS and return matching concepts."""
        try:
            path = f'/rest/search/{self.version}'
            params = {
                'string': term,
                'apiKey': self.api_key,
                'searchType': 'words',
                'pageSize': 5,
                'sabs': 'MSH,SNOMEDCT_US',  # Restrict to reliable sources
                'rela': 'causes,treats',
                'pageNumber': 1
            }

            logger.info(f"Searching UMLS for term: {term}")
            response = await self.client.get(
                self.base_uri + path,
                params=params
            )
            response.raise_for_status()

            results = response.json()['result']['results']
            logger.info(f"Found {len(results)} results for term: {term}")
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error searching term '{term}': {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Error in _search_term: {str(e)}")
            return []

    async def _get_concept_details(self, cui: str) -> Dict[str, Any]:
        """Get detailed information about a concept by its CUI."""
        try:
            path = f'/rest/content/{self.version}/CUI/{cui}'
            response = await self.client.get(
                self.base_uri + path,
                params={'apiKey': self.api_key}
            )
            response.raise_for_status()
            return response.json()['result']
        except httpx.HTTPStatusError:
            return {}
        except Exception as e:
            logger.error(f"Error getting concept details for CUI {cui}: {str(e)}")
            return {}

    async def _get_definitions(self, cui: str) -> List[Dict]:
        """Get definitions for a concept."""
        try:
            path = f'/rest/content/{self.version}/CUI/{cui}/definitions'
            response = await self.client.get(
                self.base_uri + path,
                params={'apiKey': self.api_key}
            )
            response.raise_for_status()
            return response.json()['result']
        except httpx.HTTPStatusError:
            return []
        except Exception as e:
            logger.error(f"Error getting definitions for CUI {cui}: {str(e)}")
            return []

    async def _get_semantic_types(self, cui: str) -> List[Dict]:
        """Get semantic types for a concept."""
        try:
            path = f'/rest/content/{self.version}/CUI/{cui}/semantictypes'
            response = await self.client.get(
                self.base_uri + path,
                params={'apiKey': self.api_key}
            )
            response.raise_for_status()
            return response.json()['result']
        except httpx.HTTPStatusError:
            return []
        except Exception as e:
            logger.error(f"Error getting semantic types for CUI {cui}: {str(e)}")
            return []

    # In UMLSAPIProcessor class
    async def _get_concept_relationships(self, cui: str) -> List[Dict]:
        """Get relationships for a single CUI from UMLS API."""
        try:
            # Make sure cui is a single string value, not a dict or complex object
            if not isinstance(cui, str):
                logger.error(f"Invalid CUI type: {type(cui)}. Expected string.")
                return []
            
            path = f'/rest/content/{self.version}/CUI/{cui}/relations'
            params = {
                'apiKey': self.api_key,
                'pageSize': 50,
                'includeAdditionalRelationLabels': 'true'
            }
            
            response = await self.client.get(self.base_uri + path, params=params)
            if response.status_code == 404:
                logger.warning(f"No relationships found for CUI {cui}")
                return []  # Graceful handling
            
            response.raise_for_status()
            results = response.json().get('result', [])
            formatted_results = []

            for result in results:
                relationship = {
                    'sourceName': result.get('relatedFromIdName', 'Unknown'),
                    'sourceUi': result.get('relatedFromId', '').split('/')[-1],
                    'relationLabel': self.RELATION_MAP.get(result.get('relationLabel'), result.get('relationLabel')),
                    'relatedName': result.get('relatedIdName', 'Unknown'),
                    'relatedId': result.get('relatedId', '').split('/')[-1]
                }
                formatted_results.append(relationship)

            return formatted_results

        except Exception as e:
            logger.error(f"Error getting relationships for CUI {cui}: {str(e)}")
            return []

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.info("UMLS API client closed")