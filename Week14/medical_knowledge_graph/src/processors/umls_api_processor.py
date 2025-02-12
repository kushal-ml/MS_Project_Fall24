import requests
import logging
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class UMLSAPIProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('API_KEY')
        self.base_uri = 'https://uts-ws.nlm.nih.gov'
        self.version = 'current'
        
        if not self.api_key:
            raise ValueError("UMLS API key is required")
        
        logger.info("UMLS API processor initialized")

    async def search_and_get_info(self, term: str) -> List[Dict[str, Any]]:
        """Search UMLS for a term and get complete information"""
        try:
            # First search for the term
            search_results = await self._search_term(term)
            if not search_results:
                return []

            # Get detailed information for each concept
            detailed_results = []
            for result in search_results[:3]:  # Limit to top 3 matches
                cui = result.get('ui')
                if cui:
                    concept_info = {
                        'basic_info': result,
                        'definitions': await self._get_definitions(cui),
                        'semantic_types': await self._get_semantic_types(cui),
                        'relationships': await self.get_concept_relationships(cui)
                    }
                    detailed_results.append(concept_info)
                    logger.info(f"Retrieved detailed info for concept: {cui}")

            return detailed_results

        except Exception as e:
            logger.error(f"Error in search_and_get_info: {str(e)}")
            return []

    async def _search_term(self, term: str) -> List[Dict]:
        """Search for a term in UMLS"""
        try:
            path = f'/rest/search/{self.version}'
            params = {
                'string': term,
                'apiKey': self.api_key,
                'searchType': 'exact'
            }

            logger.info(f"Searching UMLS for term: {term}")
            response = requests.get(
                self.base_uri + path,
                params=params
            )

            if response.status_code != 200:
                logger.error(f"Error searching term '{term}': {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []

            results = response.json()['result']['results']
            logger.info(f"Found {len(results)} results for term: {term}")
            return results

        except Exception as e:
            logger.error(f"Error in _search_term: {str(e)}")
            return []

    async def _get_definitions(self, cui: str) -> List[Dict]:
        """Get definitions for a concept"""
        try:
            path = f'/rest/content/{self.version}/CUI/{cui}/definitions'
            params = {'apiKey': self.api_key}

            response = requests.get(
                self.base_uri + path,
                params=params
            )

            if response.status_code != 200:
                return []

            return response.json()['result']

        except Exception as e:
            logger.error(f"Error getting definitions: {str(e)}")
            return []

    async def _get_semantic_types(self, cui: str) -> List[Dict]:
        """Get semantic types for a concept"""
        try:
            path = f'/rest/content/{self.version}/CUI/{cui}/semantictypes'
            params = {'apiKey': self.api_key}

            response = requests.get(
                self.base_uri + path,
                params=params
            )

            if response.status_code != 200:
                return []

            return response.json()['result']

        except Exception as e:
            logger.error(f"Error getting semantic types: {str(e)}")
            return []

    async def get_concept_relationships(self, cui: str) -> List[Dict[str, Any]]:
        """Get relationships for a concept"""
        try:
            path = f'/rest/content/{self.version}/CUI/{cui}/relations'
            params = {
                'apiKey': self.api_key,
                'pageSize': 100
            }

            logger.info(f"Getting relationships for CUI: {cui}")
            response = requests.get(
                self.base_uri + path,
                params=params
            )

            if response.status_code != 200:
                logger.error(f"Error getting relationships: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []

            results = response.json()['result']
            logger.info(f"Found {len(results)} relationships for CUI: {cui}")
            return results

        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            return []