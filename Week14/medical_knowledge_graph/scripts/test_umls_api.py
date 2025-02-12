import os
from dotenv import load_dotenv
import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_umls_connection():
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv('API_KEY')  
        
        if not api_key:
            raise ValueError("UMLS API key not found in environment variables")
            
        # Test endpoints
        base_url = "https://uts-ws.nlm.nih.gov"
        version = "current"
        
        # 1. Test version endpoint
        logger.info("Testing UMLS API connection...")
        version_url = f"{base_url}/version"
        version_response = requests.get(
            version_url,
            params={"apiKey": api_key}
        )
        
        if version_response.status_code == 200:
            logger.info("✓ Successfully connected to UMLS API")
        else:
            logger.error(f"× Failed to connect to UMLS API. Status code: {version_response.status_code}")
            return
            
        # 2. Test search functionality
        logger.info("\nTesting UMLS search functionality...")
        search_url = f"{base_url}/search/{version}"
        search_params = {
            "string": "diabetes",
            "apiKey": api_key,
            "searchType": "exact"
        }
        
        search_response = requests.get(search_url, params=search_params)
        
        if search_response.status_code == 200:
            results = search_response.json()
            result_count = results.get('result', {}).get('results', [])
            logger.info(f"✓ Search successful. Found {len(result_count)} results for 'diabetes'")
            
            # Display first result
            if result_count:
                first_result = result_count[0]
                logger.info("\nSample Result:")
                logger.info(f"Name: {first_result.get('name')}")
                logger.info(f"URI: {first_result.get('uri')}")
                logger.info(f"CUI: {first_result.get('ui')}")
        else:
            logger.error(f"× Search failed. Status code: {search_response.status_code}")
            
        # 3. Test specific CUI lookup
        logger.info("\nTesting CUI lookup...")
        cui = "C0011849"  # CUI for Diabetes
        cui_url = f"{base_url}/content/{version}/CUI/{cui}"
        
        cui_response = requests.get(
            cui_url,
            params={"apiKey": api_key}
        )
        
        if cui_response.status_code == 200:
            cui_data = cui_response.json()
            logger.info(f"✓ Successfully retrieved information for CUI: {cui}")
            logger.info(f"Name: {cui_data.get('result', {}).get('name')}")
        else:
            logger.error(f"× Failed to retrieve CUI information. Status code: {cui_response.status_code}")
            
    except Exception as e:
        logger.error(f"Error testing UMLS API: {str(e)}")
        raise

if __name__ == "__main__":
    test_umls_connection() 