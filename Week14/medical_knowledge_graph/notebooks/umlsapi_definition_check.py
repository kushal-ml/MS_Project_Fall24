import sys
from pathlib import Path
import os
from dotenv import load_dotenv
import json

# Get the absolute path to the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.processors.umls_api_processor import UMLSAPIProcessor

async def print_concept_definitions(cui: str):
    # Load environment variables
    load_dotenv()
    
    # Initialize the UMLS API processor
    umls_processor = UMLSAPIProcessor(api_key=os.getenv('API_KEY'))
    
    # Get definitions
    all_definitions = await umls_processor._get_definitions(cui)
    
    # Define English sources
    english_sources = {
        'NCI',      # National Cancer Institute
        'MSH',      # Medical Subject Headings
        'SNOMEDCT_US', # SNOMED CT US Edition
        'CSP',      # Current Separations Personnel
        'MTH',      # UMLS Metathesaurus
        'NIC',      # Nursing Interventions Classification
        'NOC',      # Nursing Outcomes Classification
        'NDFRT',    # National Drug File
        'AOD',      # Alcohol and Other Drug Thesaurus
        'CHV',      # Consumer Health Vocabulary
        'MEDLINEPLUS', # MedlinePlus
        'HPO'       # Human Phenotype Ontology
    }
    
    # Filter English definitions
    english_definitions = [
        definition for definition in all_definitions 
        if definition.get('rootSource') in english_sources
    ]
    
    # Print formatted JSON
    print("\nEnglish Definitions for CUI:", cui)
    print(json.dumps(english_definitions, indent=2))

    # Print in a more readable format
    print("\nReadable format:")
    for idx, definition in enumerate(english_definitions, 1):
        print(f"\nDefinition {idx}:")
        print(f"Root Source: {definition.get('rootSource', 'N/A')}")
        print(f"Value: {definition.get('value', 'N/A')}")
        if 'sourceVersion' in definition:
            print(f"Version: {definition.get('sourceVersion', 'N/A')}")

    # Print source statistics
    print("\nSource Statistics:")
    source_counts = {}
    for def_item in all_definitions:
        source = def_item.get('rootSource', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in source_counts.items():
        print(f"{source}: {count} definition(s)")

# Run the function
if __name__ == "__main__":
    import asyncio
    cui = "C0065374"  # Example CUI for Peritoneal Dialysis
    asyncio.run(print_concept_definitions(cui))