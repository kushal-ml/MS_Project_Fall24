#!/usr/bin/env python
# coding: utf-8

"""
UMLS Semantic Type Mapper for USMLE Knowledge Graph

This script analyzes MRSTY.RRF to extract semantic types and 
creates a mapping for USMLE-relevant medical concepts.
"""

import os
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Map UMLS semantic types to USMLE domains')
    parser.add_argument('--mrsty_path', required=True, help='Path to MRSTY.RRF file')
    parser.add_argument('--output', default='usmle_semantic_types.py', help='Output Python file')
    return parser.parse_args()

def extract_semantic_types(mrsty_path):
    """Extract all semantic types from MRSTY.RRF"""
    print(f"Reading MRSTY file from: {mrsty_path}")
    
    # Define column names for MRSTY.RRF
    columns = [
        'CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF'
    ]
    
    # Read MRSTY file in chunks to handle large files
    semantic_types = {}
    type_counts = defaultdict(int)
    
    # Read in chunks to handle large files
    for chunk in pd.read_csv(
        mrsty_path, sep='|', header=None, 
        names=columns, usecols=[1, 3], # Only need TUI and STY
        chunksize=100000, dtype=str
    ):
        # Extract unique TUI to semantic type mappings
        for _, row in chunk.iterrows():
            tui = row['TUI']
            sty = row['STY']
            semantic_types[tui] = sty
            type_counts[tui] += 1
    
    print(f"Found {len(semantic_types)} unique semantic types")
    
    # Sort by frequency for analysis
    sorted_types = sorted(
        [(tui, sty, type_counts[tui]) for tui, sty in semantic_types.items()],
        key=lambda x: x[2], 
        reverse=True
    )
    
    return semantic_types, sorted_types

def create_usmle_mapping(semantic_types, sorted_types):
    """Create USMLE-relevant mapping from semantic types"""
    
    # Define mappings from semantic types to USMLE domains
    # These are based on clinical relevance for USMLE
    usmle_mapping = {
        # Priority 1 - Core Clinical Concepts
        'priority_1': {
            # Diseases and Disorders
            'T047': 'Disease_or_Syndrome',
            'T191': 'Neoplastic_Process',
            'T046': 'Pathologic_Function',
            'T019': 'Congenital_Abnormality',
            'T020': 'Acquired_Abnormality',
            'T037': 'Injury_or_Poisoning',
            'T048': 'Mental_or_Behavioral_Dysfunction',
            
            # Signs, Symptoms, Findings
            'T184': 'Sign_or_Symptom',
            'T033': 'Finding',
            'T034': 'Laboratory_or_Test_Result',
            
            # Physiological Processes
            'T039': 'Physiologic_Function',
            'T040': 'Organism_Function',
            'T042': 'Organ_or_Tissue_Function',
            'T043': 'Cell_Function',
            'T044': 'Molecular_Function',
            
            # Pharmacology
            'T121': 'Pharmacologic_Substance',
            'T131': 'Hazardous_or_Poisonous_Substance',
            'T195': 'Antibiotic',
            'T200': 'Clinical_Drug',
            
            # Anatomy
            'T022': 'Body_System',
            'T023': 'Body_Part_Organ_or_Organ_Component',
            'T024': 'Tissue',
            'T025': 'Cell',
            'T026': 'Cell_Component',
            'T116': 'Amino_Acid_Peptide_or_Protein',
            'T126': 'Enzyme',
            
            # Microbiology
            'T004': 'Fungus',
            'T005': 'Virus',
            'T007': 'Bacterium',
            'T129': 'Immunologic_Factor',
        },
        
        # Priority 2 - Clinical Procedures & Systems
        'priority_2': {
            # Procedures
            'T058': 'Health_Care_Activity',
            'T059': 'Laboratory_Procedure',
            'T060': 'Diagnostic_Procedure',
            'T061': 'Therapeutic_or_Preventive_Procedure',
            
            # Healthcare Concepts
            'T091': 'Biomedical_Occupation_or_Discipline',
            'T170': 'Intellectual_Product',
            'T081': 'Quantitative_Concept',
            'T089': 'Regulation_or_Law',
            
            # Additional Anatomy
            'T017': 'Anatomical_Structure',
            'T018': 'Embryonic_Structure',
            'T021': 'Fully_Formed_Anatomical_Structure',
            
            # Additional Biological Concepts
            'T123': 'Biologically_Active_Substance',
            'T045': 'Genetic_Function',
            'T028': 'Gene_or_Genome',
            'T114': 'Nucleic_Acid_Nucleoside_or_Nucleotide',
        },
        
        # Priority 3 - Supporting Concepts
        'priority_3': {
            # Concepts, Ideas, Findings
            'T169': 'Functional_Concept',
            'T080': 'Qualitative_Concept',
            'T082': 'Spatial_Concept',
            'T078': 'Idea_or_Concept',
            
            # Additional Biological Entities
            'T029': 'Body_Location_or_Region',
            'T030': 'Body_Space_or_Junction',
            'T031': 'Body_Substance',
            'T120': 'Chemical_Viewed_Functionally',
            'T104': 'Chemical_Viewed_Structurally',
            'T167': 'Substance',
            'T168': 'Food',
            
            # Additional Concepts
            'T041': 'Mental_Process',
            'T032': 'Organism_Attribute',
            'T201': 'Clinical_Attribute',
        }
    }
    
    # Create reverse mapping for validation
    reverse_mapping = {}
    for priority, types in usmle_mapping.items():
        for tui, label in types.items():
            reverse_mapping[tui] = (label, priority)
    
    # Check for missing high-frequency types
    print("\nTop 20 semantic types by frequency:")
    for i, (tui, sty, count) in enumerate(sorted_types[:20]):
        priority = reverse_mapping.get(tui, ('Not mapped', 'Not mapped'))[1]
        print(f"{i+1}. {tui}: {sty} ({count:,} occurrences) - {priority}")
    
    # Check for unmapped but potentially relevant types
    print("\nPotentially relevant unmapped types:")
    relevant_keywords = ['disease', 'syndrome', 'disorder', 'finding', 'symptom', 
                        'procedure', 'drug', 'anatomy', 'organ', 'cell', 'tissue',
                        'function', 'process', 'genetic', 'clinical', 'laboratory']
    
    for tui, sty in semantic_types.items():
        if tui not in reverse_mapping:
            if any(keyword in sty.lower() for keyword in relevant_keywords):
                print(f"{tui}: {sty} - Not currently mapped")
    
    return usmle_mapping

def generate_usmle_domains(usmle_mapping, semantic_types):
    """Generate USMLE_DOMAINS structure using semantic types"""
    
    # Create the USMLE_DOMAINS structure
    usmle_domains = {
        'SNOMEDCT_US': {},
        'RXNORM': {},
        'MSH': {},
        'LNC': {}
    }
    
    # Add semantic type mappings to each vocabulary
    for priority, types in usmle_mapping.items():
        # Add to SNOMEDCT_US - clinical concepts
        usmle_domains['SNOMEDCT_US'][priority] = {}
        for tui, label in types.items():
            usmle_domains['SNOMEDCT_US'][priority][tui] = label
        
        # Add to MSH - medical concepts
        usmle_domains['MSH'][priority] = {}
        for tui, label in types.items():
            usmle_domains['MSH'][priority][tui] = label
        
        # Add subset to RXNORM - focus on drugs
        if priority == 'priority_1':
            usmle_domains['RXNORM'][priority] = {}
            drug_tuis = [tui for tui, label in types.items() 
                        if any(word in label.lower() for word in 
                              ['drug', 'substance', 'antibiotic', 'pharmacologic'])]
            for tui in drug_tuis:
                usmle_domains['RXNORM'][priority][tui] = types[tui]
        
        # Add subset to LNC - focus on laboratory tests
        if priority == 'priority_1':
            usmle_domains['LNC'][priority] = {}
            lab_tuis = [tui for tui, label in types.items() 
                       if any(word in label.lower() for word in 
                             ['laboratory', 'test', 'procedure', 'diagnostic'])]
            for tui in lab_tuis:
                usmle_domains['LNC'][priority][tui] = types[tui]
    
    # Add TTY mappings for common term types that appear in MRCONSO
    # This provides backward compatibility with the original approach
    common_ttys = {
        'SNOMEDCT_US': {
            'priority_1': {
                'PT': 'Preferred_Term',
                'FN': 'Fully_Specified_Name',
                'SY': 'Synonym',
                'SB': 'Subset_Member',
                'SCTSPA': 'Spanish_Term'
            }
        },
        'RXNORM': {
            'priority_1': {
                'IN': 'Ingredient',
                'PIN': 'Precise_Ingredient',
                'MIN': 'Multiple_Ingredients',
                'BN': 'Brand_Name',
                'SCD': 'Semantic_Clinical_Drug',
                'SBD': 'Semantic_Branded_Drug'
            }
        },
        'MSH': {
            'priority_1': {
                'MH': 'Main_Heading',
                'PM': 'Permuted_Term',
                'ET': 'Entry_Term',
                'PEP': 'Preferred_Entry_Point'
            }
        },
        'LNC': {
            'priority_1': {
                'LC': 'Long_Common_Name',
                'LN': 'LOINC_Number',
                'OSN': 'Official_Short_Name',
                'CN': 'Component_Name'
            }
        }
    }
    
    # Merge TTY mappings with semantic type mappings
    for vocab in usmle_domains:
        if vocab in common_ttys:
            for priority in common_ttys[vocab]:
                if priority not in usmle_domains[vocab]:
                    usmle_domains[vocab][priority] = {}
                usmle_domains[vocab][priority].update(common_ttys[vocab][priority])
    
    return usmle_domains

def write_output(usmle_domains, output_path):
    """Write the USMLE_DOMAINS structure to a Python file"""
    
    with open(output_path, 'w') as f:
        f.write("# USMLE Domains mapped to UMLS Semantic Types\n")
        f.write("# Generated from MRSTY.RRF\n\n")
        
        f.write("USMLE_DOMAINS = {\n")
        
        for vocab, priorities in usmle_domains.items():
            f.write(f"    '{vocab}': {{\n")
            
            for priority, types in priorities.items():
                f.write(f"        '{priority}': {{\n")
                
                # Group by category for readability
                categories = defaultdict(list)
                for tui, label in types.items():
                    category = label.split('_')[0] if '_' in label else 'Other'
                    categories[category].append((tui, label))
                
                for category, items in sorted(categories.items()):
                    f.write(f"            # {category} concepts\n")
                    for tui, label in sorted(items):
                        f.write(f"            '{tui}': '{label}',\n")
                
                f.write("        },\n")
            
            f.write("    },\n")
        
        f.write("}\n")
    
    print(f"\nUMLS semantic type mapping written to: {output_path}")
    print("You can now import this in your constants.py file.")

def main():
    args = parse_args()
    
    # Extract semantic types from MRSTY.RRF
    semantic_types, sorted_types = extract_semantic_types(args.mrsty_path)
    
    # Create USMLE-relevant mapping
    usmle_mapping = create_usmle_mapping(semantic_types, sorted_types)
    
    # Generate USMLE_DOMAINS structure
    usmle_domains = generate_usmle_domains(usmle_mapping, semantic_types)
    
    # Write output
    write_output(usmle_domains, args.output)
    
    # Provide instructions for updating code
    print("\n=== Next Steps ===")
    print("1. Add the generated USMLE_DOMAINS to your constants.py file")
    print("2. Update your processor to use TUI from MRSTY instead of TTY from MRCONSO:")
    print("""
    def _process_concept_row(self, row) -> Dict:
        cui = row[0]       # CUI
        source = row[11]   # SAB column
        tty = row[12]      # TTY column
        term = row[14]     # STR column
        
        # First try TTY-based mapping (backward compatibility)
        semantic_type = None
        domain = None
        priority = None
        
        for vocab, priorities in self.usmle_domains.items():
            if source == vocab:
                for priority_level, domains in priorities.items():
                    if tty in domains:
                        domain = domains[tty]
                        semantic_type = domain
                        priority = priority_level
                        break
                if domain:
                    break
        
        # If no match, try semantic type mapping
        if not domain and cui in self.semantic_types_dict:
            tui = self.semantic_types_dict[cui]
            for priority_level, domains in self.usmle_domains[source].items():
                if tui in domains:
                    domain = domains[tui]
                    semantic_type = domain
                    priority = priority_level
                    break
        
        # Skip if not in USMLE domains
        if not domain:
            return None
            
        # Rest of your method...
    """)
    print("3. Load semantic types for each CUI before processing MRCONSO:")
    print("""
    def _load_semantic_types(self, mrsty_path):
        self.semantic_types_dict = {}
        
        # Read MRSTY file to get semantic types for each CUI
        columns = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF']
        for chunk in pd.read_csv(
            mrsty_path, sep='|', header=None, 
            names=columns, usecols=[0, 1], # Only need CUI and TUI
            chunksize=100000, dtype=str
        ):
            for _, row in chunk.iterrows():
                self.semantic_types_dict[row['CUI']] = row['TUI']
                
        logger.info(f"Loaded semantic types for {len(self.semantic_types_dict)} concepts")
    """)

if __name__ == "__main__":
    main()