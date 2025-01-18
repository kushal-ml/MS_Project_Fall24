USMLE_DOMAINS = {
    'SNOMEDCT_US': {  
        'priority_1': {
            # Core Clinical Concepts - Additional Important Ones
            'DISO': 'Disease_or_Syndrome',
            'FIND': 'Finding',
            'SYMP': 'Sign_or_Symptom',
            'NEOP': 'Neoplastic_Process',
            'MHLT': 'Mental_Health',
            'CONG': 'Congenital_Abnormality',
            'PATH': 'Pathologic_Function',
            'CLIN': 'Clinical_Attribute',
            'EMRG': 'Emergency_Condition',     # Added: Emergency medicine
            'IMMN': 'Immune_Response',         # Added: Immunology
            'GNRC': 'Genetic_Condition',       # Added: Genetics
            'METB': 'Metabolic_Disorder'       # Added: Metabolism
        },
        'priority_2': {
            # Clinical Procedures & Systems
            'PROC': 'Therapeutic_Procedure',
            'DIAG': 'Diagnostic_Procedure',
            'LAB': 'Laboratory_Procedure',
            'ANAT': 'Anatomical_Structure',
            'PHYS': 'Physiologic_Function',
            'INJ': 'Injury_or_Poisoning',
            'MICR': 'Microorganism',
            'CELL': 'Cell_Structure',          # Added: Cell biology
            'MOLC': 'Molecular_Function',      # Added: Molecular biology
            'EMBR': 'Embryologic_Development'  # Added: Embryology
        }
    },
    'RXNORM': {
        'priority_1': {
            # Medications and Pharmacology
            'IN': 'Pharmacologic_Substance',
            'CLIN': 'Clinical_Drug',
            'BAS': 'Biologically_Active_Substance',
            'ANT': 'Antibiotic',              # Added: Antimicrobials
            'CHEM': 'Chemical_Structure',      # Added: Drug chemistry
            'DOSE': 'Drug_Dosage'             # Added: Pharmacology
        }
    },
    'LNC': {
        'priority_1': {
            # Laboratory and Diagnostics
            'LP': 'Laboratory_Procedure',
            'LAB': 'Laboratory_Test_Result',
            'CHEM': 'Chemistry_Test',
            'HEM': 'Hematology_Test',
            'MICRO': 'Microbiology_Test',
            'IMM': 'Immunology_Test',
            'PATH': 'Pathology_Test',          # Added: Pathology
            'IMG': 'Imaging_Study',            # Added: Radiology
            'VITL': 'Vital_Signs'              # Added: Physical exam
        }
    },
    'MSH': {
        'priority_1': {
            # Medical Concepts and Research
            'MH': 'Main_Heading',
            'DISE': 'Disease',
            'CHEM': 'Chemical_Drug',
            'ANAT': 'Anatomy',
            'PROC': 'Procedure',
            'EPID': 'Epidemiology',            # Added: Public health
            'STAT': 'Statistics',              # Added: Biostatistics
            'ETIO': 'Etiology',                # Added: Disease causes
            'PREV': 'Prevention',              # Added: Preventive medicine
            'PROG': 'Prognosis'                # Added: Clinical outcomes
        }
    }
}

# Relationship types focused on USMLE-relevant connections
IMPORTANT_RELATIONS = {
    'priority_1': {  # Critical clinical relationships
        'PAR': 'parent_of',         # Hierarchical relationships
        'CHD': 'child_of',
        'SY': 'synonym_of',         # Term relationships
        'TRE': 'treats',            # Treatment relationships
        'CAU': 'causes',            # Disease causation
        'MAN': 'manifests_as',      # Clinical manifestations
        'DIA': 'diagnosed_by',      # Diagnostic relationships
        'PRE': 'presents_with',     # Clinical presentation
        'LOC': 'location_of',       # Anatomical relationships
        'MOL': 'molecular_mechanism_of', # Added: Molecular mechanisms
        'GEN': 'genetic_basis_of',   # Added: Genetic relationships
        'DEV': 'develops_from',      # Added: Embryological development
        'IMM': 'immune_response_to'  # Added: Immunological relationships
    },
    'priority_2': {  # Important clinical associations
        'TST': 'tested_by',         # Laboratory relationships
        'INT': 'interacts_with',    # Drug interactions
        'RIS': 'risk_factor_for',   # Risk factors
        'COM': 'complication_of',   # Complications
        'EXA': 'examined_by',       # Physical examination
        'PRV': 'prevents',          # Prevention
        'DOS': 'dosage_for',        # Added: Drug dosing
        'MET': 'metabolized_by',    # Added: Drug metabolism
        'EPD': 'epidemiology_of',   # Added: Population statistics
        'PRG': 'prognosis_of',      # Added: Clinical outcomes
        'EMG': 'emergency_management_of'  # Added: Emergency procedures
    }
}

# Semantic types aligned with USMLE domains
IMPORTANT_SEMANTIC_TYPE = {
    'priority_1': {  # Core clinical concepts
        'T047': 'Disease_or_Syndrome',
        'T048': 'Mental_Behavioral_Dysfunction',
        'T184': 'Sign_or_Symptom',
        'T121': 'Pharmacologic_Substance',
        'T191': 'Neoplastic_Process',
        'T046': 'Pathologic_Function',
        'T033': 'Finding',
        'T201': 'Clinical_Attribute',
        'T019': 'Congenital_Abnormality',     # Added: Congenital conditions
        'T045': 'Genetic_Function',           # Added: Genetics
        'T129': 'Immunologic_Factor',         # Added: Immunology
        'T043': 'Cell_Function',              # Added: Cell biology
        'T044': 'Molecular_Function',         # Added: Molecular biology
        'T042': 'Organ_or_Tissue_Function',   # Added: Physiology
        'T037': 'Injury_or_Poisoning'         # Added: Emergency medicine
    },
    'priority_2': {  # Clinical procedures and diagnostics
        'T059': 'Laboratory_Procedure',
        'T061': 'Therapeutic_Procedure',
        'T060': 'Diagnostic_Procedure',
        'T023': 'Body_Part_Organ',
        'T190': 'Anatomical_Structure',
        'T034': 'Laboratory_Test_Result',
        'T200': 'Clinical_Drug',
        'T004': 'Fungus',
        'T005': 'Virus',
        'T007': 'Bacterium',
        'T025': 'Cell',                       # Added: Cell structure
        'T116': 'Amino_Acid_Peptide_Protein', # Added: Biochemistry
        'T126': 'Enzyme',                     # Added: Metabolism
        'T081': 'Quantitative_Concept',       # Added: Biostatistics
        'T169': 'Functional_Concept',         # Added: Physiologic processes
        'T070': 'Natural_Phenomenon_Process'  # Added: Pathophysiology
    }
}

# Integration order for processing
INTEGRATION_ORDER = [
    'SNOMEDCT_US',
    'RXNORM',
    'LNC',
    'MSH'
]





# # Addtional

# USMLE_DOMAINS = {
#     'SNOMEDCT_US': {  
#         'priority_1': {
#             # Core Clinical Concepts
#             'DISO': 'Disease_or_Syndrome',
#             'FIND': 'Finding',
#             'SYMP': 'Sign_or_Symptom',
#             'NEOP': 'Neoplastic_Process',
#             'MHLT': 'Mental_Health',
#             'CONG': 'Congenital_Abnormality',
#             'PATH': 'Pathologic_Function',
#             'CLIN': 'Clinical_Attribute',
#             'EMRG': 'Emergency_Condition',
#             'IMMN': 'Immune_Response',
#             'GNRC': 'Genetic_Condition',
#             'METB': 'Metabolic_Disorder',
#             # Added: Genetics and Genomics
#             'GMUT': 'Genetic_Mutation',
#             'GVAR': 'Genomic_Variation',
#             # Added: Infectious Diseases
#             'PTGN': 'Pathogen',
#             'INFT': 'Infection',
#             # Added: Ethics and Professionalism
#             'ETHC': 'Ethical_Principle',
#             'PROF': 'Professional_Guideline',
#             'BIOC': 'Biochemical_Process',
#             'CSCN': 'Clinical_Scenario',
#             'DGNS': 'Diagnostic_Criteria',
#             'TRMT': 'Treatment_Protocol',
#             'OUTC': 'Clinical_Outcome',
#             'ASSM': 'Clinical_Assessment'
#         },
#         'priority_2': {
#             # Clinical Procedures & Systems
#             'PROC': 'Therapeutic_Procedure',
#             'DIAG': 'Diagnostic_Procedure',
#             'LAB': 'Laboratory_Procedure',
#             'ANAT': 'Anatomical_Structure',
#             'PHYS': 'Physiologic_Function',
#             'INJ': 'Injury_or_Poisoning',
#             'MICR': 'Microorganism',
#             'CELL': 'Cell_Structure',
#             'MOLC': 'Molecular_Function',
#             'EMBR': 'Embryologic_Development',
#             # Added: Healthcare Systems
#             'HLTH': 'Healthcare_Policy',
#             'INSR': 'Insurance_Plan'
#         }
#     },
#     'RXNORM': {
#         'priority_1': {
#             # Medications and Pharmacology
#             'IN': 'Pharmacologic_Substance',
#             'CLIN': 'Clinical_Drug',
#             'BAS': 'Biologically_Active_Substance',
#             'ANT': 'Antibiotic',
#             'CHEM': 'Chemical_Structure',
#             'DOSE': 'Drug_Dosage',
#             # Added: Nutrition
#             'NUTR': 'Nutrient',
#             'METB': 'Metabolic_Pathway',
#             # Added for biochemistry coverage
#             'BCHEM': 'Biochemical_Substance',
#             'ENZYME': 'Enzyme',
#             'METAB': 'Metabolite'
#         }
#     },
#     'LNC': {
#         'priority_1': {
#             # Laboratory and Diagnostics
#             'LP': 'Laboratory_Procedure',
#             'LAB': 'Laboratory_Test_Result',
#             'CHEM': 'Chemistry_Test',
#             'HEM': 'Hematology_Test',
#             'MICRO': 'Microbiology_Test',
#             'IMM': 'Immunology_Test',
#             'PATH': 'Pathology_Test',
#             'IMG': 'Imaging_Study',
#             'VITL': 'Vital_Signs',
#             # Added: Public Health
#             'EPID': 'Epidemiological_Study',
#             'PHLT': 'Public_Health_Issue',
#             # Added for diagnostic coverage
#             'DCRIT': 'Diagnostic_Criteria',
#             'TPROT': 'Treatment_Protocol',
#             'CPLAN': 'Care_Plan'
#         }
#     },
#     'MSH': {
#         'priority_1': {
#             # Medical Concepts and Research
#             'MH': 'Main_Heading',
#             'DISE': 'Disease',
#             'CHEM': 'Chemical_Drug',
#             'ANAT': 'Anatomy',
#             'PROC': 'Procedure',
#             'EPID': 'Epidemiology',
#             'STAT': 'Statistics',
#             'ETIO': 'Etiology',
#             'PREV': 'Prevention',
#             'PROG': 'Prognosis',
#             # Added: Additional Research Concepts
#             'GNOM': 'Genomics_Research',
#             'NUTR': 'Nutrition_Research',
#             'ETHC': 'Ethics_Research',
#             'HLTH': 'Healthcare_Systems_Research',
#             # Added for comprehensive coverage
#             'BCHE': 'Biochemical_Process',
#             'CSCE': 'Clinical_Scenario',
#             'DIAG': 'Diagnostic_Method',
#             'THER': 'Therapeutic_Approach'
#         }
#     }
# }

# # Define relationships between domains
# RELATIONSHIPS = {
#     'HAS_SYMPTOM': ['DISO', 'SYMP'],
#     'TREATS': ['IN', 'DISO'],
#     'CAUSES': ['PTGN', 'INFT'],
#     'ASSOCIATED_WITH': ['GMUT', 'DISO'],
#     'INVOLVED_IN': ['NUTR', 'METB'],
#     'STUDIES': ['EPID', 'PHLT'],
#     'GOVERNS': ['ETHC', 'PROF'],
#     'REGULATES': ['HLTH', 'INSR'],
#     'HAS_FINDING': ['DISO', 'FIND'],
#     'INDICATES': ['SYMP', 'DISO'],
#     'PART_OF': ['ANAT', 'PHYS'],
#     'MEASURES': ['LAB', 'FIND'],
#     'PREVENTS': ['PREV', 'DISO'],
#     'DIAGNOSES': ['DIAG', 'DISO'],
#     'HAS_OUTCOME': ['CSCN', 'OUTC'],
#     'REQUIRES_ASSESSMENT': ['CSCN', 'ASSM'],
#     'FOLLOWS_PROTOCOL': ['TRMT', 'TPROT'],
#     'USES_CRITERIA': ['DGNS', 'DCRIT'],
#     'INVOLVES_PROCESS': ['BIOC', 'ENZYME'],
#     'RESULTS_IN': ['METAB', 'BCHEM']
# }

# # Define semantic types for relationships
# SEMANTIC_TYPES = {
#     'Clinical': ['DISO', 'SYMP', 'FIND', 'CLIN'],
#     'Diagnostic': ['LAB', 'IMG', 'DIAG'],
#     'Therapeutic': ['IN', 'PROC', 'NUTR'],
#     'Research': ['EPID', 'STAT', 'GNOM'],
#     'System': ['HLTH', 'INSR', 'PROF'],
#     'Biological': ['ANAT', 'PHYS', 'CELL', 'MOLC'],
#     'Genetic': ['GMUT', 'GVAR', 'GNRC'],
#     'Public_Health': ['PHLT', 'PREV', 'EPID'],
#     'Ethical': ['ETHC', 'PROF'],
#     'Clinical_Scenario': ['CSCN', 'OUTC', 'ASSM'],
#     'Diagnostic_Protocol': ['DGNS', 'DCRIT'],
#     'Treatment_Protocol': ['TRMT', 'TPROT'],
#     'Biochemical': ['BIOC', 'BCHEM', 'ENZYME', 'METAB']
# }