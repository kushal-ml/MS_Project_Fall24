USMLE_DOMAINS = {
    'SNOMEDCT_US': {  
        'priority_1': {
            # Core Clinical Concepts
            'DISO': 'Disease_or_Syndrome',
            'FIND': 'Finding',
            'SYMP': 'Sign_or_Symptom',
            'NEOP': 'Neoplastic_Process',
            'MHLT': 'Mental_Health',
            'CONG': 'Congenital_Abnormality',
            'PATH': 'Pathologic_Function',
            'CLIN': 'Clinical_Attribute',
            'EMRG': 'Emergency_Condition',
            'IMMN': 'Immune_Response',
            'GNRC': 'Genetic_Condition',
            'METB': 'Metabolic_Disorder',
            # Added: Genetics and Genomics
            'GMUT': 'Genetic_Mutation',
            'GVAR': 'Genomic_Variation',
            # Added: Infectious Diseases
            'PTGN': 'Pathogen',
            'INFT': 'Infection',
            # Added: Ethics and Professionalism
            'ETHC': 'Ethical_Principle',
            'PROF': 'Professional_Guideline',
            'BIOC': 'Biochemical_Process',
            'CSCN': 'Clinical_Scenario',
            'DGNS': 'Diagnostic_Criteria',
            'TRMT': 'Treatment_Protocol',
            'OUTC': 'Clinical_Outcome',
            'ASSM': 'Clinical_Assessment'
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
            'CELL': 'Cell_Structure',
            'MOLC': 'Molecular_Function',
            'EMBR': 'Embryologic_Development',
            # Added: Healthcare Systems
            'HLTH': 'Healthcare_Policy',
            'INSR': 'Insurance_Plan'
        }
    },
    'RXNORM': {
        'priority_1': {
            # Medications and Pharmacology
            'IN': 'Pharmacologic_Substance',
            'CLIN': 'Clinical_Drug',
            'BAS': 'Biologically_Active_Substance',
            'ANT': 'Antibiotic',
            'CHEM': 'Chemical_Structure',
            'DOSE': 'Drug_Dosage',
            # Added: Nutrition
            'NUTR': 'Nutrient',
            'METB': 'Metabolic_Pathway',
            # Added for biochemistry coverage
            'BCHEM': 'Biochemical_Substance',
            'ENZYME': 'Enzyme',
            'METAB': 'Metabolite'
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
            'PATH': 'Pathology_Test',
            'IMG': 'Imaging_Study',
            'VITL': 'Vital_Signs',
            # Added: Public Health
            'EPID': 'Epidemiological_Study',
            'PHLT': 'Public_Health_Issue',
            # Added for diagnostic coverage
            'DCRIT': 'Diagnostic_Criteria',
            'TPROT': 'Treatment_Protocol',
            'CPLAN': 'Care_Plan'
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
            'EPID': 'Epidemiology',
            'STAT': 'Statistics',
            'ETIO': 'Etiology',
            'PREV': 'Prevention',
            'PROG': 'Prognosis',
            # Added: Additional Research Concepts
            'GNOM': 'Genomics_Research',
            'NUTR': 'Nutrition_Research',
            'ETHC': 'Ethics_Research',
            'HLTH': 'Healthcare_Systems_Research',
            # Added for comprehensive coverage
            'BCHE': 'Biochemical_Process',
            'CSCE': 'Clinical_Scenario',
            'DIAG': 'Diagnostic_Method',
            'THER': 'Therapeutic_Approach'
        }
    }
}



# # Relationship types focused on USMLE-relevant connections
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
        'MAY': 'may_treat',
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

SEMANTIC_TYPE_TO_LABEL = {
    # Drug related
    'T121': 'Drug',        # Pharmacologic Substance
    'T200': 'Drug',        # Clinical Drug
    
    # Disease related
    'T047': 'Disease',     # Disease or Syndrome
    'T048': 'Disease',     # Mental/Behavioral Dysfunction
    'T191': 'Disease',     # Neoplastic Process
    
    # Symptom related
    'T184': 'Symptom',     # Sign or Symptom
    
    # Procedure related
    'T059': 'Procedure',   # Laboratory Procedure
    'T060': 'Procedure',   # Diagnostic Procedure
    'T061': 'Procedure',   # Therapeutic Procedure
    
    # Anatomy related
    'T023': 'Anatomy',     # Body Part, Organ
    'T190': 'Anatomy',     # Anatomical Structure
    
    # Clinical Scenario (can be derived from combinations)
    'T033': 'ClinicalScenario'  # Finding
}

# Relations mapping from MRREL
RELATION_TYPE_MAPPING = {
    # Basic clinical relationships
    'may_be_treated_by': 'MAY_BE_TREATED_BY',
    'has_causative_agent': 'HAS_CAUSATIVE_AGENT',
    'associated_with': 'ASSOCIATED_WITH',
    'gene_associated_with_disease': 'GENE_ASSOCIATED_WITH_DISEASE',
    'related_to': 'RELATED_TO',
    
    # Disease relationships
    'may_treat': 'MAY_TREAT',
    'may_prevent': 'MAY_PREVENT',
    'disease_has_finding': 'DISEASE_HAS_FINDING',
    'associated_finding_of': 'ASSOCIATED_FINDING_OF',
    # symptom 
    'clinical_course_of': 'CLINICAL_COURSE_OF', 
    # symptom
    'manifestation_of': 'MANIFESTATION_OF',
    'gene_associated_with_disease': 'GENE_ASSOCIATED_WITH_DISEASE',
    'is_finding_of_disease': 'IS_FINDING_OF_DISEASE',
    'is_not_finding_of_disease': 'IS_NOT_FINDING_OF_DISEASE',


    # Drug relationships
    'has_ingredient': 'HAS_INGREDIENT',
    'has_precise_ingredient': 'HAS_PRECISE_INGREDIENT',
    'chemical_or_drug_affects_gene_product': 'CHEMICAL_OR_DRUG_AFFECTS_GENE_PRODUCT',
    'contraindicated_with_disease': 'CONTRAINDICATED_WITH_DISEASE',
    'has_mechanism_of_action': 'HAS_MECHANISM_OF_ACTION',
    'contraindicated_mechanism_of_action_of': 'CONTRAINDICATED_MECHANISM_OF_ACTION_OF',
    'mechanism_of_action_of': 'MECHANISM_OF_ACTION_OF',
    'chemical_or_drug_has_mechanism_of_action': 'CHEMICAL_OR_DRUG_HAS_MECHANISM_OF_ACTION',
    
    # Anatomical relationships
    'occurs_in': 'OCCURS_IN',
    'location_of': 'LOCATION_OF',
    'is_location_of_biological_process': 'IS_LOCATION_OF_BIOLOGICAL_PROCESS',
    'has_location': 'HAS_LOCATION',
    'is_location_of_anatomic_structure': 'IS_LOCATION_OF_ANATOMIC_STRUCTURE',
    'part_of': 'PART_OF',
    'drains_into': 'DRAINS_INTO',
    
    # Process relationships
    'occurs_before': 'OCCURS_BEFORE',
    'regulates': 'REGULATES',
    'negatively_regulates': 'NEGATIVELY_REGULATES',
    'positively_regulates': 'POSITIVELY_REGULATES',

    # Diagnostic relationships
    'may_be_diagnosed_by':'MAY_BE_DIAGNOSED_BY',
    'may_be_finding_of_disease': 'MAY_BE_FINDING_OF_DISEASE',
    'associated_etiologic_finding_of': 'ASSOCIATED_ETIOLOGIC_FINDING_OF',
    'disease_has_finding': 'DISEASE_HAS_FINDING',
    'disease_may_have_finding': 'DISEASE_MAY_HAVE_FINDING',
    'is_finding_of_disease': 'IS_FINDING_OF_DISEASE',
    'associated_finding_of': 'ASSOCIATED_FINDING_OF',

    
    # Clinical progression
    'has_course': 'HAS_COURSE',
    'develops_into': 'DEVELOPS_INTO',
    'cause_of': 'CAUSE_OF',

    
    # Treatment priority
    'disease_has_accepted_treatment_with_regimen': 'DISEASE_HAS_ACCEPTED_TREATMENT_WITH_REGIMEN',

}

# Hierarchical relationship mappings from MRHIER
HIER_TYPE_MAPPING = {
    # Basic hierarchical
    'CHD': 'PART_OF',
    'PAR': 'IS_A',
    'RB': 'BROADER_THAN',
    'RN': 'NARROWER_THAN',
    
    # Anatomical hierarchies
    'ANC': 'ANCESTOR_OF',
    'DEL': 'DELIVERS_TO',
    'ISA': 'IS_TYPE_OF',
    'PAR': 'PARENT_OF',
    
    # Disease hierarchies
    'SIB': 'SIBLING_OF',
    'SY': 'SYMPTOM_OF',
    'RO': 'OCCURS_WITH',
    
    # Drug hierarchies
    'RQ': 'RELATED_QUAL',
    'SY': 'SYNONYM_OF',
    'XR': 'NOT_RELATED_TO'
}

# Semantic groups for filtering
SEMANTIC_GROUPS = {
    'ANAT': ['T017', 'T029', 'T023', 'T030', 'T031'],  # Anatomical Structure
    'DISO': ['T047', 'T048', 'T049', 'T050'],          # Disease or Syndrome
    'CHEM': ['T116', 'T121', 'T122', 'T123'],          # Chemical, Drug
    'PROC': ['T060', 'T061', 'T062', 'T063'],          # Therapeutic Procedure
    'PHYS': ['T039', 'T040', 'T041', 'T042'],          # Physiologic Function
    'SYMP': ['T184', 'T185']                           # Sign or Symptom
}