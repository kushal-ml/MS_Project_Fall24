USMLE_DOMAINS = {
    'SNOMEDCT_US': {  
        'priority_1': {
            # Core Clinical Concepts - Essential for all Steps
            'T047': 'Disease_or_Syndrome',
            'T184': 'Sign_or_Symptom',
            'T033': 'Finding',
            'T191': 'Neoplastic_Process',
            'T046': 'Pathologic_Function',
            'T037': 'Injury_or_Poisoning',
            'T048': 'Mental_Disorder',  # Psychiatry questions
            'T060': 'Diagnostic_Procedure',
            'T061': 'Therapeutic_Procedure',
            'T121': 'Pharmacologic_Substance',
            'T034': 'Laboratory_Finding',
            'T201': 'Clinical_Attribute',
            'T026': 'Cell_Component', 
            
            # Step 1 Heavy Concepts - Biochemistry/Physiology
            'T116': 'Amino_Acid_Peptide_Protein',
            'T123': 'Biologically_Active_Substance',
            'T126': 'Enzyme',
            'T118': 'Carbohydrate',
            'T119': 'Lipid',
            'T127': 'Vitamin',
            'T114': 'Nucleic_Acid',
            'T044': 'Molecular_Function',
            'T045': 'Genetic_Function',
            'T028': 'Gene_or_Genome',
            'T043': 'Cell_Function',
            'T025': 'Cell',
            'T042': 'Organ_Function',
            
            # Step 2 Key Concepts - Clinical Reasoning/Management
            'T058': 'Healthcare_Activity',
            'T032': 'Organism_Attribute',
            'T184': 'Patient_Symptom',
            'T097': 'Professional_Occupation',
            'T169': 'Functional_Concept',
            'T196': 'Element_Ion_Isotope',  # Electrolyte disorders
            'T074': 'Medical_Device',
            'T195': 'Antibiotic',
            'T200': 'Drug_Delivery_Device',
            'T129': 'Immunologic_Factor',
            
            # Step 3 Focused Concepts - System-Based Practice
            'T093': 'Healthcare_System',
            'T094': 'Healthcare_Policy',
            'T058': 'Healthcare_Process',
            'T091': 'Biomedical_Institution',
            'T101': 'Patient_Population',
            'T170': 'Clinical_Guideline'
        },
        'priority_2': {
            # Comprehensive Anatomical Concepts
            'T017': 'Anatomical_Structure',
            'T023': 'Body_Part',
            'T029': 'Body_Location',
            'T030': 'Body_Space',
            'T021': 'Fully_Formed_Anatomical_Structure',
            'T022': 'Body_System',
            'T024': 'Tissue',
            'T018': 'Embryonic_Structure',
            'T020': 'Acquired_Anatomical_Structure',
            
            # Pathophysiology Concepts
            'T039': 'Physiologic_Function',
            'T040': 'Organism_Function',
            'T019': 'Congenital_Abnormality',
            'T190': 'Anatomical_Abnormality',
            'T049': 'Cell_or_Molecular_Dysfunction',
            
            # Clinical Procedure Details
            'T059': 'Laboratory_Procedure',
            'T063': 'Medical_Procedure',
            'T062': 'Research_Activity',
            'T065': 'Educational_Activity',
            
            # Pharmacology Details
            'T109': 'Organic_Chemical',
            'T131': 'Hazardous_Substance',
            'T125': 'Hormone',
            'T110': 'Steroid',
            'T197': 'Inorganic_Chemical',
            
            # Health Determinants
            'T054': 'Social_Behavior',
            'T051': 'Event',
            'T064': 'Governmental_Activity',
            'T052': 'Activity',
            'T041': 'Mental_Process'
        },
        'priority_3': {
            # Specialized Clinical Concepts
            'T185': 'Classification',
            'T080': 'Qualitative_Concept',
            'T081': 'Quantitative_Concept',
            'T082': 'Spatial_Concept',
            'T078': 'Idea_or_Concept',
            'T170': 'Intellectual_Product',
            
            # Time and Process Concepts
            'T079': 'Temporal_Concept',
            'T038': 'Biologic_Function',
            'T067': 'Phenomenon_or_Process',
            'T068': 'Human_Caused_Phenomenon',
            'T070': 'Natural_Phenomenon',
            
            # Additional Organism Concepts
            'T002': 'Plant',
            'T004': 'Fungus',
            'T005': 'Virus',
            'T007': 'Bacterium',
            'T008': 'Animal',
            'T010': 'Vertebrate',
            'T011': 'Amphibian',
            'T012': 'Bird',
            'T013': 'Fish',
            'T014': 'Reptile',
            'T015': 'Mammal',
            'T016': 'Human'
        }
    },
    'RXNORM': {
        'priority_1': {
            'T121': 'Pharmacologic_Substance',
            'T200': 'Clinical_Drug',
            'T203': 'Drug_Contraindication',
            'T131': 'Adverse_Drug_Effect',
            'T195': 'Antibiotic',
            'T125': 'Hormone',
            'T197': 'Inorganic_Chemical',  # Electrolytes/contrast media
            'T109': 'Organic_Chemical',
            'T127': 'Vitamin',
            'T114': 'Nucleic_Acid',        # For gene therapy concepts
            'T116': 'Protein_Therapeutic'   # Biologics
        },
        'priority_2': {
            'T130': 'Indicator_or_Reagent',
            'T196': 'Element_Ion_Isotope',
            'T110': 'Steroid',
            'T118': 'Carbohydrate',
            'T119': 'Lipid',
            'T120': 'Chemical_Viewed_Functionally',
            'T122': 'Biomedical_or_Dental_Material',
            'T123': 'Biologically_Active_Substance',
            'T129': 'Immunologic_Factor',
            'T192': 'Receptor',
            'T126': 'Enzyme'
        }
    },
    'LNC': {
        'priority_1': {
            'T034': 'Laboratory_Test_Result',
            'T059': 'Laboratory_Procedure',
            'T060': 'Diagnostic_Procedure',
            'T115': 'Biomarker_Measurement',
            'T201': 'Clinical_Attribute',   # BP, vitals, metrics
            'T196': 'Element_Ion_Isotope',  # Electrolyte measurements
            'T033': 'Finding',
            'T184': 'Sign_or_Symptom',
            'T079': 'Temporal_Concept',     # Time-based measurements
            'T081': 'Quantitative_Concept'  # Reference ranges
        },
        'priority_2': {
            'T074': 'Medical_Device',
            'T082': 'Spatial_Concept',
            'T080': 'Qualitative_Concept',
            'T170': 'Intellectual_Product',
            'T171': 'Language',
            'T091': 'Biomedical_Occupation',
            'T092': 'Organization',
            'T169': 'Functional_Concept'
        }
    },
    'ICD-10-CM': {
        'priority_1': {
            'T047': 'Disease_or_Syndrome',
            'T048': 'Mental_Disorder',
            'T191': 'Neoplastic_Process',
            'T046': 'Pathologic_Function',
            'T184': 'Sign_or_Symptom',
            'T033': 'Finding',
            'T037': 'Injury_or_Poisoning',
            'T019': 'Congenital_Abnormality',
            'T190': 'Anatomical_Abnormality',
            'T049': 'Cell_or_Molecular_Dysfunction'
        }
    },
    'CPT': {
        'priority_1': {
            'T060': 'Diagnostic_Procedure',
            'T061': 'Therapeutic_Procedure',
            'T059': 'Laboratory_Procedure',
            'T058': 'Healthcare_Activity',
            'T074': 'Medical_Device',
            'T091': 'Biomedical_Occupation',
            'T092': 'Organization',
            'T169': 'Functional_Concept'
        }
    },
    'MeSH': {
        'priority_1': {
            'T047': 'Disease_or_Syndrome',
            'T184': 'Sign_or_Symptom',
            'T033': 'Finding',
            'T191': 'Neoplastic_Process',
            'T121': 'Pharmacologic_Substance',
            'T170': 'Intellectual_Product',  # Research concepts
            'T081': 'Quantitative_Concept',  # Research metrics
            'T062': 'Research_Activity',
            'T091': 'Biomedical_Occupation',
            'T092': 'Organization',
            'T093': 'Healthcare_System',
            'T094': 'Healthcare_Policy'
        }
    },
    'NCI': {
        'priority_1': {
            'T191': 'Neoplastic_Process',
            'T121': 'Pharmacologic_Substance',  # Chemotherapy
            'T200': 'Clinical_Drug',
            'T060': 'Diagnostic_Procedure',     # Cancer screening
            'T061': 'Therapeutic_Procedure',    # Cancer treatments
            'T026': 'Cell_Component',           # Cancer cell biology
            'T025': 'Cell',
            'T028': 'Gene_or_Genome',
            'T045': 'Genetic_Function',
            'T116': 'Amino_Acid_Peptide_Protein' # Cancer markers
        }
    },
    'FMA': {
        'priority_1': {
            'T017': 'Anatomical_Structure',
            'T023': 'Body_Part',
            'T029': 'Body_Location',
            'T030': 'Body_Space',
            'T021': 'Fully_Formed_Anatomical_Structure',
            'T022': 'Body_System',
            'T024': 'Tissue',
            'T025': 'Cell'
        }
    },
    'GO': {
        'priority_1': {
            'T044': 'Molecular_Function',
            'T043': 'Cell_Function',
            'T042': 'Organ_Function',
            'T026': 'Cell_Component',
            'T045': 'Genetic_Function',
            'T028': 'Gene_or_Genome',
            'T116': 'Amino_Acid_Peptide_Protein',
            'T123': 'Biologically_Active_Substance',
            'T126': 'Enzyme'
        }
    },
    'HGVS': {  # Add genetic variation source
        'priority_1': {
            'T028': 'Gene_or_Genome',
            'T089': 'Genetic_Variation',
            'T045': 'Genetic_Function',
            'T049': 'Cell_or_Molecular_Dysfunction'
        }
    },
    'HPO': {  # Human Phenotype Ontology - important for genetic diseases
        'priority_1': {
            'T047': 'Disease_or_Syndrome',
            'T019': 'Congenital_Abnormality',
            'T033': 'Finding',
            'T184': 'Sign_or_Symptom',
            'T089': 'Genetic_Variation',
            'T049': 'Cell_or_Molecular_Dysfunction'
        }
    }
}

CONCEPT_TIERS = {
    'tier_1': [
        'Disease_or_Syndrome',
        'Sign_or_Symptom',
        'Finding',
        'Clinical_Drug',
        'Pharmacologic_Substance',
        'Diagnostic_Procedure',
        'Therapeutic_Procedure',
        'Laboratory_Test_Result',
        'Mental_Disorder',
        'Neoplastic_Process',
        'Pathologic_Function',
        'Injury_or_Poisoning',
        'Clinical_Attribute',
        'Amino_Acid_Peptide_Protein',
        'Enzyme',
        'Gene_or_Genome',
        'Anatomical_Structure',
        'Body_Part'
    ],
    'tier_2': [
        'Body_System',
        'Drug_Contraindication',
        'Genetic_Function',
        'Healthcare_Activity',
        'Adverse_Drug_Effect',
        'Antibiotic',
        'Hormone',
        'Organic_Chemical',
        'Cell',
        'Cell_Function',
        'Organ_Function',
        'Physiologic_Function',
        'Congenital_Abnormality',
        'Healthcare_System',
        'Clinical_Guideline',
        'Laboratory_Procedure',
        'Biomarker_Measurement'
    ],
    'tier_3': [
        'Healthcare_Policy',
        'Professional_Occupation',
        'Biomedical_Institution',
        'Social_Behavior',
        'Research_Activity',
        'Educational_Activity',
        'Intellectual_Product',
        'Spatial_Concept',
        'Temporal_Concept',
        'Fungus',
        'Virus',
        'Bacterium'
    ]
}


# Neo4j free tier limits
NEO4J_FREE_TIER_LIMITS = {
    'max_nodes': 700000,
    'max_relationships': 700000,
    'node_buffer': 5000,  # Keep 5000 nodes below limit as a safety buffer
    'rel_buffer': 25000   # Keep 25000 relationships below limit as a safety buffer
}

# USMLE Step-specific priorities
STEP1_PRIORITY = {
    # Basic science concepts weighted higher for Step 1
    'Biochemical_Process': 2.0,
    'Pathologic_Function': 2.0,
    'Molecular_Function': 1.8,
    'Cell_Structure': 1.8,
    'Pathophysiologic_Process': 1.8,
    'Genetic_Mechanism': 1.8,
    'Embryologic_Development': 1.5,
    'Molecular_Mechanism': 1.5,
    'Regulatory_Pathway': 1.5,
    'Physiologic_Function': 1.5,
    'Molecular_Structure': 2.2,          # Drug-target interactions
    'Biomarker_Measurement': 1.8         # Diagnostic accuracy
}

STEP2_PRIORITY = {
    # Clinical concepts weighted higher for Step 2
    'Disease_or_Syndrome': 2.0,
    'Treatment_Protocol': 2.0,
    'Diagnostic_Procedure': 1.8,
    'Clinical_Attribute': 1.8,
    'Therapeutic_Procedure': 1.8,
    'Clinical_Recommendation': 1.8,
    'Differential_Diagnosis': 2.0,
    'First_Line_Therapy': 1.8,
    'Risk_Factor': 1.5,
    'Clinical_Outcome': 1.5
}

STEP3_PRIORITY = {
    # Clinical management concepts weighted higher for Step 3
    'Clinical_Recommendation': 2.0,
    'First_Line_Therapy': 2.0,
    'Treatment_Failure': 1.8,
    'Monitoring_Parameter': 1.8,
    'Social_Determinant': 1.5,
    'Healthcare_Policy': 1.5,
    'Risk_Score': 1.5,
    'Comorbidity': 1.5,
    'Alternative_Therapy': 1.5,
    'Disease_Stage': 1.5,
    'Healthcare_Organization': 1.7,      # Quality/safety questions
    'Nutritional_Supplement': 1.4        # Alternative medicine
}

SEMANTIC_TYPES = {
    # Disease related
    'T047': 'Disease',                   # Disease or Syndrome
    'T048': 'Disease',                   # Mental/Behavioral Dysfunction
    'T191': 'Disease',                   # Neoplastic Process
    'T046': 'Disease',                   # Pathologic Function
    
    # Symptom related
    'T184': 'Symptom',                   # Sign or Symptom
    'T033': 'Finding',                   # Finding
    
    # Drug related
    'T121': 'Drug',                      # Pharmacologic Substance
    'T200': 'Drug',                      # Clinical Drug
    'T195': 'Drug',                      # Antibiotic
    
    # Anatomy related
    'T023': 'Anatomy',                   # Body Part, Organ
    'T029': 'Anatomy',                   # Body Location or Region
    'T030': 'Anatomy',                   # Body Space or Junction
    'T190': 'Anatomy',                   # Anatomical Structure
    
    # Procedure related
    'T060': 'Procedure',                 # Diagnostic Procedure
    'T061': 'Procedure',                 # Therapeutic Procedure
    'T059': 'Procedure',                 # Laboratory Procedure
    
    # Biological concepts
    'T116': 'Biochemical',               # Amino Acid, Peptide, or Protein
    'T126': 'Enzyme',                    # Enzyme
    'T129': 'Immunologic',               # Immunologic Factor
    
    # Physiological concepts
    'T039': 'Physiologic',               # Physiologic Function
    'T040': 'Organism',                  # Organism Function
    'T044': 'Molecular',                 # Molecular Function
    
    # Microorganisms
    'T007': 'Bacterium',                 # Bacterium
    'T005': 'Virus',                     # Virus
    'T004': 'Fungus'                     # Fungus
}



SEMANTIC_TYPE_TO_LABEL = {
    # Disease related
    'T047': 'Disease',        # Disease or Syndrome
    'T048': 'Disease',        # Mental/Behavioral Dysfunction
    'T191': 'Disease',        # Neoplastic Process
    'T046': 'Disease',        # Pathologic Function
    
    # Symptom related
    'T184': 'Symptom',        # Sign or Symptom
    'T033': 'Finding',        # Finding
    
    # Drug related
    'T121': 'Drug',           # Pharmacologic Substance
    'T200': 'Drug',           # Clinical Drug
    'T195': 'Drug',           # Antibiotic
    
    # Anatomy related
    'T023': 'Anatomy',        # Body Part, Organ
    'T029': 'Anatomy',        # Body Location or Region
    'T030': 'Anatomy',        # Body Space or Junction
    'T190': 'Anatomy',        # Anatomical Structure
    
    # Procedure related
    'T059': 'Procedure',      # Laboratory Procedure
    'T060': 'Procedure',      # Diagnostic Procedure
    'T061': 'Procedure',      # Therapeutic Procedure
    
    # Clinical Scenario
    'T033': 'ClinicalScenario',  # Finding
    
    # Biological concepts
    'T116': 'Biochemical',    # Amino Acid, Peptide, or Protein
    'T126': 'Enzyme',         # Enzyme
    'T129': 'Immunologic',    # Immunologic Factor
    
    # Physiological concepts
    'T039': 'Physiologic',    # Physiologic Function
    'T040': 'Physiologic',    # Organism Function
    'T044': 'Molecular',      # Molecular Function
    
    # Microorganisms
    'T007': 'Pathogen',       # Bacterium
    'T005': 'Pathogen',       # Virus
    'T004': 'Pathogen',       # Fungus
    
    # Diagnostic and Laboratory
    'T201': 'ClinicalAttribute',  # Clinical Attribute
    'T034': 'LaboratoryResult',   # Laboratory or Test Result
    
    # Genetic and Molecular
    'T028': 'GeneticFunction',    # Gene or Genome
    'T045': 'GeneticProcess',     # Genetic Function
    'T099': 'GeneticMutation',    # Family Group (for genetic disorders)
    
    # Risk and Epidemiology
    'T081': 'RiskFactor',         # Quantitative Concept (for risk scores)
    'T170': 'ClinicalConcept',    # Intellectual Product (for clinical guidelines)
    
    # Therapeutic
    'T169': 'TherapeuticConcept', # Functional Concept (for therapeutic approaches)
    'T074': 'MedicalDevice',      # Medical Device
    
    # Pathophysiology
    'T042': 'PathologicProcess',  # Organ or Tissue Function
    'T043': 'CellFunction',       # Cell Function
    
    # Social and Environmental
    'T054': 'SocialFactor',       # Social Behavior
    'T051': 'EnvironmentalFactor', # Event

    'T028': 'Gene',                       # Gene or Genome (Step 1 genetics)
    'T087': 'Amino_Acid_Sequence',        # Molecular basis (Step 1)
    'T088': 'Molecular_Structure',        # Drug-receptor interactions
    'T114': 'Nutritional_Supplement',     # Alternative therapies (Step 3)
    'T092': 'Healthcare_Organization'     # Systems-based practice
}


# Mapping from UMLS relationship attributes to standardized relationship types
RELATION_TYPE_MAPPING = {
    # Disease-Symptom relationships
    'disease_has_finding': 'DISEASE_HAS_FINDING',
    'is_finding_of_disease': 'IS_FINDING_OF_DISEASE',
    'manifestation_of': 'MANIFESTATION_OF',
    'has_finding_site': 'HAS_FINDING_SITE',
    
    # Treatment relationships
    'may_be_treated_by': 'MAY_BE_TREATED_BY',
    'may_treat': 'MAY_TREAT',
    'contraindicated_with_disease': 'CONTRAINDICATED_WITH_DISEASE',
    'has_contraindicated_drug': 'HAS_CONTRAINDICATED_DRUG',
    
    # Mechanism relationships
    'has_causative_agent': 'HAS_CAUSATIVE_AGENT',
    'due_to': 'DUE_TO',
    'pathological_process_of': 'PATHOLOGICAL_PROCESS_OF',
    'has_mechanism_of_action': 'HAS_MECHANISM_OF_ACTION',
    
    # Diagnostic relationships
    'may_be_diagnosed_by': 'MAY_BE_DIAGNOSED_BY',
    
    # Disease progression
    'develops_into': 'DEVELOPS_INTO',
    'cause_of': 'CAUSE_OF',
    'has_course': 'HAS_COURSE',
    
    # Anatomical relationships
    'location_of': 'LOCATION_OF',
    'part_of': 'PART_OF',
    'occurs_in': 'OCCURS_IN',
    
    # Pharmacology details
    'has_active_ingredient': 'HAS_ACTIVE_INGREDIENT',
    'induces': 'INDUCES',
    'has_physiologic_effect': 'HAS_PHYSIOLOGIC_EFFECT',
    
    # Diagnostic details
    'measures': 'MEASURES',
    'has_measured_component': 'HAS_MEASURED_COMPONENT',
    
    # Prevention
    'may_prevent': 'MAY_PREVENT',
    'may_be_prevented_by': 'MAY_BE_PREVENTED_BY',
    
    # Synonym relations
    'same_as': 'SAME_AS',
    'tradename_of': 'TRADENAME_OF',
    
    # Regulatory relationships
    'regulates': 'REGULATES',
    'negatively_regulates': 'NEGATIVELY_REGULATES',
    'positively_regulates': 'POSITIVELY_REGULATES',
    
    # Associated findings
    'associated_with': 'ASSOCIATED_WITH',
    'associated_finding_of': 'ASSOCIATED_FINDING_OF',
    'has_associated_finding': 'HAS_ASSOCIATED_FINDING',
    'disease_may_have_finding': 'DISEASE_MAY_HAVE_FINDING',
    
    # Advanced anatomical
    'direct_procedure_site_of': 'DIRECT_PROCEDURE_SITE_OF',
    'procedure_site_of': 'PROCEDURE_SITE_OF',
    'finding_site_of': 'FINDING_SITE_OF',
    'continuous_with': 'CONTINUOUS_WITH',
    'superior_to': 'SUPERIOR_TO',
    'posterior_to': 'POSTERIOR_TO',
    'has_regional_part': 'HAS_REGIONAL_PART',
    
    # Additional genetics & molecular
    'gene_associated_with_disease': 'GENE_ASSOCIATED_WITH_DISEASE',
    'genetic_biomarker_related_to': 'GENETIC_BIOMARKER_RELATED_TO',
    'gene_product_plays_role_in_biological_process': 'GENE_PRODUCT_PLAYS_ROLE_IN_BIOLOGICAL_PROCESS',
    
    # Specialized treatment
    'disease_has_accepted_treatment_with_regimen': 'DISEASE_HAS_ACCEPTED_TREATMENT_WITH_REGIMEN',
    
    # Context relationships
    'has_finding_context': 'HAS_FINDING_CONTEXT',
    'may_be_molecular_abnormality_of_disease': 'MAY_BE_MOLECULAR_ABNORMALITY_OF_DISEASE',
    
    # Default mappings for common REL types
    'rb': 'HAS_BROADER_RELATIONSHIP',
    'rn': 'HAS_NARROWER_RELATIONSHIP',
    'rl': 'SIMILAR_OR_LIKE',
    'ro': 'HAS_RELATIONSHIP_OTHER',
    'par': 'PARENT_OF',
    'chd': 'CHILD_OF',
    'sib': 'SIBLING_OF',
    'isa': 'IS_A',
    'inverse_isa': 'INVERSE_IS_A',
    'mapped_to': 'MAPPED_TO',
    'mapped_from': 'MAPPED_FROM'
}


# Relations mapping from MRREL
RELATIONSHIP_TIERS = {
    'tier_1': [
        # Disease-Symptom Relationships (core clinical reasoning)
        'DISEASE_HAS_FINDING',          # Key for connecting diseases to symptoms
        'IS_FINDING_OF_DISEASE',        # Inverse relationship for symptom->disease
        'MANIFESTATION_OF',             # How diseases manifest clinically
        'HAS_FINDING_SITE',             # Anatomical location of findings
        
        # Treatment Relationships (therapeutic decision making)
        'MAY_BE_TREATED_BY',            # Disease to treatment options
        'MAY_TREAT',                    # Drug to disease it treats
        'CONTRAINDICATED_WITH_DISEASE', # What NOT to use (patient safety)
        'HAS_CONTRAINDICATED_DRUG',     # Inverse of contraindication
        
        # Mechanism Relationships (pathophysiology understanding)
        'HAS_CAUSATIVE_AGENT',          # What causes the disease
        'DUE_TO',                       # Underlying cause of a condition
        'PATHOLOGICAL_PROCESS_OF',      # Disease mechanism
        'HAS_MECHANISM_OF_ACTION',      # Drug mechanism
        
        # Diagnostic Relationships (diagnostic reasoning)
        'MAY_BE_DIAGNOSED_BY',          # How to diagnose conditions
    ],

    'tier_2': [
        # Disease Progression
        'DEVELOPS_INTO',                # Disease progression
        'CAUSE_OF',                     # Disease etiology
        'HAS_COURSE',                   # Disease timeline
        
        # Anatomical Relationships
        'LOCATION_OF',                  # Where things are located
        'PART_OF',                      # Anatomical relationships
        'OCCURS_IN',                    # Where processes happen
        
        # Pharmacology Details
        'HAS_ACTIVE_INGREDIENT',        # Drug composition
        'INDUCES',                      # Drug effects/side effects
        'HAS_PHYSIOLOGIC_EFFECT',       # Physiological impact
        
        # Diagnostic Details
        'MEASURES',                     # What tests measure
        'HAS_MEASURED_COMPONENT',       # Components of lab tests
        
        # Prevention
        'MAY_PREVENT',                  # Preventive measures
        'MAY_BE_PREVENTED_BY',          # How to prevent
        
        # Synonym Relations (terminology)
        'SAME_AS',                      # Equivalent terms
        'TRADENAME_OF',                 # Brand/generic drug names
    ],

    'tier_3': [
        # Regulatory Relationships
        'REGULATES',                    # Regulatory relationships
        'NEGATIVELY_REGULATES',         # Inhibition
        'POSITIVELY_REGULATES',         # Activation
        
        # Associated Findings
        'ASSOCIATED_WITH',              # General associations
        'ASSOCIATED_FINDING_OF',        # Associated clinical findings
        'HAS_ASSOCIATED_FINDING',       # What findings are associated
        'DISEASE_MAY_HAVE_FINDING',     # Possible findings
        
        # Advanced Anatomical
        'DIRECT_PROCEDURE_SITE_OF',     # Procedure sites
        'PROCEDURE_SITE_OF',            # Where procedures happen
        'FINDING_SITE_OF',              # Where findings are located
        'CONTINUOUS_WITH',              # Anatomical continuity
        'SUPERIOR_TO',                  # Anatomical positioning
        'POSTERIOR_TO',                 # Anatomical positioning
        'HAS_REGIONAL_PART',            # Anatomical subdivision
        
        # Additional Genetics & Molecular
        'GENE_ASSOCIATED_WITH_DISEASE', # Genetic associations
        'GENETIC_BIOMARKER_RELATED_TO', # Genetic biomarkers
        'GENE_PRODUCT_PLAYS_ROLE_IN_BIOLOGICAL_PROCESS',  # Gene function
        
        # Specialized Treatment
        'DISEASE_HAS_ACCEPTED_TREATMENT_WITH_REGIMEN',  # Treatment guidelines
        
        # Context Relationships
        'HAS_FINDING_CONTEXT',          # Clinical context
        'MAY_BE_MOLECULAR_ABNORMALITY_OF_DISEASE',  # Molecular basis
    ]
}


# Semantic groups for filtering
SEMANTIC_GROUPS = {
    'ANAT': ['T017', 'T029', 'T023', 'T030', 'T031'],  # Anatomical Structure
    'DISO': ['T047', 'T048', 'T049', 'T050'],          # Disease or Syndrome
    'CHEM': ['T116', 'T121', 'T122', 'T123'],          # Chemical, Drug
    'PROC': ['T060', 'T061', 'T062', 'T063'],          # Therapeutic Procedure
    'PHYS': ['T039', 'T040', 'T041', 'T042'],          # Physiologic Function
    'SYMP': ['T184', 'T185'],                           # Sign or Symptom
    'CLIN_DECISION': [  # Step 2/3 focused
        'T041',  # Clinical Reasoning
        'T170',  # Guideline
        'T074'   # Medical Device
    ],
    'MOLEC_MECH': [     # Step 1 focused
        'T087',  # Amino Acid
        'T088',  # Molecular Structure
        'T085'   # Enzyme Reaction
    ]

}