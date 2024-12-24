# Relationship types
IMPORTANT_RELATIONS = {
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

USMLE_DOMAINS = {
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