# Relationship types
IMPORTANT_RELATIONS = {
            'PAR': 'parent_of',
            'CHD': 'child_of',
            'RB': 'broader_than',
            'RN': 'narrower_than',
            'SY': 'synonym_of'
        }

USMLE_DOMAINS = {
            'SNOMED_CT': {
                'priority_1': {
                    'DISO': 'Disease',
                    'FIND': 'Clinical_Finding',
                    'SYMP': 'Symptom'
                }
            },
            'ICD10': {
                'priority_1': {
                    'A00-B99': 'Infectious_Diseases',
                    'C00-D49': 'Neoplasms',
                    'I00-I99': 'Circulatory_Diseases',
                    'J00-J99': 'Respiratory_Diseases',
                    'K00-K95': 'Digestive_Diseases'
                }
            },
            'RXNORM': {
                'priority_1': {
                    'IN': 'Ingredient',
                    'BN': 'Brand_Name'
                }
            },
            'MSH': {
                'priority_1': {
                    'ET': 'Disease',  # Add Entry Term
                    'MH': 'Disease',  # Add Main Heading
                    'PT': 'Disease'   # Add Preferred Term
            }
            },
        }

IMPORTANT_SEMANTIC_TYPE = {
    'T047': 'Disease or Syndrome',
    'T048': 'Mental/Behavioral Dysfunction',
    'T059': 'Laboratory Procedure',
    'T061': 'Therapeutic Procedure',
    'T121': 'Pharmacologic Substance',
    'T123': 'Biologically Active Substance',
    'T184': 'Sign or Symptom',
    'T190': 'Anatomical Structure',
    'T195': 'Antibiotic',
    'T033': 'Finding',
    'T034': 'Laboratory or Test Result',
    'T037': 'Injury or Poisoning',
    'T046': 'Pathologic Function',
    'T060': 'Diagnostic Procedure',
    'T074': 'Medical Device',
    'T200': 'Clinical Drug'

}