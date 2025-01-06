import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from langchain.chat_models import ChatOpenAI
import json

logger = logging.getLogger(__name__)

class MedicalTermExtractor:
    def __init__(self):
        # Initialize Bio-Epidemiology NER model
        try:
            self.ner_tokenizer = AutoTokenizer.from_pretrained("jayantd/Bio-Epidemiology-NER")
            self.ner_model = AutoModelForTokenClassification.from_pretrained("jayantd/Bio-Epidemiology-NER")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {str(e)}")
            self.ner_model = None
            
        # Initialize LLM for direct/indirect extraction
        self.llm = ChatOpenAI(temperature=0)
        
    def extract_terms_direct(self, question: str) -> list:
        """Direct extraction of medical terms from question"""
        prompt = f"""
        Only return the medical terminologies contained in the input question.
        Please return in JSON format.
        Output Format:
        {{
            "medical_terminologies": ["<name>", "<name>"]
        }}
        Please only return the JSON format information.
        Input: {question}
        """
        
        try:
            response = self.llm.predict(prompt)
            terms = json.loads(response)['medical_terminologies']
            logger.info(f"Direct extraction found {len(terms)} terms")
            return terms
        except Exception as e:
            logger.error(f"Direct extraction failed: {str(e)}")
            return []

    def extract_terms_indirect(self, question: str) -> list:
        """Indirect extraction of related medical terms"""
        prompt = f"""
        Return medical terminologies related to the input question.
        Please return in JSON format.
        Output Format:
        {{
            "medical_terminologies": ["<name>", "<name>"]
        }}
        Please only return the JSON format information.
        Input: {question}
        """
        
        try:
            response = self.llm.predict(prompt)
            terms = json.loads(response)['medical_terminologies']
            logger.info(f"Indirect extraction found {len(terms)} terms")
            return terms
        except Exception as e:
            logger.error(f"Indirect extraction failed: {str(e)}")
            return []

    def extract_terms_ner(self, question: str) -> list:
        """Extract terms using Bio-Epidemiology NER model"""
        if not self.ner_model:
            return []
            
        try:
            inputs = self.ner_tokenizer(question, return_tensors="pt", padding=True)
            outputs = self.ner_model(**inputs)
            
            # Process NER results
            predictions = outputs.logits.argmax(dim=2)
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Extract medical entities
            entities = []
            current_entity = []
            
            for token, pred in zip(tokens, predictions[0]):
                if pred != 0:  # Non-O tag
                    current_entity.append(token)
                elif current_entity:
                    entities.append("".join(current_entity).replace("##", ""))
                    current_entity = []
                    
            logger.info(f"NER extraction found {len(entities)} terms")
            return entities
            
        except Exception as e:
            logger.error(f"NER extraction failed: {str(e)}")
            return []

    def extract_all_terms(self, question: str) -> dict:
        """Extract terms using all methods"""
        results = {
            'direct': self.extract_terms_direct(question),
            'indirect': self.extract_terms_indirect(question),
            'ner': self.extract_terms_ner(question)
        }
        
        # Combine and deduplicate terms
        all_terms = list(set(
            results['direct'] + 
            results['indirect'] + 
            results['ner']
        ))
        
        return {
            'individual_results': results,
            'combined_terms': all_terms
        }