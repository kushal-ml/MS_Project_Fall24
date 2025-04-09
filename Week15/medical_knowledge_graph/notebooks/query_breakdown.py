from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import json
import logging

logger = logging.getLogger(__name__)

class PICOFormatter:
    def __init__(self, llm=None):
        if llm is None:
            # Initialize OpenAI client if not provided
            load_dotenv()
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0
            )
        else:
            self.llm = llm

        self.prompt_template = PromptTemplate(
            input_variables=["question"],
            template="""You are a medical term extractor. Extract medical terms from the given question and return ONLY a JSON object.

Rules:
1. Return ONLY the JSON object, no other text
2. Include empty arrays if no terms found for a category
3. Use exact terms found in the question
4. Do not include explanatory text or placeholders

JSON Format:
{
    "diseases": [],
    "symptoms": [],
    "tests": [],
    "medications": [],
    "conditions": [],
    "demographics": {
        "age": null,
        "sex": null
    }
}

Question: {question}"""
        )

    def analyze_medical_question(self, user_query: str) -> str:
        """Break down medical question into PICO components and extract key terms"""
        try:
            # Create the final prompt
            formatted_prompt = self.prompt_template.format(question=user_query)
            
            # Get response from OpenAI
            response = self.llm.invoke(formatted_prompt)
            
            # Clean the response content
            content = response.content.strip()
            
            # If response starts with ``` or ends with ```, remove them
            if content.startswith('```'):
                content = content[content.find('{'):content.rfind('}')+1]
            
            # Ensure we're returning valid JSON
            try:
                # Validate JSON by parsing
                json_response = json.loads(content)
                
                # Ensure all required keys exist
                default_structure = {
                    "diseases": [],
                    "symptoms": [],
                    "tests": [],
                    "medications": [],
                    "conditions": [],
                    "demographics": {
                        "age": None,
                        "sex": None
                    }
                }
                
                # Update default structure with any valid data from response
                for key in default_structure:
                    if key in json_response:
                        if key == "demographics":
                            default_structure[key].update(json_response[key])
                        else:
                            default_structure[key] = json_response[key]
                
                return json.dumps(default_structure)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)} in content: {content}")
                return json.dumps({
                    "diseases": [],
                    "symptoms": [],
                    "tests": [],
                    "medications": [],
                    "conditions": [],
                    "demographics": {
                        "age": None,
                        "sex": None
                    }
                })
                
        except Exception as e:
            logger.error(f"Error in PICO analysis: {str(e)}")
            return json.dumps({
                "diseases": [],
                "symptoms": [],
                "tests": [],
                "medications": [],
                "conditions": [],
                "demographics": {
                    "age": None,
                    "sex": None
                }
            })
