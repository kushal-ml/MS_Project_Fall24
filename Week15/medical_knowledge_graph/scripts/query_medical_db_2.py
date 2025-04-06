import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone
import time
from collections import defaultdict
from difflib import SequenceMatcher
import re
import argparse
import sys

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# Connect to existing index
index_name = "medical-textbook-embeddings"
index = pc.Index(index_name)

# Initialize GPT-4
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

# Update the RAG prompt to be more precise
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical information synthesizer. Your role is to ONLY use the provided reference information to answer queries.
    IMPORTANT RULES:
    - ONLY use information that is EXPLICITLY present in the provided references
    - DO NOT make any inferences or use external medical knowledge
    - When citing, use exact quotes from the references and indicate the chunk number
    - If information needed to answer the question is missing from the references, explicitly state this
    - Format response in these sections:
        1. Correct answer choice
        2. Direct Evidence from References (with exact quotes) that supports your reasoning that led to the correct answer choice
        3. Missing Information (what we need but don't have)
        4. Conclusion (based ONLY on available evidence)
    
    If you cannot find explicit information to answer the question, say: "The provided references do not contain sufficient direct evidence to answer this question."
    """),
    ("user", """Question: {question}

Available Reference Information:
{context}

Remember: Only use information explicitly stated in these references.""")
])

# NEW: Clinical case extraction prompt
CLINICAL_CASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical case analyzer. Extract structured information from clinical case presentations.
    If information for a category is not present, write "None mentioned"."""),
    ("user", """Extract the following information from this clinical case in a structured format:
    
    CLINICAL CASE:
    {case_text}
    
    FORMAT:
    Patient: [Age, Sex, Pre-existing conditions]
    Investigations: [Vital signs, Lab results, Imaging findings]
    Key_Findings: [Most significant abnormal findings, pathology results]
    Symptoms: [Symptoms with duration]
    Additional_Factors: [Lifestyle, medications, social history]
    Question_Focus: [What is being asked specifically]
    Options: [Answer choices]
    
    Keep it concise but include all relevant clinical details.""")
])

def extract_clinical_case_structure(case_text):
    """Extract structured information from a clinical case using LLM"""
    try:
        response = llm.invoke(
            CLINICAL_CASE_PROMPT.format(
                case_text=case_text
            )
        )
        
        # Parse the structured response
        structure = {}
        current_key = None
        
        for line in response.content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                structure[key.strip()] = value.strip()
        
        print("\n📋 Extracted Clinical Case Structure:")
        for key, value in structure.items():
            print(f"  • {key}: {value}")
            
        return structure
        
    except Exception as e:
        print(f"Error extracting case structure: {e}")
        # Return a basic structure to allow system to continue
        return {
            "Patient": "",
            "Investigations": "",
            "Key_Findings": "",
            "Symptoms": "",
            "Additional_Factors": "",
            "Question_Focus": case_text,
            "Options": ""
        }

def is_clinical_case(text):
    """Determine if a query is likely a clinical case presentation"""
    # Check for common clinical case indicators
    indicators = [
        r'\b\d+(-|\s)year(-|\s)old\b',  # Age pattern
        r'\b(man|woman|male|female|patient)\b',  # Patient descriptors
        r'\b(presents|presented|comes|admitted)\b',  # Presentation verbs
        r'\b(temperature|pulse|respirations|blood pressure)\b',  # Vital signs
        r'\b(lab|laboratory|results|findings)\b',  # Test references
        r'\b(A\)|B\)|C\)|D\)|E\))',  # Multiple choice markers
        r'most likely|diagnosis|treatment|cause|mechanism'  # Question patterns
    ]
    
    # Count how many indicators are present
    indicator_count = sum(1 for pattern in indicators if re.search(pattern, text, re.IGNORECASE))
    
    # If more than 2 indicators, likely a clinical case
    return indicator_count >= 2

def similarity_score(text1, text2):
    """Calculate text similarity using SequenceMatcher"""
    return SequenceMatcher(None, text1, text2).ratio()

def combine_similar_chunks(chunks, similarity_threshold=0.3):
    """Combine similar text chunks into coherent passages"""
    combined_chunks = []
    used_chunks = set()

    for i, chunk in enumerate(chunks):
        if i in used_chunks:
            continue

        related_text = [chunk['text']]
        used_chunks.add(i)

        # Look for similar chunks
        for j, other_chunk in enumerate(chunks):
            if j not in used_chunks:
                if (similarity_score(chunk['text'], other_chunk['text']) > similarity_threshold or
                    any(text in other_chunk['text'] for text in related_text) or
                    any(text in chunk['text'] for text in [other_chunk['text']])):
                    
                    related_text.append(other_chunk['text'])
                    used_chunks.add(j)

        # Combine related chunks and their sources
        combined_text = ' '.join(related_text)
        sources = list(set(chunks[i]['source'] for i in used_chunks))
        avg_score = sum(chunks[i]['score'] for i in used_chunks) / len(used_chunks)

        combined_chunks.append({
            'text': combined_text,
            'sources': sources,
            'score': avg_score
        })

    return combined_chunks

def synthesize_medical_response(question, context_chunks):
    """Use GPT-4 to synthesize a coherent response strictly from context chunks"""
    # Prepare context with clear source markers
    context = "\n\n".join([
        f"[REF{i+1}] Source: {', '.join(chunk['sources'])}\nRelevance Score: {chunk['score']:.2f}\nContent: {chunk['text'][:1000]}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    # Generate response using GPT-4
    response = llm.invoke(
        RAG_PROMPT.format(
            question=question,
            context=context
        )
    )
    
    # Add a separator between response and debug information
    debug_info = "\n\n---DEBUG: Retrieved Chunks---\n"
    for i, chunk in enumerate(context_chunks):
        debug_info += f"\nChunk {i+1} (Score: {chunk['score']:.2f}):\n"
        debug_info += f"Source: {chunk['sources']}\n"
        debug_info += f"First 500 chars: {chunk['text'][:500]}...\n"
    
    return response.content + debug_info

def enhanced_clinical_retrieval(case_text, top_k=8):
    """Multi-strategy retrieval for clinical cases"""
    # Extract structured case information
    structure = extract_clinical_case_structure(case_text)
    
    # If extraction failed, fall back to standard retrieval
    if not structure:
        print("⚠️ Case structure extraction failed, falling back to standard retrieval")
        return standard_retrieval(case_text, top_k)
    
    # Create targeted retrieval queries
    retrieval_strategies = [
        # Strategy 1: Key findings + Question focus
        {
            "query": f"{structure.get('Key_Findings', '')} {structure.get('Question_Focus', '')}",
            "weight": 0.35,
            "description": "Key diagnostic findings"
        },
        # Strategy 2: Pathophysiology focus
        {
            "query": f"pathophysiology {structure.get('Key_Findings', '')} {structure.get('Question_Focus', '')}",
            "weight": 0.25,
            "description": "Underlying mechanisms"
        },
        # Strategy 3: Patient-specific context
        {
            "query": f"{structure.get('Patient', '')} with {structure.get('Symptoms', '')} {structure.get('Key_Findings', '')}",
            "weight": 0.15,
            "description": "Clinical context"
        },
        # Strategy 4: Differential diagnosis
        {
            "query": f"differential diagnosis {structure.get('Key_Findings', '')} {structure.get('Symptoms', '')}",
            "weight": 0.25,
            "description": "Diagnostic alternatives"
        }
    ]
    
    # Execute all retrieval strategies
    all_results = []
    
    for strategy in retrieval_strategies:
        print(f"\n🔍 Executing retrieval strategy: {strategy['description']}")
        print(f"   Query: {strategy['query']}")
        
        # Get embeddings for this strategy
        query_embedding = openai_embeddings.embed_query(strategy['query'])
        
        # Retrieve from vector store
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Process and weight results
        for match in results.matches:
            if match.score < 0.5:  # Keep threshold check
                continue
                
            text = match.metadata.get('text', '').strip()
            if text.startswith('.'):  # Clean fragmented sentences
                text = text[1:].strip()
            
            # Create weighted score
            weighted_score = match.score * strategy['weight']
            
            # Add to results with strategy info
            all_results.append({
                'text': text,
                'source': match.metadata.get('source', ''),
                'score': weighted_score,
                'strategy': strategy['description'],
                'original_score': match.score
            })
    
    # Combine and deduplicate results
    combined_results = {}
    
    for result in all_results:
        # Use text as deduplication key
        text = result['text']
        
        if text in combined_results:
            # If we've seen this text before, keep the highest weighted score
            if result['score'] > combined_results[text]['score']:
                combined_results[text] = result
        else:
            combined_results[text] = result
    
    # Sort by score and convert to standard format
    sorted_results = sorted(combined_results.values(), 
                           key=lambda x: x['score'], reverse=True)
    
    # Convert to the expected format for further processing
    standard_format = []
    for item in sorted_results[:top_k]:
        standard_format.append({
            'text': item['text'],
            'source': item['source'],
            'score': item['score']
        })
    
    return standard_format

def standard_retrieval(query_text, top_k=8):
    """Standard vector similarity retrieval"""
    print(f"\n🔍 Performing standard vector retrieval for: {query_text[:100]}...")
    
    # Get query embedding
    query_embedding = openai_embeddings.embed_query(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Prepare chunks for combination
    chunks = []
    for match in results.matches:
        if match.score < 0.5:  # Threshold check
            continue
                
        text = match.metadata.get('text', '').strip()
        if text.startswith('.'):  # Clean fragmented sentences
            text = text[1:].strip()
        
        chunks.append({
            'text': text,
            'source': match.metadata.get('source', ''),
            'score': match.score
        })
    
    return chunks

def query_medical_knowledge(query_text, top_k=8):
    """Query medical knowledge base and return relevant information"""
    try:
        print(f"\n📚 Medical Query: {query_text}")
        
        # Detect if this is a clinical case presentation
        if is_clinical_case(query_text):
            print("\n🏥 Detected clinical case presentation - using enhanced retrieval")
            chunks = enhanced_clinical_retrieval(query_text, top_k)
        else:
            print("\n📖 Standard query detected - using direct vector retrieval")
            chunks = standard_retrieval(query_text, top_k)

        if not chunks:
            print("\n⚠️ No highly relevant information found in the medical database.")
            return

        # Combine similar chunks
        combined_results = combine_similar_chunks(chunks)
        
        print(f"\n🔍 Retrieved {len(chunks)} relevant chunks, combined into {len(combined_results)} passages")
        
        # Generate synthesized response using GPT-4
        print("\n🧠 Synthesizing response from retrieved information...")
        synthesized_response = synthesize_medical_response(query_text, combined_results)
        
        print("\n📑 Medical Response:")
        print(synthesized_response)
            
    except Exception as e:
        print(f"Error querying database: {e}")
        raise

def process_file_questions(file_path, top_k=8):
    """Process medical questions from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            questions = file.read().strip().split('\n\n')  # Assuming questions are separated by blank lines
            
        print(f"\n📚 Loaded {len(questions)} questions from {file_path}")
        
        for i, question in enumerate(questions, 1):
            if not question.strip():
                continue
                
            print(f"\n\n{'='*80}")
            print(f"Processing Question {i}/{len(questions)}")
            print(f"{'='*80}\n")
            
            start_time = time.time()
            query_medical_knowledge(question, top_k)
            end_time = time.time()
            
            print(f"\n⏱️ Query processing time: {end_time - start_time:.2f} seconds")
            print(f"\n{'='*80}")
            
            # Brief pause between processing questions to avoid rate limits
            time.sleep(1)
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

def interactive_query(top_k=8):
    """Interactive query interface"""
    print("\n🏥 Welcome to the Enhanced Medical Knowledge Base Query System!")
    print("Powered by GPT-4 and RAG technology")
    print("Type 'quit' or 'exit' to end the session")
    
    while True:
        query = input("\n🔍 Enter your medical question: ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("\nThank you for using the Medical Knowledge Base!")
            break
            
        if not query:
            print("Please enter a valid question!")
            continue
            
        start_time = time.time()
        query_medical_knowledge(query, top_k)
        end_time = time.time()
        print(f"\n⏱️ Query processing time: {end_time - start_time:.2f} seconds")
        time.sleep(0.5)  # Rate limiting

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Medical Knowledge Base Query System')
    
    # Define input mode group (file or interactive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-f', '--file', type=str, help='Path to text file containing medical questions')
    input_group.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode (default)')
    
    # Additional parameters
    parser.add_argument('-k', '--top_k', type=int, default=8, help='Number of top passages to retrieve (default: 8)')
    parser.add_argument('-o', '--output', type=str, help='Path to save outputs (not implemented yet)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process based on mode
    if args.file:
        process_file_questions(args.file, args.top_k)
    else:
        # Default to interactive mode
        interactive_query(args.top_k)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")