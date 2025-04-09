import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone
import time
from collections import defaultdict
from difflib import SequenceMatcher

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

def query_medical_knowledge(query_text, top_k=8):
    """Query medical knowledge base and return relevant information"""
    try:
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
            if match.score < 0.5:  # Reduced threshold from 0.7 to 0.5
                continue
                
            text = match.metadata.get('text', '').strip()
            if text.startswith('.'):  # Clean fragmented sentences
                text = text[1:].strip()
            
            chunks.append({
                'text': text,
                'source': match.metadata.get('source', ''),
                'score': match.score
            })

        if not chunks:
            print("\n⚠️ No highly relevant information found in the medical database.")
            return

        # Combine similar chunks
        combined_results = combine_similar_chunks(chunks)
        
        print(f"\n📚 Medical Query: {query_text}")
        print("\n🔍 Retrieving and synthesizing information from medical sources...")
        
        # Generate synthesized response using GPT-4
        synthesized_response = synthesize_medical_response(query_text, combined_results)
        
        print("\n📑 Medical Response:")
        print(synthesized_response)
            
    except Exception as e:
        print(f"Error querying database: {e}")
        raise

def interactive_query():
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
            
        query_medical_knowledge(query)
        time.sleep(0.5)  # Rate limiting

if __name__ == "__main__":
    try:
        interactive_query()
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}") 