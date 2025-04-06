import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from combined_usmle import USMLEProcessor

def initialize_processor():
    """Initialize the USMLE processor with all necessary connections"""
    try:
        # Add debugging output
        st.info("Initializing USMLE processor...")
        
        # Check environment variables
        import os
        load_dotenv()
        
        required_vars = [
            "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
            "OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENV"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return None
        
        # Verify Neo4j URI format and connectivity
        neo4j_uri = os.getenv("NEO4J_URI")
        st.info(f"Connecting to Neo4j at: {neo4j_uri}")
        
        # Check URI format
        if not neo4j_uri:
            st.error("NEO4J_URI environment variable is empty.")
            return None
            
        # Attempt to create a standalone Neo4j connection to verify it before creating processor
        try:
            from langchain_neo4j import Neo4jGraph
            
            # Try to connect with timeout
            import time
            start_time = time.time()
            timeout = 10  # seconds
            
            st.info(f"Testing Neo4j connection (timeout: {timeout}s)...")
            
            # Use a timeout for the connection test
            graph = Neo4jGraph(
                url=neo4j_uri,
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD"),
                database="neo4j"
            )
            
            # Simple query to test connection
            try:
                result = graph.query("RETURN 1 as test", {})
                st.success(f"Neo4j connection successful: {result}")
            except Exception as neo4j_query_error:
                st.error(f"Neo4j connection succeeded but query failed: {neo4j_query_error}")
                return None
                
        except Exception as neo4j_error:
            st.error(f"Neo4j connection failed: {neo4j_error}")
            st.error("""
            **Troubleshooting Neo4j connection:**
            1. Verify your NEO4J_URI is correct (should be in format 'bolt://hostname:port' or 'neo4j://hostname:port')
            2. Check your network connectivity to the Neo4j server
            3. Verify your Neo4j credentials (username and password)
            4. Make sure your Neo4j instance is running and accessible
            
            Common URIs:
            - Local: 'bolt://localhost:7687'
            - AuraDB: 'neo4j+s://[database-id].databases.neo4j.io:7687'
            """)
            return None
        
        # Now try to initialize the processor
        st.info("Neo4j connection successful, initializing USMLE processor...")
        
        # Create a dummy processor with basic functionality if full initialization fails
        try:
            processor = USMLEProcessor()
            st.info("USMLE processor initialized successfully")
            return processor
        except Exception as processor_error:
            st.error(f"Error initializing USMLE processor: {processor_error}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            
            # Provide fallback simple processor that only uses RAG
            st.warning("Setting up fallback processor with limited functionality...")
            
            # Create a dummy processor class with the process_question method
            from query_medical_db import query_medical_knowledge
            
            class SimpleFallbackProcessor:
                def process_question(self, question_text):
                    """Simple fallback processor that only uses the medical knowledge database"""
                    # Query the medical database
                    try:
                        from query_medical_db import query_medical_knowledge, openai_embeddings, index
                        
                        # Get query embedding
                        query_embedding = openai_embeddings.embed_query(question_text)
                        
                        # Get results from Pinecone
                        results = index.query(
                            vector=query_embedding,
                            top_k=8,
                            include_metadata=True
                        )
                        
                        # Format chunks for display
                        chunks = []
                        for match in results.matches:
                            if match.score < 0.5:
                                continue
                                
                            text = match.metadata.get('text', '').strip()
                            chunks.append({
                                'text': text,
                                'sources': match.metadata.get('source', 'Unknown'),
                                'score': match.score
                            })
                        
                        # Get top results
                        from query_medical_db import synthesize_medical_response, combine_similar_chunks
                        combined_results = combine_similar_chunks(chunks)
                        response = synthesize_medical_response(question_text, combined_results)
                        
                        # Return results in expected format
                        return {
                            'answer': response,
                            'kg_results': {
                                'concepts': [],
                                'relationships': []
                            },
                            'textbook_results': combined_results
                        }
                    except Exception as e:
                        return {
                            'answer': f"Error processing question with fallback method: {str(e)}",
                            'kg_results': {'concepts': [], 'relationships': []},
                            'textbook_results': []
                        }
            
            return SimpleFallbackProcessor()
            
    except Exception as e:
        st.error(f"Error initializing processor: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    # Set page config
    st.set_page_config(
        page_title="USMLE Question Processor",
        page_icon="🏥",
        layout="wide"
    )
    
    # Header
    st.title("🏥 USMLE Question Processor")
    st.markdown("""
    This application helps process and analyze USMLE-style medical questions using:
    - Medical Knowledge Graph (Neo4j)
    - Medical Textbook Database (Pinecone)
    - Advanced Language Models
    """)
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = initialize_processor()
    
    # Input area
    st.header("Enter Your Question")
    question_input = st.text_area(
        "Type or paste your USMLE question here:",
        height=200,
        help="Enter the complete question including all answer choices"
    )
    
    # Add manual reinitialization option
    if st.button("Reinitialize Processor"):
        with st.spinner("Reinitializing processor..."):
            st.session_state.processor = initialize_processor()
            if st.session_state.processor:
                st.success("Processor reinitialized successfully")
    
    # Process button
    if st.button("Process Question", type="primary"):
        if not question_input.strip():
            st.warning("Please enter a question first.")
            return
            
        # Check if processor is available
        if not st.session_state.processor:
            st.error("Processor not available. Please check the error messages above and reinitialize.")
            return
            
        with st.spinner("Processing your question..."):
            try:
                results = st.session_state.processor.process_question(question_input)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Answer", "Knowledge Graph", "Textbook References"])
                
                with tab1:
                    st.markdown("### Analysis and Answer")
                    st.markdown(results['answer'])
                
                with tab2:
                    st.markdown("### Knowledge Graph Evidence")
                    
                    # Display concepts
                    st.subheader("Relevant Medical Concepts")
                    for concept in results['kg_results'].get('concepts', []):
                        with st.expander(f"📌 {concept['term']}"):
                            st.markdown(f"**CUI:** {concept.get('cui', 'N/A')}")
                            if concept.get('definition'):
                                st.markdown(f"**Definition:** {concept['definition']}")
                    
                    # Display relationships
                    st.subheader("Key Relationships")
                    for rel in results['kg_results'].get('relationships', []):
                        st.markdown(
                            f"- {rel['source_name']} → "
                            f"**{rel['relationship_type']}** → "
                            f"{rel['target_name']}"
                        )
                
                with tab3:
                    st.markdown("### Textbook References")
                    for i, chunk in enumerate(results['textbook_results'], 1):
                        with st.expander(f"📚 Reference {i} (Score: {chunk['score']:.2f})"):
                            st.markdown(f"**Source:** {chunk['sources']}")
                            st.markdown(f"**Content:**\n{chunk['text']}")
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using Neo4j, Pinecone, and LangChain"
    )

if __name__ == "__main__":
    main() 