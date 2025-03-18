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
        processor = USMLEProcessor()
        return processor
    except Exception as e:
        st.error(f"Error initializing processor: {e}")
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
    
    # Process button
    if st.button("Process Question", type="primary"):
        if not question_input.strip():
            st.warning("Please enter a question first.")
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
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using Neo4j, Pinecone, and LangChain"
    )

if __name__ == "__main__":
    main() 