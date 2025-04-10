import streamlit as st
import sys
from pathlib import Path
import os
import json
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Import the USMLE processor from your existing file
sys.path.append(str(Path(__file__).parent))
from combined_usmle_2 import USMLEProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the processor
@st.cache_resource
def get_processor():
    return USMLEProcessor()

def format_metrics_table(evaluation):
    """Format evaluation metrics as a markdown table"""
    try:
        # Extract metrics
        llm_only_score = evaluation['evidence_based_evaluation']['llm_only'].get('combined_score', 0)
        context_strict_score = evaluation['evidence_based_evaluation']['context_strict'].get('combined_score', 0)
        llm_informed_score = evaluation['evidence_based_evaluation']['llm_informed'].get('combined_score', 0)
        
        # Extract evidence and correctness scores
        llm_only_evidence = evaluation['evidence_based_evaluation']['llm_only'].get('evidence_score', 0)
        context_strict_evidence = evaluation['evidence_based_evaluation']['context_strict'].get('evidence_score', 0)
        llm_informed_evidence = evaluation['evidence_based_evaluation']['llm_informed'].get('evidence_score', 0)
        
        llm_only_correctness = evaluation['evidence_based_evaluation']['llm_only'].get('correctness_score', 0)
        context_strict_correctness = evaluation['evidence_based_evaluation']['context_strict'].get('correctness_score', 0)
        llm_informed_correctness = evaluation['evidence_based_evaluation']['llm_informed'].get('correctness_score', 0)
        
        kg_coverage = evaluation.get('kg_coverage', {}).get('coverage_percentage', 0)
        value_added = evaluation.get('context_contribution', {}).get('value_added_score', 0)
        
        # Format as a DataFrame for display
        metrics_df = pd.DataFrame({
            'Metric': ['Combined Score', 'Evidence Score', 'Correctness Score'],
            'LLM Only': [f"{llm_only_score:.1f}/10", f"{llm_only_evidence:.1f}/10", f"{llm_only_correctness:.1f}/10"],
            'Context Strict': [f"{context_strict_score:.1f}/10", f"{context_strict_evidence:.1f}/10", f"{context_strict_correctness:.1f}/10"],
            'LLM Informed': [f"{llm_informed_score:.1f}/10", f"{llm_informed_evidence:.1f}/10", f"{llm_informed_correctness:.1f}/10"]
        })
        
        additional_metrics = pd.DataFrame({
            'Metric': ['KG Coverage', 'Value Added by Context'],
            'Value': [f"{kg_coverage:.1f}%", f"{value_added:.1f}/10"]
        })
        
        return metrics_df, additional_metrics
    
    except Exception as e:
        logger.error(f"Error formatting metrics: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def plot_metrics_comparison(evaluation):
    """Create a bar chart comparing the answer types"""
    try:
        # Extract metrics
        llm_only_score = float(evaluation['evidence_based_evaluation']['llm_only'].get('combined_score', 0))
        context_strict_score = float(evaluation['evidence_based_evaluation']['context_strict'].get('combined_score', 0))
        llm_informed_score = float(evaluation['evidence_based_evaluation']['llm_informed'].get('combined_score', 0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        answer_types = ['LLM Only', 'Context Strict', 'LLM Informed']
        scores = [llm_only_score, context_strict_score, llm_informed_score]
        
        bars = ax.bar(answer_types, scores, color=['blue', 'green', 'red'])
        ax.set_ylim(0, 10)
        ax.set_ylabel('Quality Score (0-10)')
        ax.set_title('Answer Quality Comparison')
        ax.axhline(y=5, color='gray', linestyle='--')  # Add a reference line at score=5
        
        # Add score labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error creating metrics comparison plot: {str(e)}")
        return None

def plot_evidence_vs_correctness(evaluation):
    """Create a scatter plot of evidence vs correctness scores"""
    try:
        # Extract evidence and correctness scores
        llm_only_evidence = float(evaluation['evidence_based_evaluation']['llm_only'].get('evidence_score', 0))
        context_strict_evidence = float(evaluation['evidence_based_evaluation']['context_strict'].get('evidence_score', 0))
        llm_informed_evidence = float(evaluation['evidence_based_evaluation']['llm_informed'].get('evidence_score', 0))
        
        llm_only_correctness = float(evaluation['evidence_based_evaluation']['llm_only'].get('correctness_score', 0))
        context_strict_correctness = float(evaluation['evidence_based_evaluation']['context_strict'].get('correctness_score', 0))
        llm_informed_correctness = float(evaluation['evidence_based_evaluation']['llm_informed'].get('correctness_score', 0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot points
        ax.scatter(llm_only_evidence, llm_only_correctness, color='blue', s=100, label='LLM Only')
        ax.scatter(context_strict_evidence, context_strict_correctness, color='green', s=100, label='Context Strict')
        ax.scatter(llm_informed_evidence, llm_informed_correctness, color='red', s=100, label='LLM Informed')
        
        ax.set_xlabel('Evidence Score (0-10)')
        ax.set_ylabel('Correctness Score (0-10)')
        ax.set_title('Evidence vs Correctness Comparison')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        # Draw quadrant lines
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
        
        # Label quadrants
        ax.text(7.5, 7.5, 'High Evidence\nHigh Correctness', ha='center')
        ax.text(2.5, 7.5, 'Low Evidence\nHigh Correctness', ha='center')
        ax.text(7.5, 2.5, 'High Evidence\nLow Correctness', ha='center')
        ax.text(2.5, 2.5, 'Low Evidence\nLow Correctness', ha='center')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error creating evidence vs correctness plot: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="USMLE Medical Question Processor", layout="wide")
    
    # Set theme to light
    st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff;
            }
            .stTextInput > div > div > input {
                background-color: #ffffff;
            }
            .stTextArea > div > div > textarea {
                background-color: #ffffff;
            }
            .stSelectbox > div > div > div {
                background-color: #ffffff;
            }
            .stButton > button {
                background-color: #ffffff;
                color: #000000;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Set up the sidebar
    st.sidebar.title("USMLE Medical Question Processor")
    st.sidebar.markdown("This app processes medical questions using a knowledge graph, vector database, and LLM.")
    
    st.sidebar.markdown("## About")
    st.sidebar.markdown("""
    This application:
    1. Processes USMLE-style medical questions
    2. Retrieves relevant medical concepts from a knowledge graph
    3. Extracts textbook passages from a vector database
    4. Generates three types of answers for comparison
    """)
    
    # Main content
    st.title("USMLE Medical Question Processor")
    
    # Initialize processor
    with st.spinner("Initializing medical knowledge processor..."):
        processor = get_processor()
        st.success("Processor initialized successfully!")
    
    # Input section
    st.markdown("## Enter Medical Question")
    
    # Sample questions dropdown
    sample_questions = [
        "Select a sample question...",
        "A 65-year-old man presents to the emergency department with sudden onset crushing chest pain that radiates to his left arm and jaw. The pain began 2 hours ago while he was resting. He is diaphoretic and nauseated. ECG shows ST-segment elevation in leads V1-V4. Which of the following is the most likely diagnosis?",
        "A 67-year-old woman with congenital bicuspid aortic valve is admitted to the hospital because of a 2-day history of fever and chills. Current medication is lisinopril. Temperature is 38.0°C (100.4°F), pulse is 90/min, respirations are 20/min, and blood pressure is 110/70 mm Hg. Cardiac examination shows a grade 3/6 systolic murmur that is best heard over the second right intercostal space. Blood culture grows viridans streptococci susceptible to penicillin. In addition to penicillin, an antibiotic synergistic to penicillin is administered that may help shorten the duration of this patient's drug treatment. Which of the following is the most likely mechanism of action of this additional antibiotic on bacteria?",
        "A 12-year-old girl is brought to the physician because of a 2-month history of intermittent yellowing of the eyes and skin. Physical examination shows no abnormalities except for jaundice. Her serum total bilirubin concentration is 3 mg/dL, with a direct component of 1 mg/dL. Serum studies show a haptoglobin concentration and AST and ALT activities that are within the reference ranges. There is no evidence of injury or exposure to toxins. Which of the following additional findings is most likely in this patient?"
    ]
    
    selected_sample = st.selectbox("Choose a sample question or enter your own below:", sample_questions)
    
    if selected_sample != "Select a sample question...":
        user_question = selected_sample
    else:
        user_question = ""
        
    user_question = st.text_area("Enter your USMLE-style medical question:", user_question, height=150)
    
    # Process button
    if st.button("Process Question"):
        if not user_question or user_question == "Select a sample question...":
            st.error("Please enter a question to process.")
        else:
            # Process the question
            with st.spinner("Processing your question... This may take a minute or two."):
                start_time = time.time()
                result = processor.process_question(user_question)
                processing_time = time.time() - start_time
                
                # Store result in session state for tab access
                st.session_state.result = result
                st.session_state.processing_time = processing_time
                
                # Evaluate answer
                evaluation = processor.evaluate_evidence_based_answer(result)
                st.session_state.evaluation = evaluation
            
            st.success(f"Question processed in {processing_time:.2f} seconds!")
            
            # Create tabs for different sections
            tabs = st.tabs([
                "🔍 Answers", 
                "📚 Knowledge Graph", 
                "📖 Textbook Passages"
            ])
            
            # Tab 1: Answers
            with tabs[0]:
                st.markdown("## Question")
                st.write(user_question)
                
                st.markdown("### 🤔 LLM-Only Answer (No Knowledge Graph)")
                st.write(result['answers']['llm_only'])
                
                st.markdown("### 📚 Context-Strict Answer (Only Knowledge Graph + Textbook Data)")
                st.write(result['answers']['context_strict'])
                
                st.markdown("### ✨ LLM-Informed Answer (Combined Knowledge)")
                st.write(result['answers']['llm_informed'])
            
            # Tab 2: Knowledge Graph Data
            with tabs[1]:
                st.markdown("## Knowledge Graph Data Retrieved")
                
                st.markdown("### Extracted Terms")
                if 'terms' in result['kg_results']:
                    terms = result['kg_results']['terms']
                    # Check the type of terms and handle appropriately
                    if isinstance(terms, list):
                        if all(isinstance(term, str) for term in terms):
                            # If it's a list of strings
                            st.write(", ".join(terms))
                        else:
                            # If it's a list of dictionaries or other objects
                            try:
                                # Try to extract term value if it's a dict
                                term_strings = [t['term'] if isinstance(t, dict) and 'term' in t else str(t) for t in terms]
                                st.write(", ".join(term_strings))
                            except:
                                # Fall back to displaying raw data
                                st.write(terms)
                else:
                    st.write("No terms extracted.")
                
                st.markdown("### Top Medical Concepts")
                concepts = result['kg_results']['concepts']
                if concepts:
                    concept_df = pd.DataFrame([
                        {
                            "CUI": c.get('cui', 'Unknown'),
                            "Term": c.get('term', 'Unknown'),
                            "Definition": c.get('definition', 'No definition available')[:100] + "..." 
                            if c is not None and len(c.get('definition') or 'No definition available') > 100 
                            else c.get('definition', 'No definition available'),
                            "Score": c.get('score', 0)
                        }
                        for c in concepts[:15]  # Display top 15
                    ])
                    st.dataframe(concept_df)
                    st.write(f"Showing top 15 of {len(concepts)} concepts")
                else:
                    st.write("No concepts retrieved.")
                
                st.markdown("### Top Relationships")
                relationships = result['kg_results']['relationships']
                if relationships:
                    rel_df = pd.DataFrame([
                        {
                            "Source": r.get('source_name', 'Unknown'),
                            "Relationship": r.get('relationship_type', 'related_to'),
                            "Target": r.get('target_name', 'Unknown'),
                            "Score": r.get('score', 0)
                        }
                        for r in relationships[:20]  # Display top 20
                    ])
                    st.dataframe(rel_df)
                    st.write(f"Showing top 20 of {len(relationships)} relationships")
                else:
                    st.write("No relationships retrieved.")
                
                # New Section: Multihop Paths
                st.markdown("### Multihop Paths")
                multihop_paths = result['kg_results'].get('multihop_paths', [])
                if multihop_paths:
                    # Create a DataFrame to display the paths in a tabular format
                    path_data = []
                    for i, path in enumerate(multihop_paths):
                        # Check if path is a dictionary with structured data
                        if isinstance(path, dict):
                            path_data.append({
                                "Path #": i+1,
                                "Source": path.get('source_term', 'Unknown'),
                                "Target": path.get('target_term', 'Unknown'),
                                "Length": path.get('path_length', 'N/A'),
                                "Description": path.get('path_description', 'N/A'),
                                "Relevance": f"{path.get('relevance_score', 0):.3f}"
                            })
                        # If it's just a list of nodes
                        elif isinstance(path, list):
                            path_data.append({
                                "Path #": i+1,
                                "Path": " → ".join(path)
                            })
                    
                    if path_data:
                        # Display as a DataFrame
                        st.dataframe(pd.DataFrame(path_data))
                        st.write(f"Showing {len(path_data)} multihop paths")
                        
                        # Optionally add a more detailed view for a few top paths
                        st.markdown("#### Top Path Details")
                        for i, path in enumerate(multihop_paths[:3]):  # Show top 3 paths
                            if isinstance(path, dict):
                                st.markdown(f"**Path {i+1}:** {path.get('source_term', '')} → {path.get('target_term', '')}")
                                st.markdown(f"**Description:** {path.get('path_description', 'N/A')}")
                                
                                # If path_nodes and path_rels are available, display the full path
                                if 'path_nodes' in path and 'path_rels' in path:
                                    nodes = path.get('path_nodes', [])
                                    rels = path.get('path_rels', [])
                                    
                                    if nodes and len(nodes) > 1:
                                        full_path = []
                                        for j in range(len(nodes)-1):
                                            full_path.append(f"({nodes[j]})-[{rels[j]}]->")
                                        full_path.append(f"({nodes[-1]})")
                                        st.markdown(f"**Full Path:** {''.join(full_path)}")
                                
                                st.markdown(f"**Relevance Score:** {path.get('relevance_score', 0):.3f}")
                                st.markdown("---")
                    else:
                        st.write("No structured path data available.")
                else:
                    st.write("No multihop paths found.")
            
            # Tab 3: Textbook Passages
            with tabs[2]:
                st.markdown("## Textbook Passages Retrieved")
                
                textbook_context = result['pinecone_results']
                if textbook_context:
                    sections = textbook_context.split("[SOURCE")
                    
                    for section in sections[1:]:  # Skip the first empty split
                        section_title = section.split("]", 1)[0].strip()
                        section_content = section.split("]", 1)[1].strip()
                        
                        relevance = "Unknown"
                        if "Relevance:" in section_content:
                            relevance_parts = section_content.split("Relevance:", 1)
                            relevance = relevance_parts[1].split("\n", 1)[0].strip()
                            section_content = relevance_parts[1].split("\n", 1)[1].strip()
                        
                        st.markdown(f"### Source {section_title}")
                        st.markdown(f"**Relevance Score**: {relevance}")
                        st.markdown(f"**Content**:")
                        st.markdown(section_content)
                        st.markdown("---")
                else:
                    st.write("No textbook passages retrieved.")

if __name__ == "__main__":
    main()