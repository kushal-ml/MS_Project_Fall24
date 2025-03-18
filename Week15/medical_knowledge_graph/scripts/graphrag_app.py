import streamlit as st
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import networkx as nx
import time
from dotenv import load_dotenv
from PIL import Image

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

# Import the USMLEProcessor
from combined_usmle_2 import USMLEProcessor

# Set page configuration
st.set_page_config(
    page_title="Medical Knowledge Graph Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Base styles */
    .main-header {
        font-size: 2.5rem;
        color: #3498db !important; /* Bright blue - visible on dark backgrounds */
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2ecc71 !important; /* Bright green - visible on dark backgrounds */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Card styling for dark theme */
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.1) !important; /* Semi-transparent light background */
        border-left: 4px solid #3498db;
        color: #ffffff !important; /* White text for dark background */
    }
    
    .evaluation-card {
        background-color: rgba(46, 204, 113, 0.1) !important; /* Semi-transparent green background */
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2ecc71;
        color: #ffffff !important; /* White text for dark background */
    }
    
    /* Tags styling */
    .concept-tag {
        background-color: rgba(52, 152, 219, 0.3) !important; /* Semi-transparent blue */
        color: #ffffff !important;
        padding: 0.3rem 0.6rem;
        border-radius: 16px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-size: 0.85rem;
        border: 1px solid rgba(52, 152, 219, 0.7);
    }
    
    .relationship-tag {
        background-color: rgba(243, 156, 18, 0.3) !important; /* Semi-transparent orange */
        color: #ffffff !important;
        padding: 0.3rem 0.6rem;
        border-radius: 16px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-size: 0.85rem;
        border: 1px solid rgba(243, 156, 18, 0.7);
    }
    
    /* Text containers for dark theme */
    .answer-container {
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
    }
    
    .source-text {
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 1rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        border-left: 3px solid #f39c12;
        color: #ffffff !important;
    }
    
    /* General text colors for dark theme */
    p, li, h1, h2, h3, h4, h5, h6, span, div {
        color: #ffffff !important;
    }
    
    /* Info text */
    .info-text {
        color: #bdc3c7 !important;
        font-size: 0.9rem;
    }
    
    /* For Streamlit's elements */
    .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Make dataframe headers more visible */
    .dataframe th {
        background-color: rgba(52, 152, 219, 0.3) !important;
        color: white !important;
    }
    
    /* Highlighted content */
    .highlight {
        background-color: rgba(46, 204, 113, 0.2);
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        color: #ffffff !important;
    }
    
    /* Code blocks - light background with dark text for better readability */
    pre, code {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        padding: 0.5rem;
        border-radius: 4px;
    }
    
    /* For st.code elements */
    .stCodeBlock {
        background-color: #2d3748 !important;
        color: #f7fafc !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_processor():
    """Initialize the USMLEProcessor (cached to avoid reinitialization)"""
    load_dotenv()
    return USMLEProcessor()

def render_knowledge_graph(concepts, relationships):
    """Render a visual representation of the knowledge graph with improved connections"""
    G = nx.DiGraph()
    
    # First, collect all unique terms from both concepts and relationships
    all_terms = set()
    concept_terms = {}
    
    # Add nodes (concepts) - include more concepts for better connectivity
    for i, concept in enumerate(concepts[:20]):  # Increased from 15 to 20
        term = concept.get('term', f'Concept {i}')
        concept_terms[concept.get('cui', f'cui_{i}')] = term
        all_terms.add(term)
        G.add_node(term, type='concept')
    
    # Track added edges to avoid duplicates
    added_edges = set()
    
    # Add edges (relationships) - process more relationships
    for rel in relationships[:60]:  # Increased from 30 to 60
        source = rel.get('source_name', '')
        target = rel.get('target_name', '')
        rel_type = rel.get('relationship_type', '')
        
        # Only add if terms are valid and edge hasn't been added yet
        edge_key = (source, target, rel_type)
        if source and target and edge_key not in added_edges:
            # Add missing nodes if needed
            if source not in all_terms and source:
                G.add_node(source, type='concept')
                all_terms.add(source)
            
            if target not in all_terms and target:
                G.add_node(target, type='concept')
                all_terms.add(target)
            
            # Add the edge
            G.add_edge(source, target, type=rel_type)
            added_edges.add(edge_key)
    
    # If we have multihop paths, add those connections too (from the result object)
    # This would require passing multihop_paths as a parameter - optional enhancement
    
    # Check if we have a reasonable number of edges
    if len(G.edges()) < 5 and len(relationships) > 5:
        # If too few edges, create some basic connections between nodes
        # to show a more connected graph
        nodes = list(G.nodes())
        for i in range(min(len(nodes)-1, 10)):  # Add up to 10 backup edges
            G.add_edge(nodes[i], nodes[i+1], type="related_to")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a better layout for visualization
    if len(G.nodes()) < 10:
        pos = nx.spring_layout(G, k=0.5, seed=42)  # Adjust k for more spacing
    else:
        pos = nx.fruchterman_reingold_layout(G, seed=42)  # Alternative layout for larger graphs
    
    # Draw nodes with different colors and larger size
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', 
                          node_size=700, alpha=0.8)
    
    # Draw edges with better visibility
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                          width=1.5, arrows=True, arrowsize=15, alpha=0.7,
                          connectionstyle='arc3,rad=0.1')  # Curved edges for better visibility
    
    # Draw node labels with better font size and weight
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
    
    # Draw edge labels (relationship types) with better positioning
    edge_labels = {(u, v): d['type'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=8, font_color='darkred',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
    
    plt.title("Knowledge Graph Representation", fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def plot_evidence_vs_correctness(evaluation_data):
    """Create an improved scatter plot showing the tradeoff between evidence quality and correctness"""
    approaches = ["LLM Only", "Context Strict", "LLM Informed"]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    evidence_scores = []
    correctness_scores = []
    combined_scores = []
    
    for approach in approaches:
        approach_key = approach.lower().replace(" ", "_")
        evidence_scores.append(
            evaluation_data["evidence_based_evaluation"][approach_key]["evidence_score"]
        )
        correctness_scores.append(
            evaluation_data["evidence_based_evaluation"][approach_key]["correctness_score"]
        )
        combined_scores.append(
            evaluation_data["evidence_based_evaluation"][approach_key]["combined_score"]
        )
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Add grid lines
    ax.grid(which='major', color='#dddddd', linewidth=0.8, linestyle='-', alpha=0.7)
    ax.grid(which='minor', color='#eeeeee', linewidth=0.5, linestyle='-', alpha=0.5)
    
    # Create the quadrants with proper labels
    ax.axhline(y=7, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=7, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add quadrant labels
    ax.text(3.5, 8.5, "Accurate but\nPoorly Evidenced", 
            ha='center', va='center', fontsize=10, color='#808080',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    
    ax.text(8.5, 8.5, "Ideal:\nWell-Evidenced\nand Accurate", 
            ha='center', va='center', fontsize=10, color='green',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', boxstyle='round,pad=0.3'))
    
    ax.text(3.5, 3.5, "Poor:\nInaccurate and\nPoorly Evidenced", 
            ha='center', va='center', fontsize=10, color='#808080',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    
    ax.text(8.5, 3.5, "Well-Evidenced but\nInaccurate", 
            ha='center', va='center', fontsize=10, color='#808080',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Shade the ideal quadrant
    ax.fill_between([7, 10], 7, 10, color='green', alpha=0.1)
    
    # Create bubble size based on combined score (multiplied for visibility)
    bubble_sizes = [score * 25 for score in combined_scores]
    
    # Create scatter plot with improved styling
    for i, approach in enumerate(approaches):
        # Create a more prominent scatter point
        scatter = ax.scatter(
            evidence_scores[i], 
            correctness_scores[i], 
            s=bubble_sizes[i], 
            color=colors[i], 
            label=f"{approach} (Score: {combined_scores[i]:.1f})",
            edgecolor='white',
            linewidth=2,
            alpha=0.8,
            zorder=100  # Ensure points are drawn on top
        )
        
        # Add a faint connecting line to the origin (0,0) to show distance
        ax.plot([0, evidence_scores[i]], [0, correctness_scores[i]], 
                color=colors[i], alpha=0.2, linestyle='-', linewidth=1)
        
        # Add data labels with background
        ax.annotate(
            approach, 
            (evidence_scores[i], correctness_scores[i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=colors[i], alpha=0.8)
        )
    
    # Add titles, labels with improved styling
    ax.set_title('Evidence Quality vs. Correctness Trade-off', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Evidence Quality Score (0-10)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Correctness Score (0-10)', fontsize=14, fontweight='bold')
    
    # Set axis limits with some padding
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    
    # Add legend with better positioning and styling
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              frameon=True, fancybox=True, shadow=True, ncol=3, fontsize=12)
    
    # Add explanation text
    plt.figtext(0.5, -0.05, 
                "Bubble size represents combined score (60% evidence + 40% correctness)",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def create_evidence_metrics_chart(evaluation_data):
    """Create a radar chart of evidence quality metrics for each approach"""
    approaches = ["LLM Only", "Context Strict", "LLM Informed"]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Evidence quality metrics
    metrics = ["Citation Density", "Source Diversity", "Conflict Resolution", "Traceability"]
    
    # Extract values for each approach
    values = []
    for approach in approaches:
        approach_key = approach.lower().replace(" ", "_")
        approach_metrics = evaluation_data["evidence_based_evaluation"][approach_key]["evidence_quality_details"]
        
        values.append([
            approach_metrics["citation_density"] * 10,  # Scale to 0-10
            approach_metrics["source_diversity"] * 10,
            approach_metrics["conflict_resolution"] * 10,
            approach_metrics["traceability"] * 10
        ])
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one line per approach and fill area
    for i, approach in enumerate(approaches):
        values_for_approach = values[i]
        values_for_approach += values_for_approach[:1]  # Close the loop
        
        ax.plot(angles, values_for_approach, linewidth=2, linestyle='solid', label=approach, color=colors[i])
        ax.fill(angles, values_for_approach, color=colors[i], alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    plt.xticks(angles[:-1], metrics, fontsize=12)
    
    # Set y-axis limit
    ax.set_ylim(0, 10)
    
    # Draw y-axis labels (10, 8, 6, 4, 2)
    ax.set_rticks([2, 4, 6, 8, 10])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Evidence Quality Metrics by Approach', fontsize=15, fontweight='bold', pad=20)
    
    return fig

def create_correctness_evidence_breakdown(evaluation_data):
    """Create a stacked bar chart showing the contribution of evidence and correctness to final scores"""
    approaches = ["LLM Only", "Context Strict", "LLM Informed"]
    
    evidence_contribution = []
    correctness_contribution = []
    
    for approach in approaches:
        approach_key = approach.lower().replace(" ", "_")
        evidence_contribution.append(
            evaluation_data["evidence_based_evaluation"][approach_key]["evidence_score"] * 0.6
        )
        correctness_contribution.append(
            evaluation_data["evidence_based_evaluation"][approach_key]["correctness_score"] * 0.4
        )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the stacked bar chart
    ax.bar(approaches, evidence_contribution, label='Evidence (60%)', 
           color='#3498db', edgecolor='white', linewidth=1)
    ax.bar(approaches, correctness_contribution, bottom=evidence_contribution, 
           label='Correctness (40%)', color='#e74c3c', edgecolor='white', linewidth=1)
    
    # Add the total values on top of each bar
    for i, approach in enumerate(approaches):
        total = evidence_contribution[i] + correctness_contribution[i]
        ax.text(i, total + 0.1, f'{total:.1f}', ha='center', fontweight='bold')
    
    # Add value labels within each section of the stacked bars
    for i, v in enumerate(evidence_contribution):
        ax.text(i, v/2, f'{v:.1f}', ha='center', color='white', fontweight='bold')
    
    for i, v in enumerate(correctness_contribution):
        ax.text(i, evidence_contribution[i] + v/2, f'{v:.1f}', ha='center', color='white', fontweight='bold')
    
    # Add titles and labels
    ax.set_title('Score Composition: Evidence vs. Correctness Contribution', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score Points (0-10)', fontsize=12, fontweight='bold')
    
    # Set axis limits
    ax.set_ylim(0, 10)
    
    # Add a horizontal line at 7 (good performance threshold)
    ax.axhline(y=7, color='green', linestyle='--', alpha=0.7, label='Good Performance Threshold')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig

def main():
    # Initialize processor
    processor = initialize_processor()
    
    # Header
    st.markdown("<h1 class='main-header'>Medical Knowledge Graph Q&A System</h1>", unsafe_allow_html=True)
    st.markdown("Powered by knowledge graphs, large language models, and medical textbooks")
    
    # Sidebar with system information
    with st.sidebar:
        st.header("System Information")
        
        # Add theme selector
        theme = st.radio("Theme", ["Dark (Default)", "Light"])
        if theme == "Light":
            # Apply light theme CSS
            st.markdown("""
            <style>
                body {
                    color: #212529 !important;
                    background-color: #ffffff !important;
                }
                p, li, h1, h2, h3, h4, h5, h6, span, div {
                    color: #212529 !important;
                }
                .stMarkdown div {
                    color: #212529 !important;
                }
                /* Add more light theme overrides as needed */
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("This system integrates:")
        st.markdown("- 🧠 Medical Knowledge Graph (Neo4j)")
        st.markdown("- 📚 Medical Textbook Corpus (Pinecone)")
        st.markdown("- 🤖 Large Language Models")
        
        st.markdown("---")
        st.subheader("Current Configuration")
        st.markdown(f"- LLM Model: {processor.llm.model_name}")
        st.markdown(f"- Top-K Concepts: {processor.kg_evaluator.settings['top_k_concepts']}")
        st.markdown(f"- Top-K Relationships: {processor.kg_evaluator.settings['top_k_relationships']}")
        st.markdown(f"- Multi-hop Enabled: {processor.kg_evaluator.settings['multihop_enabled']}")
        st.markdown(f"- Multi-hop Max Depth: {processor.kg_evaluator.settings['multihop_max_depth']}")
        
        # Advanced settings collapsible section
        with st.expander("Advanced Settings"):
            new_top_k_concepts = st.slider("Top-K Concepts", 10, 100, processor.kg_evaluator.settings['top_k_concepts'])
            new_top_k_relationships = st.slider("Top-K Relationships", 10, 200, processor.kg_evaluator.settings['top_k_relationships'])
            new_multihop_max_depth = st.slider("Multi-hop Max Depth", 1, 5, processor.kg_evaluator.settings['multihop_max_depth'])
            new_vector_threshold = st.slider("Vector Search Threshold", 0.1, 0.9, processor.kg_evaluator.settings['vector_search_threshold'])
            
            if st.button("Apply Settings"):
                processor.kg_evaluator.settings['top_k_concepts'] = new_top_k_concepts
                processor.kg_evaluator.settings['top_k_relationships'] = new_top_k_relationships
                processor.kg_evaluator.settings['multihop_max_depth'] = new_multihop_max_depth
                processor.kg_evaluator.settings['vector_search_threshold'] = new_vector_threshold
                st.success("Settings updated successfully!")
    
    # Main area with question input
    st.markdown("<h2 class='sub-header'>Ask a Medical Question</h2>", unsafe_allow_html=True)
    st.markdown("Enter a USMLE-style medical question below:")
    
    # Sample questions dropdown
    sample_questions = [
        "Select a sample question...",
        "A 65-year-old man presents with crushing chest pain radiating to his left arm. ECG shows ST elevation in leads V1-V4. What is the most likely diagnosis?",
        "A 30-year-old woman has been on isotretinoin for severe acne for 3 months. What important precaution should be discussed with this patient?",
        "A 42-year-old man with a history of alcohol abuse presents with epigastric abdominal pain, nausea, and vomiting. Labs show elevated lipase and amylase. What is the diagnosis?"
    ]
    
    selected_sample = st.selectbox("Or choose a sample question:", sample_questions)
    
    # Text area for question input
    if selected_sample != "Select a sample question...":
        question = st.text_area("Your question:", value=selected_sample, height=100)
    else:
        question = st.text_area("Your question:", height=100)
    
    # Options for multiple choice (optional)
    with st.expander("Add multiple choice options (optional)"):
        st.markdown("Enter one option per line:")
        options_text = st.text_area("Options:", height=100)
        options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
    
    # Process button
    if st.button("Process Question"):
        if not question:
            st.error("Please enter a question to proceed.")
            return
        
        with st.spinner("Processing your question... This may take a minute."):
            try:
                # Process the question
                start_time = time.time()
                if options:
                    result = processor.process_question(question, options)
                else:
                    result = processor.process_question(question)
                
                # Evaluate the answer
                evaluation = processor.evaluate_evidence_based_answer(result)
                processing_time = time.time() - start_time
                
                # Show results in tabs
                st.markdown(f"<h2 class='sub-header'>Results (Processed in {processing_time:.2f} seconds)</h2>", unsafe_allow_html=True)
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Answers Comparison", 
                    "Knowledge Graph Data", 
                    "Textbook Evidence", 
                    "Evaluation Metrics with Evidence Priority",
                    "Visualization"
                ])
                
                # Tab 1: Answers Comparison
                with tab1:
                    st.markdown("<h3>Question</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card'>{question}</div>", unsafe_allow_html=True)
                    
                    if options:
                        st.markdown("<h3>Options</h3>", unsafe_allow_html=True)
                        for i, option in enumerate(options):
                            st.markdown(f"{chr(65+i)}. {option}")
                    
                    st.markdown("<h3>LLM Informed Answer (Combined Approach)</h3>", unsafe_allow_html=True)
                    # Create a text area with white background and dark text for better visibility
                    st.text_area("", value=result['answers']['llm_informed'], height=400, key="llm_informed")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h3>Context Strict Answer</h3>", unsafe_allow_html=True)
                        st.text_area("", value=result['answers']['context_strict'], height=300, key="context_strict")
                    
                    with col2:
                        st.markdown("<h3>LLM Only Answer</h3>", unsafe_allow_html=True)
                        st.text_area("", value=result['answers']['llm_only'], height=300, key="llm_only")
                
                # Tab 2: Knowledge Graph Data
                with tab2:
                    st.markdown("<h3>Extracted Medical Terms</h3>", unsafe_allow_html=True)
                    terms_df = pd.DataFrame(result['kg_results']['terms'])
                    st.dataframe(terms_df)
                    
                    st.markdown("<h3>Medical Concepts Found</h3>", unsafe_allow_html=True)
                    
                    # Display concepts as interactive tags
                    concept_html = "<div style='margin-bottom: 20px;'>"
                    for i, concept in enumerate(result['kg_results']['concepts'][:20]):
                        concept_id = f"C{i+1}"
                        concept_term = concept.get('term', 'Unknown')
                        concept_def = concept.get('definition', 'No definition available')
                        concept_html += f"<div class='concept-tag' title='{concept_def}'>{concept_id}: {concept_term}</div>"
                    concept_html += "</div>"
                    st.markdown(concept_html, unsafe_allow_html=True)
                    
                    # Show concepts in a table
                    concepts_data = []
                    for i, concept in enumerate(result['kg_results']['concepts']):
                        concepts_data.append({
                            'ID': f"C{i+1}",
                            'Term': concept.get('term', 'Unknown'),
                            'Definition': concept.get('definition', 'No definition available'),
                            'CUI': concept.get('cui', 'Unknown'),
                            'Vector Match': concept.get('vector_match', False)
                        })
                    
                    concepts_df = pd.DataFrame(concepts_data)
                    st.dataframe(concepts_df)
                    
                    st.markdown("<h3>Relationships Between Concepts</h3>", unsafe_allow_html=True)
                    
                    # Display relationships as tags
                    relationship_html = "<div style='margin-bottom: 20px;'>"
                    for i, rel in enumerate(result['kg_results']['relationships'][:15]):
                        rel_id = f"R{i+1}"
                        rel_type = rel.get('relationship_type', 'Unknown')
                        source = rel.get('source_name', 'Unknown')
                        target = rel.get('target_name', 'Unknown')
                        relationship_html += f"<div class='relationship-tag'>{rel_id}: {source} → {rel_type} → {target}</div>"
                    relationship_html += "</div>"
                    st.markdown(relationship_html, unsafe_allow_html=True)
                    
                    # Show relationships in a table
                    relationships_data = []
                    for i, rel in enumerate(result['kg_results']['relationships']):
                        relationships_data.append({
                            'ID': f"R{i+1}",
                            'Type': rel.get('relationship_type', 'Unknown'),
                            'Source': rel.get('source_name', 'Unknown'),
                            'Target': rel.get('target_name', 'Unknown')
                        })
                    
                    relationships_df = pd.DataFrame(relationships_data)
                    st.dataframe(relationships_df)
                    
                    if result['kg_results']['multihop_paths']:
                        st.markdown("<h3>Multi-hop Paths</h3>", unsafe_allow_html=True)
                        
                        multihop_data = []
                        for i, path in enumerate(result['kg_results']['multihop_paths']):
                            multihop_data.append({
                                'ID': f"P{i+1}",
                                'Source': path.get('source_term', 'Unknown'),
                                'Target': path.get('target_term', 'Unknown'),
                                'Length': path.get('path_length', 0),
                                'Path': path.get('path_description', 'Unknown')
                            })
                        
                        multihop_df = pd.DataFrame(multihop_data)
                        st.dataframe(multihop_df)
                
                # Tab 3: Textbook Evidence
                with tab3:
                    st.markdown("<h3>Relevant Textbook Passages</h3>", unsafe_allow_html=True)
                    
                    for i, chunk in enumerate(result['textbook_results'], 1):
                        st.markdown(f"<h4>Source {i}: {chunk.get('sources', 'Unknown Source')}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Relevance Score:</strong> {chunk.get('score', 0):.2f}</p>", unsafe_allow_html=True)
                        
                        # Use text_area with custom height for better visibility
                        st.text_area("", value=chunk.get('text', ''), height=200, key=f"chunk_{i}")
                
                # Tab 4: Evaluation Metrics with Evidence Priority
                with tab4:
                    st.markdown("<h3>Evidence-Prioritized Evaluation</h3>", unsafe_allow_html=True)
                    
                    try:
                        # Use the new evaluation approach if it hasn't been applied yet
                        if "evidence_based_evaluation" not in evaluation:
                            with st.spinner("Running evidence-prioritized evaluation..."):
                                evaluation = processor.evaluate_evidence_based_answer(result)
                        
                        # Create evidence vs correctness plot
                        st.markdown("<h4>Evidence Quality vs. Correctness</h4>", unsafe_allow_html=True)
                        st.markdown("This chart shows the balance between evidence usage and correctness for each approach.")
                        evidence_vs_correctness_fig = plot_evidence_vs_correctness(evaluation)
                        st.pyplot(evidence_vs_correctness_fig)
                        
                        # Create radar chart of evidence metrics
                        st.markdown("<h4>Evidence Quality Metrics</h4>", unsafe_allow_html=True)
                        evidence_metrics_fig = create_evidence_metrics_chart(evaluation)
                        st.pyplot(evidence_metrics_fig)
                        
                        # Create correctness vs evidence breakdown
                        st.markdown("<h4>Score Composition Analysis</h4>", unsafe_allow_html=True)
                        st.markdown("This chart shows how evidence quality and correctness contribute to the final score for each approach.")
                        composition_fig = create_correctness_evidence_breakdown(evaluation)
                        st.pyplot(composition_fig)
                        
                        # Display detailed scores for each approach
                        st.markdown("<h4>Evidence-Based Scores (60% Evidence, 40% Correctness)</h4>", unsafe_allow_html=True)
                        
                        evidence_scores = {
                            'Approach': ['LLM Only', 'Context Strict', 'LLM Informed'],
                            'Evidence Score': [
                                evaluation["evidence_based_evaluation"]["llm_only"]["evidence_score"],
                                evaluation["evidence_based_evaluation"]["context_strict"]["evidence_score"],
                                evaluation["evidence_based_evaluation"]["llm_informed"]["evidence_score"]
                            ],
                            'Correctness Score': [
                                evaluation["evidence_based_evaluation"]["llm_only"]["correctness_score"],
                                evaluation["evidence_based_evaluation"]["context_strict"]["correctness_score"],
                                evaluation["evidence_based_evaluation"]["llm_informed"]["correctness_score"]
                            ],
                            'Calculation': [
                                f"({evaluation['evidence_based_evaluation']['llm_only']['evidence_score']:.2f} × 0.6) + ({evaluation['evidence_based_evaluation']['llm_only']['correctness_score']:.2f} × 0.4)",
                                f"({evaluation['evidence_based_evaluation']['context_strict']['evidence_score']:.2f} × 0.6) + ({evaluation['evidence_based_evaluation']['context_strict']['correctness_score']:.2f} × 0.4)",
                                f"({evaluation['evidence_based_evaluation']['llm_informed']['evidence_score']:.2f} × 0.6) + ({evaluation['evidence_based_evaluation']['llm_informed']['correctness_score']:.2f} × 0.4)"
                            ],
                            'Combined Score': [
                                evaluation["evidence_based_evaluation"]["llm_only"]["combined_score"],
                                evaluation["evidence_based_evaluation"]["context_strict"]["combined_score"],
                                evaluation["evidence_based_evaluation"]["llm_informed"]["combined_score"]
                            ]
                        }
                        
                        # Add explanation before showing the dataframe
                        st.markdown("""
                        <div style='background-color: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                        <p><strong>How Combined Scores are Calculated:</strong></p>
                        <p>Combined Score = (Evidence Score × 0.6) + (Correctness Score × 0.4)</p>
                        <p>This weighting prioritizes evidence quality (60%) over correctness (40%).</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        evidence_scores_df = pd.DataFrame(evidence_scores)
                        st.dataframe(evidence_scores_df, use_container_width=True)
                        
                        # Show original evaluation metrics for comparison if available
                        if 'answer_quality' in evaluation:
                            st.markdown("<h4>Original Quality Metrics (For Comparison)</h4>", unsafe_allow_html=True)
                            
                            quality_data = {
                                'Approach': ['LLM Only', 'Context Strict', 'LLM Informed'],
                                'Quality Score': [
                                    float(evaluation['answer_quality']['llm_only'].get('overall_quality', 0)),
                                    float(evaluation['answer_quality']['context_strict'].get('overall_quality', 0)),
                                    float(evaluation['answer_quality']['llm_informed'].get('overall_quality', 0))
                                ]
                            }
                            
                            quality_df = pd.DataFrame(quality_data)
                            st.dataframe(quality_df, use_container_width=True)
                        
                        # Evidence details for best approach
                        st.markdown("<h4>Detailed Evidence Analysis</h4>", unsafe_allow_html=True)
                        
                        best_approach = max(evidence_scores['Approach'], 
                                            key=lambda x: evidence_scores['Combined Score'][evidence_scores['Approach'].index(x)])
                        best_approach_key = best_approach.lower().replace(" ", "_")
                        
                        raw_counts = evaluation["evidence_based_evaluation"][best_approach_key]["evidence_quality_details"]["raw_counts"]
                        
                        # Create citation summary
                        st.markdown(f"**Citation Analysis for {best_approach}:**")
                        
                        citation_cols = st.columns(5)
                        with citation_cols[0]:
                            st.metric("Concept Citations", raw_counts["concept_citations"])
                        with citation_cols[1]:
                            st.metric("Relationship Citations", raw_counts["relationship_citations"])
                        with citation_cols[2]:
                            st.metric("Path Citations", raw_counts["path_citations"])
                        with citation_cols[3]:
                            st.metric("Textbook Citations", raw_counts["textbook_citations"])
                        with citation_cols[4]:
                            st.metric("Total Citations", raw_counts["total_citations"])
                        
                        # Citations per 100 words
                        citation_density = raw_counts["total_citations"] / (raw_counts["answer_length"]/100)
                        
                        st.markdown(f"**Citation Density:** {citation_density:.2f} citations per 100 words")
                        
                        # Show a chart of citation distribution
                        citation_types = ['Concept', 'Relationship', 'Path', 'Textbook']
                        citation_counts = [
                            raw_counts["concept_citations"],
                            raw_counts["relationship_citations"],
                            raw_counts["path_citations"],
                            raw_counts["textbook_citations"]
                        ]
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bars = ax.bar(citation_types, citation_counts, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
                        
                        # Add labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height}', ha='center', va='bottom')
                        
                        ax.set_title('Citation Distribution')
                        ax.set_ylabel('Count')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Improvement recommendations
                        st.markdown("<h4>Evidence Improvement Recommendations</h4>", unsafe_allow_html=True)
                        
                        # Based on the evidence scores, make recommendations
                        recommendations = []
                        
                        best_evidence_score = evaluation["evidence_based_evaluation"][best_approach_key]["evidence_score"]
                        best_correctness_score = evaluation["evidence_based_evaluation"][best_approach_key]["correctness_score"]
                        
                        if best_evidence_score < 7:
                            if raw_counts["concept_citations"] < 5:
                                recommendations.append("❌ **Increase concept citations**: More explicitly reference medical concepts from the knowledge graph")
                            
                            if raw_counts["textbook_citations"] < 2:
                                recommendations.append("❌ **Add more textbook evidence**: Include more direct quotes from textbook sources")
                            
                            if evaluation["evidence_based_evaluation"][best_approach_key]["evidence_quality_details"]["conflict_resolution"] < 0.5:
                                recommendations.append("❌ **Improve conflict handling**: Explicitly address contradictions between evidence sources")
                        else:
                            recommendations.append("✅ **Good evidence usage**: The answer effectively leverages available evidence")
                        
                        if best_correctness_score < 7:
                            recommendations.append("❌ **Verify factual accuracy**: Ensure all medical facts align with current knowledge")
                            
                            if evaluation["evidence_based_evaluation"][best_approach_key]["correctness_details"]["error_detection"] < 0.5:
                                recommendations.append("❌ **Add error detection**: Acknowledge limitations in the evidence when appropriate")
                        else:
                            recommendations.append("✅ **Good correctness**: The answer is medically accurate")
                        
                        for recommendation in recommendations:
                            st.markdown(recommendation)
                    
                    except Exception as e:
                        st.error(f"Error in evaluation: {str(e)}")
                        st.info("Some evaluation metrics couldn't be displayed. This may happen if the evaluation format has changed or if required data is missing.")
                        
                        # Display debug information
                        if st.checkbox("Show Error Details"):
                            st.code(f"Error type: {type(e).__name__}\nError message: {str(e)}")
                            st.text("Evaluation keys available:")
                            st.json({k: type(v).__name__ for k, v in evaluation.items()})
                
                # Tab 5: Knowledge Graph Visualization
                with tab5:
                    st.markdown("<h3>Knowledge Graph Visualization</h3>", unsafe_allow_html=True)
                    
                    # Create and display the knowledge graph visualization
                    kg_fig = render_knowledge_graph(
                        result['kg_results']['concepts'],
                        result['kg_results']['relationships']
                    )
                    st.pyplot(kg_fig)
                    
                    st.markdown("<p class='info-text'>Note: This visualization shows a simplified view of the knowledge graph, limited to the most relevant concepts and relationships.</p>", unsafe_allow_html=True)
                
                # Final summary with evidence prioritization
                st.markdown("<h2 class='sub-header'>Summary</h2>", unsafe_allow_html=True)
                
                # Determine best approach based on evidence-prioritized scores
                if "evidence_based_evaluation" in evaluation:
                    evidence_best_approach = max(['LLM Only', 'Context Strict', 'LLM Informed'], 
                                               key=lambda x: evaluation["evidence_based_evaluation"][x.lower().replace(" ", "_")]["combined_score"])
                elif 'answer_quality' in evaluation:
                    # Original summary if we have answer_quality but not evidence_based_evaluation
                    evidence_best_approach = max(['LLM Only', 'Context Strict', 'LLM Informed'], 
                                                key=lambda x: float(evaluation['answer_quality'][x.lower().replace(' ', '_')].get('overall_quality', 0)))
                
                summary_html = f"<div class='evaluation-card'>" \
                               f"<p><strong>Best Approach (Evidence Priority):</strong> {evidence_best_approach} (Score: {evaluation['evidence_based_evaluation'][evidence_best_approach.lower().replace(' ', '_')]['combined_score']:.1f}/10)</p>"
                
                # Only include original best approach if answer_quality exists
                if 'answer_quality' in evaluation:
                    original_best_approach = max(['LLM Only', 'Context Strict', 'LLM Informed'], 
                                            key=lambda x: float(evaluation['answer_quality'][x.lower().replace(' ', '_')].get('overall_quality', 0)))
                    summary_html += f"<p><strong>Best Approach (Original):</strong> {original_best_approach}</p>"
                
                summary_html += f"<p><strong>Knowledge Graph Coverage:</strong> {float(evaluation['kg_coverage']['coverage_percentage']):.1f}%</p>" \
                               f"<p><strong>Total Processing Time:</strong> {processing_time:.2f} seconds</p>" \
                               f"</div>"
                
                st.markdown(summary_html, unsafe_allow_html=True)
                
                # Add this at the bottom of the page after processing a question
                if st.checkbox("Show Debug Information"):
                    st.subheader("Debug Information")
                    
                    st.write("Number of concepts found:", len(result['kg_results']['concepts']))
                    st.write("Number of relationships found:", len(result['kg_results']['relationships']))
                    st.write("Number of textbook passages found:", len(result['textbook_results']))
                    
                    with st.expander("Raw Relationships Data (First 5)"):
                        if result['kg_results']['relationships']:
                            st.json(result['kg_results']['relationships'][:5])
                        else:
                            st.write("No relationships found.")
                    
                    with st.expander("Raw Concepts Data (First 5)"):
                        if result['kg_results']['concepts']:
                            st.json(result['kg_results']['concepts'][:5])
                        else:
                            st.write("No concepts found.")
                    
                    with st.expander("First Answer (Debug View)"):
                        st.text(result['answers']['llm_informed'])
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()
