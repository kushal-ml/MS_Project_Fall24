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
        color: #3498db !important;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2ecc71 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Card styling */
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #3498db;
        color: #ffffff !important;
    }
    
    .evaluation-card {
        background-color: rgba(46, 204, 113, 0.1) !important;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2ecc71;
        color: #ffffff !important;
    }
    
    /* Tags styling */
    .concept-tag {
        background-color: rgba(52, 152, 219, 0.3) !important;
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
        background-color: rgba(243, 156, 18, 0.3) !important;
        color: #ffffff !important;
        padding: 0.3rem 0.6rem;
        border-radius: 16px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-size: 0.85rem;
        border: 1px solid rgba(243, 156, 18, 0.7);
    }
    
    /* Text containers */
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
    
    /* General text colors */
    p, li, h1, h2, h3, h4, h5, h6, span, div {
        color: #ffffff !important;
    }
    
    /* Info text */
    .info-text {
        color: #bdc3c7 !important;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_processor():
    """Initialize the USMLEProcessor (cached to avoid reinitialization)"""
    load_dotenv()
    return USMLEProcessor()

def render_knowledge_graph(concepts, relationships):
    """Render a visual representation of the knowledge graph"""
    G = nx.DiGraph()
    
    # Add nodes (concepts)
    for concept in concepts[:20]:  # Limit to top 20 concepts
        if isinstance(concept, dict):
            term = concept.get('term', concept.get('name', 'Unknown'))
            relevance = concept.get('relevance_score', 0.0)
            G.add_node(term, type='concept', relevance=relevance)
    
    # Add edges (relationships)
    added_edges = set()
    for rel in relationships[:40]:  # Limit to top 40 relationships
        if isinstance(rel, dict):
            source = rel.get('source_name', rel.get('source', ''))
            target = rel.get('target_name', rel.get('target', ''))
            rel_type = rel.get('relationship_type', rel.get('type', ''))
            
            edge_key = (source, target, rel_type)
            if source and target and edge_key not in added_edges:
                G.add_edge(source, target, type=rel_type)
                added_edges.add(edge_key)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42) if len(G.nodes()) < 10 else nx.kamada_kawai_layout(G)
    
    node_relevance = [G.nodes[node].get('relevance', 0.5) * 2000 + 500 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=node_relevance, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.5, arrows=True, arrowsize=15, alpha=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold', font_color='black')
    
    edge_labels = {(u, v): d['type'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkred',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title("Knowledge Graph Representation", fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def get_evaluation_scores(evaluation_data, approach):
    """Safely get evaluation scores for a given approach"""
    approach_key = approach.lower().replace(" ", "_")
    
    # Try different possible structures
    if approach_key in evaluation_data:
        return evaluation_data[approach_key]
    elif "evidence_based_evaluation" in evaluation_data and approach_key in evaluation_data["evidence_based_evaluation"]:
        return evaluation_data["evidence_based_evaluation"][approach_key]
    else:
        # Return default scores if structure not found
        return {
            "evidence_score": 0,
            "correctness_score": 0,
            "combined_score": 0,
            "evidence_quality_details": {
                "citation_density": 0,
                "source_diversity": 0,
                "conflict_resolution": 0,
                "traceability": 0
            }
        }

def plot_evidence_vs_correctness(evaluation_data):
    """Create scatter plot showing evidence quality vs correctness"""
    approaches = ["LLM Only", "Context Strict", "LLM Informed"]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    evidence_scores = []
    correctness_scores = []
    combined_scores = []
    
    for approach in approaches:
        scores = get_evaluation_scores(evaluation_data, approach)
        evidence_scores.append(scores["evidence_score"])
        correctness_scores.append(scores["correctness_score"])
        combined_scores.append(scores["combined_score"])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f9fa')
    ax.grid(which='major', color='#dddddd', linewidth=0.8, alpha=0.7)
    
    ax.axhline(y=7, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=7, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
    
    bubble_sizes = [score * 100 for score in combined_scores]
    
    for i, approach in enumerate(approaches):
        ax.scatter(
            evidence_scores[i], 
            correctness_scores[i], 
            s=bubble_sizes[i], 
            color=colors[i], 
            label=f"{approach}\n(Score: {combined_scores[i]:.1f})",
            edgecolor='white',
            linewidth=2,
            alpha=0.8
        )
        
        ax.annotate(
            approach, 
            (evidence_scores[i], correctness_scores[i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Evidence Quality Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Correctness Score', fontsize=14, fontweight='bold')
    ax.set_title('Evidence Quality vs. Correctness Trade-off', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def _format_textbook_evidence(self, chunks):
    """
    Format textbook evidence for display - mirrors combined_usmle_2.py's _format_textbook_evidence method
    to ensure compatibility
    """
    evidence = []
    
    for i, chunk in enumerate(chunks, 1):
        if isinstance(chunk, dict):
            source = chunk.get('sources', chunk.get('source', 'Unknown'))
            score = chunk.get('score', chunk.get('relevance', 0))
            text = chunk.get('text', chunk.get('content', chunk.get('page_content', str(chunk))))
        else:
            source = "Unknown"
            score = 0
            text = str(chunk)
        
        evidence.append({
            'index': i,
            'source': source,
            'score': score,
            'text': text
        })
    
    return evidence

def retrieve_textbook_passages(processor, question_text, result):
    """
    Retrieve textbook passages using the processor's methods,
    falling back to different approaches if needed
    """
    # First, try to use existing textbook results if available
    if result.get('textbook_results') and isinstance(result['textbook_results'], list) and len(result['textbook_results']) > 0:
        # Check if the results have actual content
        has_content = False
        for chunk in result['textbook_results']:
            if isinstance(chunk, dict) and any(key in chunk for key in ['text', 'content', 'page_content']):
                if any(len(str(chunk.get(key, ''))) > 10 for key in ['text', 'content', 'page_content']):
                    has_content = True
                    break
            elif isinstance(chunk, str) and len(chunk) > 10:
                has_content = True
                break
        
        if has_content:
            return result['textbook_results']
    
    # If not, try different methods to retrieve passages
    try:
        # Try the combined_usmle_2 retrieve_from_pinecone method
        if hasattr(processor, 'retrieve_from_pinecone'):
            passages = processor.retrieve_from_pinecone(question_text, top_k=8)
            if passages and len(passages) > 0:
                return passages
        
        # Fallback to _combine_similar_chunks if available
        if hasattr(processor, '_combine_similar_chunks') and hasattr(processor, 'retrieve_documents'):
            chunks = processor.retrieve_documents(question_text)
            if chunks and len(chunks) > 0:
                return processor._combine_similar_chunks(chunks)
    except Exception as e:
        st.warning(f"Error retrieving textbook passages: {str(e)}")
    
    # Create a placeholder if nothing worked
    return [{"text": "No relevant textbook passages found for this question.", "source": "System", "score": 0}]

def display_textbook_tab(tab, processor, question, result):
    """
    Display textbook evidence directly from Pinecone with minimal processing
    """
    with tab:
        st.markdown("<h3>Relevant Textbook Passages</h3>", unsafe_allow_html=True)
        
        try:
            # First try to use existing results if they're valid
            existing_results = result.get('textbook_results', []) if isinstance(result.get('textbook_results'), list) else []
            
            # Then try to get fresh results from Pinecone
            fresh_results = []
            try:
                with st.spinner("Querying Pinecone database..."):
                    fresh_results = processor.retrieve_from_pinecone(question, top_k=10)
            except Exception as e:
                st.warning(f"Could not retrieve fresh results from Pinecone: {str(e)}")
            
            # Combine and deduplicate results
            all_results = []
            seen_texts = set()
            
            # Process both fresh and existing results
            for results in [fresh_results, existing_results]:
                if not results:
                    continue
                    
                for chunk in results:
                    if not isinstance(chunk, dict):
                        continue
                        
                    # Try to extract text content
                    text_content = None
                    for field in ['text', 'content', 'page_content', 'passage']:
                        if chunk.get(field):
                            text_content = chunk[field]
                            break
                    
                    if not text_content:
                        continue
                        
                    # Skip if we've seen this text before
                    text_hash = hash(text_content)
                    if text_hash in seen_texts:
                        continue
                    seen_texts.add(text_hash)
                    
                    # Extract metadata
                    source = None
                    for field in ['sources', 'source', 'metadata.source']:
                        if field in chunk:
                            source = chunk[field]
                            break
                        elif '.' in field and field.split('.')[0] in chunk:
                            nested = chunk[field.split('.')[0]]
                            if isinstance(nested, dict) and field.split('.')[1] in nested:
                                source = nested[field.split('.')[1]]
                                break
                    
                    # Extract score
                    score = None
                    for field in ['score', 'relevance', 'relevance_score']:
                        if field in chunk:
                            score = float(chunk[field])
                            break
                    
                    all_results.append({
                        'text': text_content,
                        'source': source or 'Unknown Source',
                        'score': score or 0.0
                    })
            
            # Sort results by score
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            if not all_results:
                st.warning("No textbook references found.")
                return
            
            # Display results count
            st.success(f"Found {len(all_results)} relevant textbook passages")
            
            # Display each passage
            for i, chunk in enumerate(all_results, 1):
                with st.expander(f"📚 Passage {i} - Source: {chunk['source']} (Score: {chunk['score']:.2f})", expanded=i==1):
                    # Add debug toggle
                    if st.checkbox(f"Show raw data for passage {i}", key=f"raw_data_{i}"):
                        st.json(chunk)
                    
                    # Display the text content
                    st.markdown("**Content:**")
                    st.text_area(
                        label="",
                        value=chunk['text'],
                        height=200,
                        key=f"chunk_{i}"
                    )
                st.markdown("---")
        
        except Exception as e:
            st.error("Error displaying textbook evidence")
            st.error(str(e))
            
            # Show detailed error in expandable section
            with st.expander("Show detailed error information"):
                import traceback
                st.code(traceback.format_exc())

def process_question(question_text, processor):
    """Process a medical question through the system"""
    try:
        st.info("Processing question... This may take a moment.")
        start_time = time.time()
        
        # Process the question using the processor from combined_usmle_2.py
        result = processor.process_question({"question": question_text})
        
        # Make sure textbook_results exist and are in the correct format
        if not result.get('textbook_results'):
            try:
                # Try to retrieve textbook passages directly
                result['textbook_results'] = processor.retrieve_from_pinecone(question_text, top_k=8)
                
                # If retrieval fails or returns empty, try to use the processor's formatted method
                if not result['textbook_results'] and hasattr(processor, '_format_textbook_evidence'):
                    raw_chunks = processor.retrieve_from_pinecone(question_text, top_k=8)
                    if raw_chunks:
                        result['textbook_results'] = processor._format_textbook_evidence(raw_chunks)
            except Exception as e:
                st.warning(f"Could not retrieve textbook passages: {str(e)}")
                # Create a placeholder
                result['textbook_results'] = [{"text": "No textbook passages could be retrieved.", "source": "System"}]
        
        # Evaluate the answer with evidence priority
        evaluation = processor.evaluate_evidence_based_answer(result)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Compile the final result
        compiled_result = {
            "question": question_text,
            "kg_results": {
                "concepts": result.get('kg_results', {}).get('concepts', []),
                "relationships": result.get('kg_results', {}).get('relationships', []),
                "multihop_paths": result.get('kg_results', {}).get('multihop_paths', []),
                "formatted_data": result.get('kg_results', {}).get('formatted_data', '')
            },
            "textbook_results": result.get('textbook_results', []),
            "answers": {
                "llm_only": result.get('answers', {}).get('llm_only', ''),
                "context_strict": result.get('answers', {}).get('context_strict', ''),
                "llm_informed": result.get('answers', {}).get('llm_informed', '')
            },
            "evaluation": evaluation,
            "processing_time": processing_time
        }
        
        return compiled_result
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        import traceback
        st.text("Detailed error:")
        st.code(traceback.format_exc())
        return None

def main():
    # Initialize processor
    processor = initialize_processor()
    
    # Header
    st.markdown("<h1 class='main-header'>Medical Knowledge Graph Q&A System</h1>", unsafe_allow_html=True)
    st.markdown("Powered by knowledge graphs, large language models, and medical textbooks")
    
    # Sidebar with system information
    with st.sidebar:
        st.header("System Information")
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
    
    # Main question input area
    st.markdown("<h2 class='sub-header'>Ask a Medical Question</h2>", unsafe_allow_html=True)
    
    sample_questions = [
        "Select a sample question...",
        "A 65-year-old man presents with crushing chest pain radiating to his left arm. ECG shows ST elevation in leads V1-V4. What is the most likely diagnosis?",
        "A 30-year-old woman has been on isotretinoin for severe acne for 3 months. What important precaution should be discussed with this patient?",
        "A 42-year-old man with a history of alcohol abuse presents with epigastric abdominal pain, nausea, and vomiting. Labs show elevated lipase and amylase. What is the diagnosis?"
    ]
    
    selected_sample = st.selectbox("Or choose a sample question:", sample_questions)
    
    if selected_sample != "Select a sample question...":
        question = st.text_area("Your question:", value=selected_sample, height=100)
    else:
        question = st.text_area("Your question:", height=100)
    
    # Optional multiple choice options
    with st.expander("Add multiple choice options (optional)"):
        options_text = st.text_area("Options:", height=100)
        options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
    
    # Process button
    if st.button("Process Question"):
        if not question:
            st.error("Please enter a question to proceed.")
            return
        
        with st.spinner("Processing your question... This may take a minute."):
            result = process_question(question, processor)
            
            if result is None:
                return
            
            # Show results in tabs
            st.markdown(f"<h2 class='sub-header'>Results (Processed in {result['processing_time']:.2f} seconds)</h2>", unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Answers Comparison", 
                "Knowledge Graph Data", 
                "Textbook Evidence", 
                "Evaluation Metrics",
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
                        'Relevance': concept.get('relevance_score', 0.0)
                    })
                
                if concepts_data:
                    st.dataframe(pd.DataFrame(concepts_data))
                
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
                        'Target': rel.get('target_name', 'Unknown'),
                        'Relevance': rel.get('relevance_score', 0.0)
                    })
                
                if relationships_data:
                    st.dataframe(pd.DataFrame(relationships_data))
            
            # Tab 3: Textbook Evidence
            display_textbook_tab(tab3, processor, question, result)
            
            # Tab 4: Evaluation Metrics
            with tab4:
                st.markdown("<h3>Evidence-Prioritized Evaluation</h3>", unsafe_allow_html=True)
                
                try:
                    # Add explanation about the evaluation method
                    st.markdown("""
                    <div style='background-color: rgba(46, 204, 113, 0.1); padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                    <p><strong>How Evidence-Based Evaluation Works:</strong></p>
                    <p>The evaluation system examines both evidence quality (60%) and answer correctness (40%).</p>
                    <ol>
                        <li><strong>Evidence Quality:</strong> Measures how well the answer cites and uses knowledge from the graph and textbook sources.</li>
                        <li><strong>Correctness:</strong> Assesses the medical accuracy, logical coherence, and error handling.</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create evidence vs correctness plot
                    st.markdown("<h4>Evidence Quality vs. Correctness</h4>", unsafe_allow_html=True)
                    evidence_vs_correctness_fig = plot_evidence_vs_correctness(result['evaluation'])
                    st.pyplot(evidence_vs_correctness_fig)
                    
                    # Display detailed scores for each approach
                    st.markdown("<h4>Evidence-Based Scores (60% Evidence, 40% Correctness)</h4>", unsafe_allow_html=True)
                    
                    approaches = ["LLM Only", "Context Strict", "LLM Informed"]
                    scores_data = []
                    
                    for approach in approaches:
                        scores = get_evaluation_scores(result['evaluation'], approach)
                        scores_data.append({
                            'Approach': approach,
                            'Evidence Score': scores['evidence_score'],
                            'Correctness Score': scores['correctness_score'],
                            'Calculation': f"({scores['evidence_score']:.2f} × 0.6) + ({scores['correctness_score']:.2f} × 0.4)",
                            'Combined Score': scores['combined_score']
                        })
                    
                    scores_df = pd.DataFrame(scores_data)
                    st.dataframe(scores_df, use_container_width=True)
                    
                    # Highlight highest scores
                    max_evidence_idx = scores_df['Evidence Score'].idxmax()
                    max_correctness_idx = scores_df['Correctness Score'].idxmax()
                    max_combined_idx = scores_df['Combined Score'].idxmax()
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(f"""
                        <div style='background-color: rgba(52, 152, 219, 0.2); padding: 10px; border-radius: 5px; text-align: center;'>
                            <p style='margin: 0;'><strong>Best Evidence:</strong></p>
                            <h3 style='margin: 5px 0; color: #3498db !important;'>{scores_df.iloc[max_evidence_idx]['Approach']}</h3>
                            <p style='margin: 0;'>Score: {scores_df.iloc[max_evidence_idx]['Evidence Score']:.2f}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        st.markdown(f"""
                        <div style='background-color: rgba(231, 76, 60, 0.2); padding: 10px; border-radius: 5px; text-align: center;'>
                            <p style='margin: 0;'><strong>Best Correctness:</strong></p>
                            <h3 style='margin: 5px 0; color: #e74c3c !important;'>{scores_df.iloc[max_correctness_idx]['Approach']}</h3>
                            <p style='margin: 0;'>Score: {scores_df.iloc[max_correctness_idx]['Correctness Score']:.2f}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        st.markdown(f"""
                        <div style='background-color: rgba(46, 204, 113, 0.2); padding: 10px; border-radius: 5px; text-align: center;'>
                            <p style='margin: 0;'><strong>Best Overall:</strong></p>
                            <h3 style='margin: 5px 0; color: #2ecc71 !important;'>{scores_df.iloc[max_combined_idx]['Approach']}</h3>
                            <p style='margin: 0;'>Score: {scores_df.iloc[max_combined_idx]['Combined Score']:.2f}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add score improvement tips
                    st.markdown("""
                    <div style='background-color: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;'>
                    <h4 style='margin-top: 0;'>Score Improvement Tips:</h4>
                    <ul>
                        <li>Explicitly reference concepts from the knowledge graph</li>
                        <li>Cite relationships between medical concepts</li>
                        <li>Quote or paraphrase relevant textbook passages</li>
                        <li>Address contradictions in evidence sources</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error in evaluation: {str(e)}")
                    st.info("Some evaluation metrics couldn't be displayed. This may happen if the evaluation format has changed or if required data is missing.")
                    
                    if st.checkbox("Show Error Details"):
                        st.json({k: type(v).__name__ for k, v in result['evaluation'].items()})
            
            # Tab 5: Knowledge Graph Visualization
            with tab5:
                st.markdown("<h3>Knowledge Graph Visualization</h3>", unsafe_allow_html=True)
                
                kg_fig = render_knowledge_graph(
                    result['kg_results']['concepts'],
                    result['kg_results']['relationships']
                )
                st.pyplot(kg_fig)
                
                st.markdown("<p class='info-text'>Note: This visualization shows a simplified view of the knowledge graph, limited to the most relevant concepts and relationships.</p>", unsafe_allow_html=True)
            
            # Final summary
            st.markdown("<h2 class='sub-header'>Summary</h2>", unsafe_allow_html=True)
            
            # Determine best approach
            try:
                approaches = ["LLM Only", "Context Strict", "LLM Informed"]
                best_approach = max(approaches, key=lambda x: get_evaluation_scores(result['evaluation'], x)["combined_score"])
                best_scores = get_evaluation_scores(result['evaluation'], best_approach)
                combined_score = best_scores["combined_score"]
            except:
                best_approach = "LLM Informed"
                combined_score = 0.0
            
            summary_html = f"""
            <div class='evaluation-card'>
                <p><strong>Best Approach:</strong> {best_approach} (Score: {combined_score:.1f}/10)</p>
                <p><strong>Processing Time:</strong> {result['processing_time']:.2f} seconds</p>
            </div>
            """
            
            st.markdown(summary_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()