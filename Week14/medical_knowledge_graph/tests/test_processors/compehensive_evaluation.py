def _visualize_domain_performance(self, summary, viz_dir):
    """Visualize performance across medical domains"""
    # Define colors for different configurations
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    
    # Get domains and configurations
    domains = summary.get("domains", [])
    configs = ["llm_only", "kg_only", "rag_only", "combined_context_strict", "combined_llm_informed"]
    display_names = {
        "llm_only": "LLM Only",
        "kg_only": "KG Only",
        "rag_only": "RAG Only",
        "combined_context_strict": "Context Strict",
        "combined_llm_informed": "LLM Informed"
    }
    
    # Rest of the existing code... 