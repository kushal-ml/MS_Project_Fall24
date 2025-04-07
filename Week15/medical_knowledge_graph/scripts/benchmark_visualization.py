import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

def generate_benchmark_visualization(benchmark_results_path, output_dir=None):
    """
    Generate comprehensive visualizations from benchmark results
    
    Args:
        benchmark_results_path: Path to the benchmark_results.json file
        output_dir: Directory to save visualizations (defaults to same directory as results)
    """
    # Load the benchmark results
    with open(benchmark_results_path, 'r') as f:
        data = json.load(f)
    
    # Extract summary and results
    summary = data.get('summary', {})
    results = data.get('results', [])
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(benchmark_results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a modern style with better colors
    plt.style.use('ggplot')
    # Create a custom colormap
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # 1. Overall Answer Quality Comparison (Enhanced)
    plt.figure(figsize=(12, 8))
    answer_types = ['LLM Only', 'Context Strict', 'LLM Informed']
    scores = [
        summary['answer_quality']['llm_only'],
        summary['answer_quality']['context_strict'],
        summary['answer_quality']['llm_informed']
    ]
    
    # Main bars
    bars = plt.bar(answer_types, scores, color=colors[:3], width=0.6, alpha=0.85)
    
    # Add horizontal line for comparison
    plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.6, label="Baseline Performance (5.0)")
    
    # Add score labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Percentage improvement annotations
    baseline = summary['answer_quality']['llm_only']
    for i, score in enumerate(scores[1:], 1):
        if score > baseline:
            pct_imp = ((score - baseline) / baseline) * 100
            plt.text(i, scores[i] + 0.4, f'+{pct_imp:.1f}%', ha='center', color='green', fontweight='bold')
    
    plt.ylim(0, 10)
    plt.ylabel('Quality Score (0-10)', fontsize=12)
    plt.title('Answer Quality Comparison Across Methods', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.savefig(os.path.join(output_dir, 'enhanced_answer_quality.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Knowledge Graph Coverage by Question
    plt.figure(figsize=(14, 8))
    
    # Convert coverage data to DataFrame for easier plotting
    coverage_data = pd.DataFrame({
        'Question': list(summary['kg_coverage']['coverage_by_question'].keys()),
        'Coverage': list(summary['kg_coverage']['coverage_by_question'].values())
    })
    
    # Sort by coverage
    coverage_data = coverage_data.sort_values('Coverage', ascending=False)
    
    # Plot
    bars = plt.bar(coverage_data['Question'], coverage_data['Coverage'], 
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(coverage_data))),
                  width=0.7)
    
    # Add average line
    avg_coverage = summary['kg_coverage']['average_coverage']
    plt.axhline(y=avg_coverage, color='red', linestyle='-', linewidth=2, 
               label=f'Average Coverage: {avg_coverage:.1f}%')
    
    # Add score labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylim(0, max(coverage_data['Coverage']) * 1.2)
    plt.ylabel('Knowledge Graph Coverage (%)', fontsize=12)
    plt.title('Knowledge Graph Coverage by Question', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'kg_coverage_by_question.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Method Performance Comparison by Question (Stacked and Heatmap)
    # First, create a DataFrame with question performance data
    question_performance = []
    for q_id, scores in summary['answer_quality']['quality_by_question'].items():
        question_performance.append({
            'Question': q_id,
            'LLM Only': scores['llm_only'],
            'Context Strict': scores['context_strict'],
            'LLM Informed': scores['llm_informed'],
            'Value Added': summary['context_contribution']['contribution_by_question'].get(q_id, 0)
        })
    
    perf_df = pd.DataFrame(question_performance)
    
    # 3.1 Grouped bar chart
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(perf_df))
    width = 0.25
    
    plt.bar(x - width, perf_df['LLM Only'], width, label='LLM Only', color=colors[0])
    plt.bar(x, perf_df['Context Strict'], width, label='Context Strict', color=colors[1])
    plt.bar(x + width, perf_df['LLM Informed'], width, label='LLM Informed', color=colors[2])
    
    plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel('Score (0-10)', fontsize=12)
    plt.title('Performance Comparison by Question and Method', fontsize=14, fontweight='bold')
    plt.xticks(x, perf_df['Question'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'performance_by_question.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 Heatmap for clearer comparison
    plt.figure(figsize=(12, 8))
    # Prepare data for heatmap
    heatmap_data = perf_df.set_index('Question')
    heatmap_data = heatmap_data.drop(columns=['Value Added'])
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=10,
                linewidths=.5, fmt='.2f', cbar_kws={'label': 'Score (0-10)'})
    
    plt.title('Answer Quality Heatmap by Question and Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Processing Time Analysis
    plt.figure(figsize=(14, 8))
    
    # Convert timing data to DataFrame
    timing_data = pd.DataFrame({
        'Question': list(summary['timing']['timing_by_question'].keys()),
        'Processing Time': [t['total_processing'] for t in summary['timing']['timing_by_question'].values()],
        'Evaluation Time': [t['evaluation_time'] for t in summary['timing']['timing_by_question'].values()]
    })
    
    # Sort by processing time
    timing_data = timing_data.sort_values('Processing Time', ascending=False)
    
    # Create stacked bar chart
    plt.bar(timing_data['Question'], 
            timing_data['Processing Time'] - timing_data['Evaluation Time'], 
            label='Knowledge Graph & Answer Generation',
            color=colors[4])
    
    plt.bar(timing_data['Question'], 
            timing_data['Evaluation Time'], 
            bottom=timing_data['Processing Time'] - timing_data['Evaluation Time'],
            label='Answer Evaluation',
            color=colors[5])
    
    # Add total time labels
    for i, (_, row) in enumerate(timing_data.iterrows()):
        plt.text(i, row['Processing Time'] + 1, 
                f'{row["Processing Time"]:.1f}s', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add average line
    avg_time = summary['timing']['average_total_processing']
    plt.axhline(y=avg_time, color='red', linestyle='-', linewidth=2, 
               label=f'Average Total Time: {avg_time:.1f}s')
    
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Processing Time Breakdown by Question', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'processing_time_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Relationship between KG Coverage and Answer Quality
    plt.figure(figsize=(12, 8))
    
    # Combine data
    correlation_data = pd.DataFrame({
        'Question': list(summary['kg_coverage']['coverage_by_question'].keys()),
        'KG Coverage': list(summary['kg_coverage']['coverage_by_question'].values())
    })
    
    # Merge with performance data
    correlation_data = correlation_data.merge(perf_df, on='Question')
    
    # Create scatter plot with regression line
    for method, color in zip(['LLM Only', 'Context Strict', 'LLM Informed'], colors[:3]):
        plt.scatter(correlation_data['KG Coverage'], correlation_data[method], 
                   label=method, color=color, s=100, alpha=0.7)
        
        # Add regression line
        z = np.polyfit(correlation_data['KG Coverage'], correlation_data[method], 1)
        p = np.poly1d(z)
        plt.plot(correlation_data['KG Coverage'], p(correlation_data['KG Coverage']), 
                '--', color=color, alpha=0.7)
    
    # Add question labels
    for i, row in correlation_data.iterrows():
        plt.annotate(row['Question'], 
                    (row['KG Coverage'], row['LLM Informed']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Knowledge Graph Coverage (%)', fontsize=12)
    plt.ylabel('Answer Quality Score (0-10)', fontsize=12)
    plt.title('Relationship Between KG Coverage and Answer Quality', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'coverage_quality_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Context Value Added Analysis
    plt.figure(figsize=(12, 6))
    
    # Value added is consistent across questions in your data, so just show the average
    plt.bar(['Value Added by Knowledge Context'], 
           [summary['context_contribution']['average_value_added']], 
           color=colors[3], width=0.4)
    
    plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.6, label="Medium Value")
    
    plt.text(0, summary['context_contribution']['average_value_added'] + 0.2, 
            f"{summary['context_contribution']['average_value_added']:.1f}/10", 
            ha='center', fontweight='bold', fontsize=14)
    
    plt.ylim(0, 10)
    plt.ylabel('Value Added Score (0-10)', fontsize=12)
    plt.title('Average Value Added by Knowledge Context', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'value_added_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a comprehensive dashboard summary image
    plt.figure(figsize=(15, 10))
    
    # Header
    plt.text(0.5, 0.98, 'USMLE Knowledge Graph Benchmark Results', 
            ha='center', fontsize=20, fontweight='bold')
    plt.text(0.5, 0.94, f'Analysis of {summary["questions_processed"]} Questions', 
            ha='center', fontsize=16)
    
    # Key Metrics
    plt.text(0.5, 0.88, 'Key Performance Metrics', ha='center', fontsize=18, fontweight='bold')
    
    metrics = [
        f'Average KG Coverage: {summary["kg_coverage"]["average_coverage"]:.1f}%',
        f'LLM Only Score: {summary["answer_quality"]["llm_only"]:.2f}/10',
        f'Context Strict Score: {summary["answer_quality"]["context_strict"]:.2f}/10',
        f'LLM Informed Score: {summary["answer_quality"]["llm_informed"]:.2f}/10',
        f'Value Added by Context: {summary["context_contribution"]["average_value_added"]:.1f}/10',
        f'Average Processing Time: {summary["timing"]["average_total_processing"]:.1f}s'
    ]
    
    for i, metric in enumerate(metrics):
        plt.text(0.5, 0.84 - i*0.04, metric, ha='center', fontsize=14)
    
    # Insights section
    plt.text(0.5, 0.56, 'Key Insights', ha='center', fontsize=18, fontweight='bold')
    
    # Calculate insights
    pct_improvement = ((summary["answer_quality"]["context_strict"] - summary["answer_quality"]["llm_only"]) / 
                       summary["answer_quality"]["llm_only"]) * 100
    
    best_question = max(summary['answer_quality']['quality_by_question'].items(), 
                       key=lambda x: x[1]['llm_informed'])[0]
    best_score = summary['answer_quality']['quality_by_question'][best_question]['llm_informed']
    
    insights = [
        f'Context-based methods improved answer quality by {pct_improvement:.1f}%',
        f'Knowledge context dramatically increased value (+{summary["context_contribution"]["average_value_added"]:.1f} points)',
        f'Best performing question: {best_question} (Score: {best_score:.2f}/10)',
        f'Average evaluation processing overhead: {summary["timing"]["average_evaluation_time"]:.1f}s'
    ]
    
    for i, insight in enumerate(insights):
        plt.text(0.5, 0.52 - i*0.04, insight, ha='center', fontsize=14)
    
    # Footer
    plt.text(0.5, 0.3, 'Visualizations saved to:', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.26, output_dir, ha='center', fontsize=12)
    
    # Remove axes
    plt.axis('off')
    
    # Save dashboard
    plt.savefig(os.path.join(output_dir, 'benchmark_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated 7 enhanced visualization charts in {output_dir}")
    return output_dir

def create_benchmark_visualizations(csv_path, output_dir):
    """
    Create visualizations from benchmark results CSV file
    
    Args:
        csv_path: Path to the simplified_results.csv file
        output_dir: Directory to save the visualization plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # 1. Performance Comparison Across Methods
    plt.figure(figsize=(12, 6))
    method_performance = df.groupby('Method')[['Evidence_Score_Raw', 'Correctness_Score_Raw', 'Total_Score']].mean()
    
    x = np.arange(len(method_performance.index))
    width = 0.25
    
    plt.bar(x - width, method_performance['Evidence_Score_Raw'], width, label='Evidence Score', color='skyblue')
    plt.bar(x, method_performance['Correctness_Score_Raw'], width, label='Correctness Score', color='lightgreen')
    plt.bar(x + width, method_performance['Total_Score'], width, label='Total Score', color='coral')
    
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Average Performance by Method')
    plt.xticks(x, method_performance.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_performance_comparison.png'))
    plt.close()
    
    # 2. Success Rate by Method
    plt.figure(figsize=(10, 6))
    success_rate = df.groupby('Method')['Correctness_Score_Raw'].apply(lambda x: (x == 10.0).mean() * 100)
    
    plt.bar(success_rate.index, success_rate.values, color=['blue', 'green', 'red'])
    plt.xlabel('Method')
    plt.ylabel('Success Rate (%)')
    plt.title('Answer Success Rate by Method')
    plt.xticks(rotation=45)
    
    # Add percentage labels on top of bars
    for i, v in enumerate(success_rate.values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_by_method.png'))
    plt.close()
    
    # 3. Score Distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Method', y='Total_Score')
    plt.xlabel('Method')
    plt.ylabel('Total Score')
    plt.title('Score Distribution by Method')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
    plt.close()
    
    # 4. Evidence vs Correctness Scatter Plot
    plt.figure(figsize=(12, 6))
    colors = {'LLM Only': 'blue', 'Context Strict': 'green', 'LLM Informed': 'red'}
    
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        plt.scatter(method_data['Evidence_Score_Raw'], 
                   method_data['Correctness_Score_Raw'],
                   label=method, 
                   alpha=0.6,
                   c=colors[method])
    
    plt.xlabel('Evidence Score')
    plt.ylabel('Correctness Score')
    plt.title('Evidence Score vs Correctness Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evidence_vs_correctness.png'))
    plt.close()
    
    # 5. Performance Over Questions
    plt.figure(figsize=(15, 6))
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        plt.plot(method_data['Question_ID'], 
                method_data['Total_Score'],
                label=method,
                marker='o',
                alpha=0.7)
    
    plt.xlabel('Question ID')
    plt.ylabel('Total Score')
    plt.title('Performance Across Questions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_across_questions.png'))
    plt.close()
    
    # Generate summary statistics
    summary = {
        'total_questions': len(df['Question_ID'].unique()),
        'method_performance': method_performance.to_dict(),
        'success_rates': success_rate.to_dict()
    }
    
    # Save summary as text file
    with open(os.path.join(output_dir, 'visualization_summary.txt'), 'w') as f:
        f.write("Benchmark Visualization Summary\n")
        f.write("=============================\n\n")
        f.write(f"Total Questions Analyzed: {summary['total_questions']}\n\n")
        
        f.write("Average Scores by Method:\n")
        f.write("-------------------------\n")
        for method in method_performance.index:
            f.write(f"\n{method}:\n")
            f.write(f"  Evidence Score: {method_performance.loc[method, 'Evidence_Score_Raw']:.2f}\n")
            f.write(f"  Correctness Score: {method_performance.loc[method, 'Correctness_Score_Raw']:.2f}\n")
            f.write(f"  Total Score: {method_performance.loc[method, 'Total_Score']:.2f}\n")
            f.write(f"  Success Rate: {success_rate[method]:.1f}%\n")
    
    return summary

# Example usage
if __name__ == "__main__":
    # Default paths 
    print(os.getcwd())
    csv_path = "usmle_evaluation_results/benchmark/simplified_results.csv"
    output_dir = "usmle_evaluation_results/benchmark/visualizations"
    
    # Create visualizations
    try:
        summary = create_benchmark_visualizations(csv_path, output_dir)
        print(f"Visualizations created successfully in {output_dir}")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
