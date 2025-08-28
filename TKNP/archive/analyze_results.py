#!/usr/bin/env python3
"""
Results Analysis Script for Batch Inference Experiments

This script analyzes the results from batch inference experiments and generates
visualizations and summaries.

Usage:
    python analyze_results.py --input-dir experiment_results
    python analyze_results.py --input-file experiment_results/tp_2_results.json
"""

import argparse
import json
import os
import glob
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def load_results(input_path: str) -> List[Dict[str, Any]]:
    """Load experiment results from file or directory."""
    if os.path.isfile(input_path):
        # Single file
        with open(input_path, 'r') as f:
            return json.load(f)
    elif os.path.isdir(input_path):
        # Directory - load all JSON files
        results = []
        for file_path in glob.glob(os.path.join(input_path, "*.json")):
            with open(file_path, 'r') as f:
                file_results = json.load(f)
                if isinstance(file_results, list):
                    results.extend(file_results)
                else:
                    results.append(file_results)
        return results
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def create_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis."""
    df_data = []
    
    for result in results:
        config = result.get('config', {})
        df_data.append({
            'tensor_parallel_size': config.get('tensor_parallel_size', 1),
            'pipeline_parallel_size': config.get('pipeline_parallel_size', 1),
            'data_parallel_size': config.get('data_parallel_size', 1),
            'batch_size': config.get('batch_size', 1),
            'seq_len': config.get('seq_len', 512),
            'model': config.get('model', 'unknown'),
            'throughput_tokens_per_sec': result.get('throughput_tokens_per_sec', 0),
            'throughput_requests_per_sec': result.get('throughput_requests_per_sec', 0),
            'latency_ms': result.get('latency_ms', 0),
            'memory_usage_gb': result.get('memory_usage_gb', 0),
            'gpu_utilization': result.get('gpu_utilization', 0),
            'run_time_seconds': result.get('run_time_seconds', 0),
            'tokens_generated': result.get('tokens_generated', 0),
            'requests_processed': result.get('requests_processed', 0),
        })
    
    df = pd.DataFrame(df_data)
    
    # Add derived columns
    df['total_gpus'] = df['tensor_parallel_size'] * df['pipeline_parallel_size'] * df['data_parallel_size']
    df['parallel_config'] = df.apply(
        lambda row: f"TP{row['tensor_parallel_size']}_PP{row['pipeline_parallel_size']}_DP{row['data_parallel_size']}", 
        axis=1
    )
    df['efficiency'] = df['throughput_tokens_per_sec'] / df['total_gpus']
    
    return df


def print_summary(df: pd.DataFrame):
    """Print a summary of the results."""
    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"Total experiments: {len(df)}")
    print(f"Models tested: {df['model'].unique()}")
    print(f"Parallel configurations: {sorted(df['parallel_config'].unique())}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"Sequence lengths: {sorted(df['seq_len'].unique())}")
    
    print("\nPerformance Summary:")
    print("-" * 40)
    print(f"Max throughput (tokens/sec): {df['throughput_tokens_per_sec'].max():.2f}")
    print(f"Max throughput (requests/sec): {df['throughput_requests_per_sec'].max():.2f}")
    print(f"Min latency (ms): {df['latency_ms'].min():.2f}")
    print(f"Max memory usage (GB): {df['memory_usage_gb'].max():.2f}")
    
    print("\nBest configurations by metric:")
    print("-" * 40)
    
    # Best by token throughput
    best_tokens = df.loc[df['throughput_tokens_per_sec'].idxmax()]
    print(f"Best token throughput: {best_tokens['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Config: {best_tokens['parallel_config']}, Batch: {best_tokens['batch_size']}, Seq: {best_tokens['seq_len']}")
    
    # Best by request throughput
    best_requests = df.loc[df['throughput_requests_per_sec'].idxmax()]
    print(f"Best request throughput: {best_requests['throughput_requests_per_sec']:.2f} requests/sec")
    print(f"  Config: {best_requests['parallel_config']}, Batch: {best_requests['batch_size']}, Seq: {best_requests['seq_len']}")
    
    # Best by latency
    best_latency = df.loc[df['latency_ms'].idxmin()]
    print(f"Best latency: {best_latency['latency_ms']:.2f} ms")
    print(f"  Config: {best_latency['parallel_config']}, Batch: {best_latency['batch_size']}, Seq: {best_latency['seq_len']}")


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create various visualizations of the results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Throughput vs Batch Size by Parallel Config
    plt.figure(figsize=(12, 8))
    for config in sorted(df['parallel_config'].unique()):
        config_data = df[df['parallel_config'] == config]
        plt.plot(config_data['batch_size'], config_data['throughput_tokens_per_sec'], 
                marker='o', label=config, linewidth=2, markersize=6)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Token Throughput vs Batch Size by Parallel Configuration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_vs_batch_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Latency vs Batch Size by Parallel Config
    plt.figure(figsize=(12, 8))
    for config in sorted(df['parallel_config'].unique()):
        config_data = df[df['parallel_config'] == config]
        plt.plot(config_data['batch_size'], config_data['latency_ms'], 
                marker='s', label=config, linewidth=2, markersize=6)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title('Latency vs Batch Size by Parallel Configuration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_vs_batch_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency vs Total GPUs
    plt.figure(figsize=(10, 6))
    efficiency_data = df.groupby('total_gpus')['efficiency'].mean().reset_index()
    plt.plot(efficiency_data['total_gpus'], efficiency_data['efficiency'], 
            marker='o', linewidth=2, markersize=8)
    
    plt.xlabel('Total GPUs')
    plt.ylabel('Efficiency (tokens/sec/GPU)')
    plt.title('Scaling Efficiency vs Number of GPUs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap: Throughput by Batch Size and Sequence Length
    plt.figure(figsize=(12, 8))
    # Use the most common parallel config for this visualization
    most_common_config = df['parallel_config'].mode()[0]
    config_data = df[df['parallel_config'] == most_common_config]
    
    pivot_data = config_data.pivot_table(
        values='throughput_tokens_per_sec', 
        index='batch_size', 
        columns='seq_len', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Throughput (tokens/sec)'})
    plt.title(f'Token Throughput Heatmap ({most_common_config})')
    plt.xlabel('Sequence Length')
    plt.ylabel('Batch Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Memory Usage by Configuration
    plt.figure(figsize=(10, 6))
    memory_data = df.groupby('parallel_config')['memory_usage_gb'].max().sort_values(ascending=False)
    plt.bar(range(len(memory_data)), memory_data.values)
    plt.xticks(range(len(memory_data)), memory_data.index, rotation=45, ha='right')
    plt.xlabel('Parallel Configuration')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Maximum Memory Usage by Parallel Configuration')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def create_detailed_report(df: pd.DataFrame, output_file: str):
    """Create a detailed CSV report of all results."""
    # Sort by parallel config, then batch size, then seq len
    df_sorted = df.sort_values(['parallel_config', 'batch_size', 'seq_len'])
    
    # Save to CSV
    df_sorted.to_csv(output_file, index=False)
    print(f"Detailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze batch inference experiment results")
    parser.add_argument("--input", required=True, 
                       help="Input file or directory containing results")
    parser.add_argument("--output-dir", default="analysis_output",
                       help="Output directory for visualizations")
    parser.add_argument("--report-file", default="detailed_results.csv",
                       help="Output file for detailed CSV report")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    
    if not results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df = create_dataframe(results)
    
    # Print summary
    print_summary(df)
    
    # Create visualizations
    if not args.no_plots:
        print(f"\nCreating visualizations...")
        create_visualizations(df, args.output_dir)
    
    # Create detailed report
    report_path = os.path.join(args.output_dir, args.report_file)
    create_detailed_report(df, report_path)
    
    print(f"\nAnalysis complete!")
    print(f"Results: {len(df)} experiments analyzed")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main() 