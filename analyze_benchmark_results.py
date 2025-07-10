#!/usr/bin/env python3
"""
Analyze benchmark results and create CSV summary.

This script reads all JSON benchmark result files from a directory and creates
a comprehensive CSV summary with all metrics and input parameters.

Usage:
    python analyze_benchmark_results.py --input-dir ./results --output results_summary.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates CSV summaries."""

    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.results: List[Dict[str, Any]] = []

    def find_json_files(self) -> List[Path]:
        """Find all JSON files in the input directory."""
        json_files = list(self.input_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {self.input_dir}")
        return json_files

    def load_benchmark_data(self, json_files: List[Path]) -> None:
        """Load and parse all benchmark JSON files."""
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extract configuration from filename if possible
                filename_parts = json_file.stem.split('-')
                if len(filename_parts) >= 4:
                    # Format: concurrency-input_len-output_len-backend.json
                    try:
                        concurrency = int(filename_parts[0])
                        input_len = int(filename_parts[1])
                        output_len = int(filename_parts[2])
                        backend = filename_parts[3]

                        data['filename_concurrency'] = concurrency
                        data['filename_input_len'] = input_len
                        data['filename_output_len'] = output_len
                        data['filename_backend'] = backend
                    except (ValueError, IndexError):
                        pass

                data['source_file'] = str(json_file.name)
                self.results.append(data)

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load {json_file}: {e}")
                continue

        print(f"Loaded {len(self.results)} benchmark results")

    def extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all relevant metrics from a benchmark result."""
        metrics = {}

        # Basic info
        metrics['source_file'] = data.get('source_file', '')
        metrics['date'] = data.get('date', '')
        metrics['backend'] = data.get('backend', '')
        metrics['model_id'] = data.get('model_id', '')

        # Configuration parameters
        metrics['num_prompts'] = data.get('num_prompts', 0)
        metrics['request_rate'] = data.get('request_rate', '')
        metrics['burstiness'] = data.get('burstiness', 0)
        metrics['max_concurrency'] = data.get('max_concurrency', 0)

        # Configuration from filename (more reliable)
        metrics['concurrency'] = data.get('filename_concurrency',
                                          data.get('max_concurrency', 0))
        metrics['input_len'] = data.get('filename_input_len', 0)
        metrics['output_len'] = data.get('filename_output_len', 0)
        metrics['backend_name'] = data.get('filename_backend',
                                           data.get('backend', ''))

        # Performance metrics
        metrics['duration_s'] = data.get('duration', 0)
        metrics['completed_requests'] = data.get('completed', 0)
        metrics['total_input_tokens'] = data.get('total_input_tokens', 0)
        metrics['total_output_tokens'] = data.get('total_output_tokens', 0)

        # Throughput metrics
        metrics['request_throughput'] = data.get('request_throughput', 0)
        metrics['request_goodput'] = data.get('request_goodput', 0)
        metrics['output_throughput'] = data.get('output_throughput', 0)
        metrics['total_token_throughput'] = data.get('total_token_throughput',
                                                     0)

        # Latency metrics (TTFT - Time to First Token)
        metrics['mean_ttft_ms'] = data.get('mean_ttft_ms', 0)
        metrics['median_ttft_ms'] = data.get('median_ttft_ms', 0)
        metrics['std_ttft_ms'] = data.get('std_ttft_ms', 0)
        metrics['p99_ttft_ms'] = data.get('p99_ttft_ms', 0)

        # Latency metrics (TPOT - Time per Output Token)
        metrics['mean_tpot_ms'] = data.get('mean_tpot_ms', 0)
        metrics['median_tpot_ms'] = data.get('median_tpot_ms', 0)
        metrics['std_tpot_ms'] = data.get('std_tpot_ms', 0)
        metrics['p99_tpot_ms'] = data.get('p99_tpot_ms', 0)

        # Latency metrics (ITL - Inter-Token Latency)
        metrics['mean_itl_ms'] = data.get('mean_itl_ms', 0)
        metrics['median_itl_ms'] = data.get('median_itl_ms', 0)
        metrics['std_itl_ms'] = data.get('std_itl_ms', 0)
        metrics['p99_itl_ms'] = data.get('p99_itl_ms', 0)

        # End-to-end latency (if available)
        metrics['mean_e2el_ms'] = data.get('mean_e2el_ms', 0)
        metrics['median_e2el_ms'] = data.get('median_e2el_ms', 0)
        metrics['std_e2el_ms'] = data.get('std_e2el_ms', 0)
        metrics['p99_e2el_ms'] = data.get('p99_e2el_ms', 0)

        # Additional percentiles (if available)
        for percentile in ['p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
            for metric_type in ['ttft', 'tpot', 'itl', 'e2el']:
                key = f'{percentile}_{metric_type}_ms'
                metrics[key] = data.get(key, 0)

        # Calculated metrics
        if metrics['total_input_tokens'] > 0 and metrics[
                'total_output_tokens'] > 0:
            metrics['avg_input_tokens_per_request'] = metrics[
                'total_input_tokens'] / max(metrics['completed_requests'], 1)
            metrics['avg_output_tokens_per_request'] = metrics[
                'total_output_tokens'] / max(metrics['completed_requests'], 1)
        else:
            metrics['avg_input_tokens_per_request'] = 0
            metrics['avg_output_tokens_per_request'] = 0

        return metrics

    def generate_csv_summary(self) -> None:
        """Generate CSV summary from all benchmark results."""
        if not self.results:
            print("No benchmark results to analyze")
            return

        # Extract metrics from all results
        all_metrics = []
        for result in self.results:
            metrics = self.extract_metrics(result)
            all_metrics.append(metrics)

        if not all_metrics:
            print("No metrics extracted")
            return

        # Get all possible fieldnames from all results
        fieldnames = set()
        for metrics in all_metrics:
            fieldnames.update(metrics.keys())

        # Define field order for better readability
        ordered_fields = [
            'source_file', 'date', 'backend_name', 'model_id', 'concurrency',
            'input_len', 'output_len', 'num_prompts', 'request_rate',
            'burstiness', 'max_concurrency', 'duration_s',
            'completed_requests', 'total_input_tokens', 'total_output_tokens',
            'avg_input_tokens_per_request', 'avg_output_tokens_per_request',
            'request_throughput', 'request_goodput', 'output_throughput',
            'total_token_throughput', 'mean_ttft_ms', 'median_ttft_ms',
            'std_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms',
            'std_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms', 'median_itl_ms',
            'std_itl_ms', 'p99_itl_ms', 'mean_e2el_ms', 'median_e2el_ms',
            'std_e2el_ms', 'p99_e2el_ms'
        ]

        # Add any remaining fields
        remaining_fields = sorted(fieldnames - set(ordered_fields))
        final_fieldnames = ordered_fields + remaining_fields

        # Write CSV file
        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames)
                writer.writeheader()

                # Sort results for consistent output
                sorted_metrics = sorted(
                    all_metrics,
                    key=lambda x:
                    (x.get('backend_name', ''), x.get('concurrency', 0),
                     x.get('input_len', 0), x.get('output_len', 0)))

                for metrics in sorted_metrics:
                    writer.writerow(metrics)

            print(f"CSV summary written to: {self.output_file}")
            print(f"Total records: {len(all_metrics)}")

        except IOError as e:
            print(f"Error writing CSV file: {e}")
            sys.exit(1)

    def generate_summary_stats(self) -> None:
        """Generate and print summary statistics."""
        if not self.results:
            return

        # Group by backend
        backends = {}
        for result in self.results:
            backend = result.get('filename_backend',
                                 result.get('backend', 'unknown'))
            if backend not in backends:
                backends[backend] = []
            backends[backend].append(result)

        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY STATISTICS")
        print(f"{'='*60}")

        for backend, results in backends.items():
            print(f"\n{backend.upper()} Backend:")
            print(f"  Total benchmark runs: {len(results)}")

            # Calculate averages
            if results:
                total_throughput = sum(
                    r.get('request_throughput', 0) for r in results)
                avg_throughput = total_throughput / len(results)
                total_token_throughput = sum(
                    r.get('total_token_throughput', 0) for r in results)
                avg_token_throughput = total_token_throughput / len(results)

                print(
                    f"  Average request throughput: {avg_throughput:.2f} req/s"
                )
                print(
                    f"  Average token throughput: {avg_token_throughput:.2f} tok/s"
                )

                # Find best performing configurations
                best_request_throughput = max(
                    results, key=lambda x: x.get('request_throughput', 0))
                best_token_throughput = max(
                    results, key=lambda x: x.get('total_token_throughput', 0))

                print(
                    f"  Best request throughput: {best_request_throughput.get('request_throughput', 0):.2f} req/s"
                )
                print(
                    f"  Best token throughput: {best_token_throughput.get('total_token_throughput', 0):.2f} tok/s"
                )

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print(f"Analyzing benchmark results in: {self.input_dir}")

        # Find and load JSON files
        json_files = self.find_json_files()
        if not json_files:
            print("No JSON files found in input directory")
            sys.exit(1)

        self.load_benchmark_data(json_files)
        if not self.results:
            print("No valid benchmark data loaded")
            sys.exit(1)

        # Generate outputs
        self.generate_csv_summary()
        self.generate_summary_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and create CSV summary")
    parser.add_argument("--input-dir",
                        default=".",
                        help="Directory containing JSON benchmark results")
    parser.add_argument("--output",
                        default="benchmark_results_summary.csv",
                        help="Output CSV file path")
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    # Run analysis
    analyzer = BenchmarkAnalyzer(args.input_dir, args.output)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
