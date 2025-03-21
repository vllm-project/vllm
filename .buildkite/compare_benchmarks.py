# SPDX-License-Identifier: Apache-2.0
"""
Compare some benchmark results (as outputted by `benchmark_serving.py`)
against a provided baseline.
"""
import re
import sys
from typing import Dict

# Define the metrics and their allowed percentage thresholds for regressions.
# Unspecified metrics will not be checked.
METRICS_THRESHOLDS = {
    # Maximum X% decrease allowed (positive values for throughput)
    "Output token throughput (tok/s)": 3.0,
    "Total Token throughput (tok/s)": 3.0,
    # Maximum X% increase allowed (negative values for latencies)
    "Median TTFT (ms)": -2.5,
    "Median ITL (ms)": -2.5,
}


def parse_metrics(file_path: str) -> Dict[str, float]:
    metrics = {}
    with open(file_path) as f:
        for line in f:
            # Match lines with format: "some_name: some_digits"
            match = re.match(r"(.+?):\s+([\d.]+)", line)
            if match:
                metric_name = match.group(1).strip()
                metric_value = float(match.group(2))
                metrics[metric_name] = metric_value
    return metrics


def compare_metrics(baseline_metrics: Dict[str, float],
                    current_metrics: Dict[str, float]):
    regressions = []

    for metric, threshold in METRICS_THRESHOLDS.items():
        baseline_value = baseline_metrics[metric]
        current_value = current_metrics[metric]

        # Calculate percentage change
        percent_change = (
            (current_value - baseline_value) / baseline_value) * 100

        # Throughput OR Latency regression
        if (threshold >= 0 and percent_change < -threshold) or (
                threshold < 0 and percent_change > -threshold):
            regressions.append(
                (metric, baseline_value, current_value, percent_change))

    return regressions


def main(baseline_file, current_file):
    baseline_metrics = parse_metrics(baseline_file)
    current_metrics = parse_metrics(current_file)

    regressions = compare_metrics(baseline_metrics, current_metrics)
    if regressions:
        print("\nRegressions detected:")
        for metric, baseline, current, percent_change in regressions:
            print(f"  - {metric}: Baseline={baseline}, \
                    Current={current}, Change={percent_change:.2f}%")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmarks.py <baseline_benchmark_file> \
                <current_benchmark_file>")
        sys.exit(1)

    baseline_file = sys.argv[1]
    current_file = sys.argv[2]

    main(baseline_file, current_file)
