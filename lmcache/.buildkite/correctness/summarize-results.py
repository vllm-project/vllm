# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, List, Tuple
import json
import os
import sys


def load_jsonl_file(filepath: str) -> Dict[str, Dict[str, float]]:
    """Load a JSONL file and return a dictionary of results."""
    results: dict[str, dict[str, float]] = {}

    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found")
        return results

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    results.update(data)
                except json.JSONDecodeError:
                    print(f"Error parsing line in {filepath}: {line}")
                    continue

    return results


def compare_results(
    baseline_results: Dict, lmcache_results: Dict
) -> Tuple[List[Tuple], float, float]:
    """Compare baseline and lmcache results, return comparison data and totals."""
    comparisons = []
    baseline_total_correct = 0
    baseline_total_questions = 0
    lmcache_total_correct = 0
    lmcache_total_questions = 0

    # Get all subjects (excluding 'total' for now)
    all_subjects = set(baseline_results.keys()) | set(lmcache_results.keys())
    all_subjects.discard("total")  # Handle total separately

    # Sort subjects alphabetically
    for subject in sorted(all_subjects):
        baseline_data = baseline_results.get(
            subject, {"accuracy": 0.0, "num_questions": 0}
        )
        lmcache_data = lmcache_results.get(
            subject, {"accuracy": 0.0, "num_questions": 0}
        )

        baseline_acc = baseline_data["accuracy"]
        lmcache_acc = lmcache_data["accuracy"]
        difference = lmcache_acc - baseline_acc

        # Use the number of questions from whichever dataset has the subject
        num_questions = max(
            baseline_data["num_questions"], lmcache_data["num_questions"]
        )

        comparisons.append(
            (subject, baseline_acc, lmcache_acc, difference, num_questions)
        )

        # Accumulate totals
        baseline_total_correct += baseline_acc * num_questions
        baseline_total_questions += num_questions
        lmcache_total_correct += lmcache_acc * num_questions
        lmcache_total_questions += num_questions

    # Calculate overall accuracies
    baseline_total_acc = (
        baseline_total_correct / baseline_total_questions
        if baseline_total_questions > 0
        else 0
    )
    lmcache_total_acc = (
        lmcache_total_correct / lmcache_total_questions
        if lmcache_total_questions > 0
        else 0
    )

    return comparisons, baseline_total_acc, lmcache_total_acc


def format_model_report(
    model_name: str,
    model_short_name: str,
    comparisons: List[Tuple],
    baseline_total: float,
    lmcache_total: float,
) -> str:
    """Format a single model's comparison report."""
    report = []
    report.append("=" * 80)
    report.append(f"MODEL: {model_name}")
    report.append("=" * 80)
    report.append("")

    # Header
    report.append(
        f"{'Subject':<35} \
        {'Baseline':<12} \
        {'LMCache':<12} \
        {'Difference':<12} \
        {'Questions':<10}"
    )
    report.append("-" * 80)

    # Subject comparisons
    for subject, baseline_acc, lmcache_acc, difference, num_questions in comparisons:
        report.append(
            f"{subject:<35} \
            {baseline_acc:<12.4f} \
            {lmcache_acc:<12.4f} \
            {difference:<12.4f} \
            {num_questions:<10}"
        )

    # Totals
    total_difference = lmcache_total - baseline_total
    report.append("-" * 80)
    report.append(
        f"{'TOTAL':<35} \
        {baseline_total:<12.4f} \
        {lmcache_total:<12.4f} \
        {total_difference:<12.4f}"
    )
    report.append("")

    # Summary
    report.append("SUMMARY:")
    report.append(f"  Baseline Total Accuracy: {baseline_total:.4f}")
    report.append(f"  LMCache Total Accuracy:  {lmcache_total:.4f}")
    report.append(f"  Net Accuracy Difference: {total_difference:.4f}")
    if total_difference >= 0:
        report.append(f"  Result: LMCache performs BETTER by {total_difference:.4f}")
    else:
        report.append(
            f"  Result: LMCache performs WORSE by {abs(total_difference):.4f}"
        )
    report.append("")

    return "\n".join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize-results.py <model1> <model2> ...")
        print(
            "Example: python summarize-results.py \
                meta-llama/Llama-3.1-8B-Instruct \
                deepseek-ai/DeepSeek-V2-Lite"
        )
        sys.exit(1)

    model_names = sys.argv[1:]
    all_reports = []

    print(f"Processing {len(model_names)} models...")

    for model_name in model_names:
        print(f"Processing model: {model_name}")

        # Extract short name using the same logic as mmlu-test.py
        model_short_name = model_name.split("/")[-1]

        # Define file paths
        baseline_file = f"{model_short_name}-baseline.jsonl"
        lmcache_file = f"{model_short_name}-lmcache.jsonl"

        print(f"  Looking for files: {baseline_file}, {lmcache_file}")

        # Load results
        baseline_results = load_jsonl_file(baseline_file)
        lmcache_results = load_jsonl_file(lmcache_file)

        if not baseline_results and not lmcache_results:
            print(f"  Warning: No data found for {model_name}")
            continue

        # Compare results
        comparisons, baseline_total, lmcache_total = compare_results(
            baseline_results, lmcache_results
        )

        # Generate report for this model
        model_report = format_model_report(
            model_name,
            model_short_name,
            comparisons,
            baseline_total,
            lmcache_total,
        )
        all_reports.append(model_report)

        print(f"  Processed {len(comparisons)} subjects")

    # Write combined report to file
    output_file = "correctness-summary.txt"
    with open(output_file, "w") as f:
        f.write("MMLU Correctness Comparison Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated for {len(model_names)} models\n\n")

        for report in all_reports:
            f.write(report)
            f.write("\n")

    print(f"\nReport written to {output_file}")


if __name__ == "__main__":
    main()
