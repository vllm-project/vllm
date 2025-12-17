#!/usr/bin/env python3
"""
Bisect Validator - Dynamic pipeline step generator for git bisect.

This script reads .buildkite/test-pipeline.yaml and outputs filtered YAML to stdout
containing only the test steps matching the TARGET_JOBS environment variable.

IMPORTANT: This script does NOT trigger any builds. It only generates YAML output.
The YAML is consumed by bisect-validator-pipeline.yaml which uploads it to Buildkite.

Environment Variables:
    TARGET_JOBS: Comma-separated list of job labels to include (optional)
                 If empty or not set, includes all jobs.

Usage:
    # Called automatically by bisect-validator-pipeline.yaml
    python3 .buildkite/bisect/bisect_validator.py | buildkite-agent pipeline upload

    # Or run locally to see generated YAML (no builds triggered)
    TARGET_JOBS="job1,job2" python3 .buildkite/bisect/bisect_validator.py

Concrete Example:
    $ cd /path/to/vllm
    $ TARGET_JOBS="Basic Correctness Test,Engine Test" \
        python3 .buildkite/bisect/bisect_validator.py

    Output (to stdout):
        steps:
        - label: Basic Correctness Test
          timeout_in_minutes: 30
          commands:
          - export VLLM_WORKER_MULTIPROC_METHOD=spawn
          - pytest -v -s basic_correctness/test_basic_correctness.py
          ...
        - label: Engine Test
          timeout_in_minutes: 15
          commands:
          - pytest -v -s engine test_sequence.py
          ...

    (stderr shows: "Generating validation pipeline for jobs: Basic Correctness Test,Engine Test")
    (stderr shows: "Generated 2 step(s)")

See .buildkite/bisect/README.md for complete documentation.
"""

import os
import sys
import yaml
import re


def load_test_pipeline():
    """Load and parse the test-pipeline.yaml file."""
    # Path relative to repository root
    pipeline_path = os.path.join(
        os.path.dirname(__file__), "..", "test-pipeline.yaml"
    )

    with open(pipeline_path, 'r') as f:
        # Read the file and strip Jinja template syntax for basic parsing
        content = f.read()

        # Remove common Jinja patterns to make it valid YAML
        # This is a simple approach - we're just extracting step definitions
        # Note: This won't work for complex Jinja logic, but works for basic templates
        content = re.sub(r'\{\{.*?\}\}', '""', content)
        content = re.sub(r'\{%.*?%\}', '', content)

    try:
        pipeline = yaml.safe_load(content)
        return pipeline
    except yaml.YAMLError as e:
        print(f"Error parsing pipeline YAML: {e}", file=sys.stderr)
        sys.exit(1)


def normalize_label(label):
    """Normalize a label for comparison (lowercase, strip whitespace)."""
    return label.strip().lower()


def filter_steps_by_labels(steps, target_labels):
    """
    Filter pipeline steps to only include those matching target labels.

    Args:
        steps: List of step definitions from pipeline YAML
        target_labels: List of job labels to include

    Returns:
        List of filtered steps
    """
    if not target_labels:
        # If no target labels specified, return all steps
        return steps

    # Normalize target labels for case-insensitive matching
    normalized_targets = {normalize_label(label) for label in target_labels}

    filtered_steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue

        label = step.get('label', '')
        if normalize_label(label) in normalized_targets:
            filtered_steps.append(step)

    return filtered_steps


def generate_validation_pipeline(target_jobs):
    """
    Generate a validation pipeline with only the specified jobs.

    Args:
        target_jobs: Comma-separated string of job labels to include

    Returns:
        Dictionary representing the pipeline YAML
    """
    # Parse target jobs
    if target_jobs:
        target_labels = [label.strip() for label in target_jobs.split(',') if label.strip()]
    else:
        target_labels = []

    # Load test pipeline
    test_pipeline = load_test_pipeline()

    if not test_pipeline or 'steps' not in test_pipeline:
        print("Error: Could not load steps from test-pipeline.yaml", file=sys.stderr)
        sys.exit(1)

    # Filter steps
    filtered_steps = filter_steps_by_labels(test_pipeline['steps'], target_labels)

    if not filtered_steps:
        if target_labels:
            print(f"Warning: No steps matched target jobs: {target_labels}", file=sys.stderr)
            print("Available steps:", file=sys.stderr)
            for step in test_pipeline['steps']:
                if isinstance(step, dict) and 'label' in step:
                    print(f"  - {step['label']}", file=sys.stderr)
            sys.exit(1)
        else:
            # No target specified but also no steps - use all
            filtered_steps = test_pipeline['steps']

    # Generate output pipeline
    output_pipeline = {
        'steps': filtered_steps
    }

    return output_pipeline


def main():
    # Get TARGET_JOBS from environment
    target_jobs = os.environ.get('TARGET_JOBS', '')

    print(f"Generating validation pipeline for jobs: {target_jobs or 'ALL'}", file=sys.stderr)

    # Generate pipeline
    pipeline = generate_validation_pipeline(target_jobs)

    # Output as YAML
    print(yaml.dump(pipeline, default_flow_style=False, sort_keys=False))

    print(f"Generated {len(pipeline['steps'])} step(s)", file=sys.stderr)


if __name__ == '__main__':
    main()
