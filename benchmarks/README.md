# Benchmarks

This directory contains vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance  
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark Tools documentation](../docs/contributing/benchmarks.md#benchmark-tools).

## Quick Start

```bash
# Online serving benchmark
vllm bench serve --model MODEL_NAME --dataset-name DATASET --dataset-path PATH

# Offline throughput benchmark  
vllm bench throughput --model MODEL_NAME --dataset-name DATASET --dataset-path PATH
```

Replace `MODEL_NAME`, `DATASET`, and `PATH` with your specific model, dataset type, and data file path.