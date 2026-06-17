# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

## Mixed Serving Boundary Benchmark

Use `mixed_serving_boundary` when a single run needs to stress several serving
boundaries at once: shared-prefix reuse, unique long-prefill requests, and
short-prefill long-decode requests.

```bash
vllm bench serve \
  --dataset-name mixed_serving_boundary \
  --num-prompts 300 \
  --request-rate 20 \
  --save-result \
  --save-detailed \
  --result-dir ./results
```

The saved JSON includes `request_class_metrics`, which breaks down TTFT, TPOT,
ITL, E2E latency, input tokens, and output tokens for each request class. This
helps compare scheduler, prefix-cache, and decode changes without running three
separate benchmarks.
