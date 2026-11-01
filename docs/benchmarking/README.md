# Benchmark Suites

vLLM provides comprehensive benchmarking tools for performance testing and evaluation.

- **[Benchmark CLI](./cli.md)**: `vllm bench` CLI tools and specialized benchmark scripts for interactive performance testing.
- **[Parameter Sweeps](./sweeps.md)**: Automate `vllm bench` runs for multiple configurations, useful for [optimization and tuning](../configuration/optimization.md).
- **[Performance Dashboard](./dashboard.md)**: Automated CI that publishes benchmarks on each commit.
- **[Benchmark using vLLM Recipes](../../tools/recipes/README.md)**: Use the Recipe conversion tool to generate `config.yaml` and `env.sh` from [vLLM Recipes](https://recipes.vllm.ai/) for Benchmarking.

## Activation Kernel Benchmarks

Compare eager PyTorch, a vLLM custom operator, and `torch.compile` using:

```bash
python benchmarks/kernels/benchmark_activation.py --func-name silu_and_mul --dtype bfloat16
```

The `PyTorch (eager)` column calls the layer's uncompiled `forward_native`
implementation. `Custom OP` calls the layer's CUDA implementation, and `Compiled`
compiles `forward_native` with `torch.compile`. All three use the same shape
grid, seeded input generation, and CUDA graph timing. Results are in
milliseconds and measure steady-state kernel latency; they exclude compilation
time and do not measure end-to-end serving throughput.
