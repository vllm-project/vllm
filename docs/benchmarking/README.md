# Benchmark Suites

vLLM provides comprehensive benchmarking tools for performance testing and evaluation.

## Benchmark a vLLM Recipes configuration

[vLLM Recipes](https://recipes.vllm.ai/) can be converted into a native vLLM
configuration and environment file. See the
[Recipes conversion tool README](../../tools/recipes/README.md) for generation and discovery
options.

Before benchmarking, load the generated environment and start the server:

```bash
source env.sh
vllm serve --config config.yaml
```

Then run the desired `vllm bench` workload against that server.

- **[Benchmark CLI](./cli.md)**: `vllm bench` CLI tools and specialized benchmark scripts for interactive performance testing.
- **[Parameter Sweeps](./sweeps.md)**: Automate `vllm bench` runs for multiple configurations, useful for [optimization and tuning](../configuration/optimization.md).
- **[Performance Dashboard](./dashboard.md)**: Automated CI that publishes benchmarks on each commit.
