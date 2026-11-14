# Configuration Options

This section lists the most common options for running vLLM.

There are three main levels of configuration, from highest priority to lowest priority:

- [Request parameters](../serving/online_serving/openai_compatible_server.md#completions-api) and [input arguments](../api/README.md#inference-parameters)
- [Engine arguments](./engine_args.md)
- [Environment variables](./env_vars.md)

For performance settings, see [Optimization and Tuning](./optimization.md),
including [Qwen3.8-Flash-Next HC sequence parallelism](./optimization.md#qwen38-flash-next-hc-sequence-parallelism).
