# Structured Outputs

This script demonstrates various structured output capabilities of vLLM's OpenAI-compatible server.
It can run individual constraint type or all of them.
It supports both streaming responses and concurrent non-streaming requests.

To use this example, you must start an vLLM server with any model of your choice.

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct
```

To serve a reasoning model, you can use the following command:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-parser deepseek_r1
```

If you want to run this script standalone with `uv`, you can use the following:

```bash

```

Examples:

Run all constraints, non-streaming:

```bash
uv run <file>
```

Run all constraints, streaming:

```bash
uv run <file> --stream
```

Run certain constraints, for example `structural_tag` and `regex`, streaming:

```bash
uv run <file> --constraint structural_tags regex --stream
```

Run all constraints, with reasoning models and streaming:

```bash
uv run <file> --reasoning --stream
```
