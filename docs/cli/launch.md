# vllm launch

The `vllm launch` command starts individual vLLM components instead of the full inference stack.

## Available Components

Currently, the public component exposed by the CLI is `render`.

## render

`vllm launch render` starts a GPU-less rendering server for preprocessing and postprocessing only.

```bash
vllm launch render meta-llama/Llama-3.2-1B-Instruct --port 8100
```

Useful help commands:

```bash
# Show all available flags
vllm launch render --help=all

# Inspect a config group
vllm launch render --help=Frontend

# Inspect one flag or search by keyword
vllm launch render --help=port
vllm launch render --help=ssl
```

The `render` component reuses the standard serving argument parser, so model, frontend, networking, and related CLI options follow the same conventions as [`vllm serve`](./serve.md).
