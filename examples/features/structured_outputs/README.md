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
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --reasoning-parser deepseek_r1
```

If you want to run this script standalone with `uv`, you can use the following:

```bash
uvx --from git+https://github.com/vllm-project/vllm#subdirectory=examples/features/structured_outputs \
    structured-outputs
```

See [feature docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) for more information.

!!! tip
    If vLLM is running remotely, then set `OPENAI_BASE_URL=<remote_url>` before running the script.

## Chimera-style edit programs

`chimera_edit_program.py` shows how to combine structured outputs with a
deterministic client-side renderer for coding-agent workloads. Instead of asking
the model to repeat every edited source file, the model emits a compact JSON edit
program such as "prepend this shared text to these paths", and the client
validates and renders the final files locally.

Start a vLLM server:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct
```

Run the edit-program example:

```bash
uv run chimera_edit_program.py
```

Run only the deterministic renderer without a vLLM server:

```bash
uv run chimera_edit_program.py --dry-run
```

## Usage

Run all constraints, non-streaming:

```bash
uv run structured_outputs_offline.py
```

Run all constraints, streaming:

```bash
uv run structured_outputs_offline.py --stream
```

Run certain constraints, for example `structural_tag` and `regex`, streaming:

```bash
uv run structured_outputs_offline.py \
    --constraint structural_tag regex \
    --stream
```

Run all constraints, with reasoning models and streaming:

```bash
uv run structured_outputs_offline.py --reasoning --stream
```
