# Local Runtime Quickstart

The local runtime layer is meant to make common source-checkout workflows faster without changing the underlying vLLM engine and serving architecture.

## 1. Install Prerequisites

Install `uv` first using Astral's official instructions:

- <https://docs.astral.sh/uv/getting-started/installation/>

Then install this checkout:

```bash
./scripts/install.sh
vllm --help
```

The installer supports:

- `./scripts/install.sh`
- `./scripts/install.sh --user`
- `./scripts/install.sh --system`
- `./scripts/install.sh --venv .venv`
- `./scripts/install.sh --recreate`

## 2. Inspect the Runtime

Check which backend the local CLI will use and whether a model looks likely to fit:

```bash
vllm doctor
vllm doctor deepseek-r1:8b
vllm preflight qwen2.5:7b-instruct --profile low-memory
```

## 3. Pull a Model

```bash
vllm aliases
vllm pull deepseek-r1:8b
```

You can also use:

- exact Hugging Face repositories such as `meta-llama/Llama-3.1-8B-Instruct`
- local model paths

## 4. Run in the Shell

```bash
vllm run deepseek-r1:8b
vllm run llama3.2:3b-instruct --prompt "Summarize prefix caching."
vllm run qwen2.5:7b-instruct --profile throughput
```

## 5. Start a Local API Service

```bash
vllm serve deepseek-r1:8b
vllm ps
vllm logs deepseek-r1-8b
vllm stop deepseek-r1-8b
```

Use `--foreground` to keep the classic blocking server behavior:

```bash
vllm serve deepseek-r1:8b --foreground
```

## 6. Inspect Local State

```bash
vllm ls
vllm list
vllm inspect deepseek-r1:8b
```

## Notes

- This UX layer does not replace the production vLLM server architecture.
- The local commands sit on top of the same engine and OpenAI-compatible serving path.
- Backend selection is explicit and inspectable through `doctor`, `status`, and `preflight`.
