# Nika

[Nika](https://github.com/supernovae-st/nika) is an open-source (AGPL, Rust) workflow engine for AI: repeatable work lives in plain `.nika.yaml` DAG files that are statically checked before execution (schema, permits, cost) and leave a tamper-evident trace after (`nika trace verify`).

By pointing Nika at a vLLM server, the `infer` tasks in a workflow run against your own models instead of a cloud API. This is useful for:

- Running fully local, auditable AI pipelines (nothing leaves the machine unless a workflow declares it)
- Mixing local vLLM inference with shell steps and tool calls in one reviewable file
- CI pipelines where workflows are checked statically first and budget-capped at run time

## How It Works

vLLM serves an OpenAI-compatible API. Nika ships `vllm` as a first-class local provider (one of five local providers alongside Ollama and llama.cpp): tasks address models as `vllm/<model>` and the engine talks to the local server — no API key involved.

## Installation

Install Nika by following the [installation guide](https://github.com/supernovae-st/nika#install) — Homebrew, `cargo binstall`, or Nix:

```bash
brew install supernovae-st/tap/nika
```

## Starting the vLLM Server

```bash
vllm serve Qwen/Qwen3-4B --port 8000
```

## Using Nika with vLLM

Write a workflow whose `infer` task targets the vLLM provider:

```yaml
# yaml-language-server: $schema=https://nika.sh/spec/v1/workflow.schema.json
nika: v1
workflow: summarize-log
tasks:
  - id: summarize
    infer:
      prompt: "Summarize the following build log: ${{ vars.log }}"
      model: "vllm/Qwen/Qwen3-4B"
      max_tokens: 400
```

Check it statically, then run it:

```bash
nika check summarize-log.nika.yaml   # schema · DAG · permits · cost — exit 0 = clean
nika run summarize-log.nika.yaml --var log="$(cat build.log)"
nika trace verify                     # hash-chained receipt of what actually ran
```

`nika doctor` confirms the vLLM lane (`local · 5 provider(s) … vllm — no key · needs a running server`), and `nika catalog` lists the provider with its model form.
