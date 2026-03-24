<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
High-performance LLM inference and serving, with a simpler local runtime
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

vLLM documentation, updates, and project resources are available at [vllm.ai](https://vllm.ai).

---

## Overview

vLLM is a high-performance inference and serving engine for large language models. This repository also adds a more direct local-runtime experience intended to make a source checkout feel closer to a local application install.

The updated workflow focuses on:

- explicit repo-local installation from this repository
- a lightweight `vllm` launcher for help, model management, and local runtime commands
- short built-in model aliases for common Hugging Face models
- direct shell-based execution with `vllm run`
- backend diagnostics and model preflight via `vllm doctor`, `vllm status`, and `vllm preflight`
- managed local services through `vllm serve`, `vllm ps`, `vllm stop`, and `vllm logs`

The underlying vLLM engine and server stack remain intact, including:

- [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- continuous batching
- OpenAI-compatible serving
- support for popular Hugging Face models
- quantization, scheduling, and distributed inference features from vLLM

## Key Capabilities

This repository is designed to support both the familiar vLLM workflows and a simpler local launcher flow.

- `./scripts/install.sh` installs the checkout with a standalone `vllm` command
- `vllm --help` and other local metadata commands start quickly without loading the full runtime stack first
- `vllm pull`, `vllm run`, and `vllm serve` provide the primary terminal workflow
- `vllm aliases` exposes built-in short names for commonly used models
- local runtime state is managed with `vllm ls`, `vllm inspect`, `vllm ps`, `vllm stop`, `vllm logs`, and `vllm rm`
- deferred parity work and planned improvements are tracked in [docs/cli/local_runtime_followups.md](./docs/cli/local_runtime_followups.md)

## Installation

The shortest installation path from this repository is:

```bash
uv --version
./scripts/install.sh
vllm --help
```

`./scripts/install.sh` expects `uv` to already be installed. If it is missing, the script exits with a link to Astral's official installation instructions rather than bootstrapping it through `curl | sh`.

The installer supports:

- system install
- user-local install
- explicit virtualenv install

Examples:

```bash
./scripts/install.sh
./scripts/install.sh --system
./scripts/install.sh --venv .venv
./scripts/install.sh --recreate
```

On Linux, the intended outcome is a usable `vllm` command installed directly into your path. On macOS, the installer currently uses the CPU source-build path; future Apple GPU support is planned separately.

## Quick Start

For a terminal-first local workflow:

```bash
./scripts/install.sh
vllm doctor
vllm aliases
vllm pull deepseek-r1:8b
vllm run deepseek-r1:8b
```

To start a local API service:

```bash
vllm serve deepseek-r1:8b
vllm ps
```

## Local Runtime Workflow

The local launcher is organized around a small set of common commands:

- `vllm pull <model>` downloads a model from an alias, Hugging Face repo, or local path
- `vllm run <model>` runs directly in your shell for chat or prompt-based generation
- `vllm serve <model>` starts a managed local service
- `vllm doctor`, `vllm status`, and `vllm preflight` expose backend selection, Apple/plugin fallback, and fit estimation
- `vllm aliases` lists the built-in short model names
- `vllm ls` or `vllm list` and `vllm inspect <model>` show local model metadata and resolution
- `vllm ps`, `vllm stop`, and `vllm logs` manage background services

This keeps the default local path simple while preserving the full vLLM runtime and API-serving capabilities.

## Backend Selection

The local UX layer keeps backend choice inspectable instead of implicit.

- Apple Silicon prefers a plugin-based Apple GPU path when available
- otherwise the CLI falls back cleanly to CPU and explains why
- NVIDIA, ROCm, XPU, and CPU paths continue to use the existing vLLM serving/runtime mechanisms
- TensorRT-LLM is surfaced as optional NVIDIA interoperability, not a replacement for native vLLM CUDA execution

Use:

```bash
vllm doctor
vllm doctor deepseek-r1:8b
vllm preflight qwen2.5:7b-instruct --profile low-memory
```

## Model Support

There are two layers of model support to keep in mind:

1. Easy aliases
2. Broader vLLM model support

### Built-in Aliases

The launcher includes a built-in alias catalog so common models can be referenced with short names instead of full Hugging Face repository IDs.

Current aliases include:

- `deepseek-r1:1.5b`, `deepseek-r1:7b`, `deepseek-r1:8b`, `deepseek-r1:14b`, `deepseek-r1:32b`, `deepseek-r1:70b`, `deepseek-v3`
- `llama3.2:1b-instruct`, `llama3.2:3b-instruct`, `llama3.1:8b-instruct`, `llama3.1:70b-instruct`, `llama3.3:70b-instruct`
- `qwen2.5:0.5b-instruct`, `qwen2.5:1.5b-instruct`, `qwen2.5:3b-instruct`, `qwen2.5:7b-instruct`, `qwen2.5:14b-instruct`, `qwen2.5:32b-instruct`, `qwen2.5:72b-instruct`
- `qwen2.5-coder:1.5b-instruct`, `qwen2.5-coder:7b-instruct`, `qwen2.5-coder:32b-instruct`
- `mistral:7b-instruct`, `ministral:8b-instruct`, `mistral-nemo:12b-instruct`, `mixtral:8x7b-instruct`
- `gemma2:2b-it`, `gemma2:9b-it`, `gemma2:27b-it`
- `phi3.5:mini-instruct`, `phi3.5:moe-instruct`, `phi4`
- `smollm2:360m-instruct`, `smollm2:1.7b-instruct`

Run `vllm aliases` to inspect the installed catalog.

### General vLLM Compatibility

The alias list is only a convenience layer. `vllm pull` and `vllm run` also accept:

- exact Hugging Face repositories such as `meta-llama/Llama-3.1-8B-Instruct`
- local filesystem paths

Broader compatibility still follows what vLLM itself supports. For the full model-compatibility matrix, see [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Existing vLLM Workflows

The local launcher flow is intended to improve the default user experience, not replace the existing vLLM interfaces.

You can still use:

- Python library APIs such as `LLM` and `AsyncLLMEngine`
- foreground `vllm serve`
- `vllm chat`, `vllm complete`, `vllm bench`, and `vllm run-batch`
- standard package installation and development workflows

## Documentation

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [CLI Guide](./docs/cli/README.md)
- [Local Runtime Quickstart](./docs/cli/local_runtime_quickstart.md)
- [Apple Silicon Quickstart](./docs/cli/apple_silicon.md)
- [Backend Selection](./docs/cli/backend_selection.md)
- [TensorRT-LLM Interoperability](./docs/cli/trtllm_interop.md)
- [Troubleshooting](./docs/cli/troubleshooting.md)
- [Local Runtime UX Design Note](./docs/design/local_runtime_ux.md)
- [Local Runtime Follow-ups](./docs/cli/local_runtime_followups.md)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
