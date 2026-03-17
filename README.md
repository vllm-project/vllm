<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## What's New

This version keeps the existing vLLM engine and serving stack, but adds a more direct local-runtime workflow on top of it.

New local-runtime features:

- A repo-local bootstrap installer via `./scripts/install.sh`
- A simpler CLI centered on `vllm pull`, `vllm run`, and `vllm serve`
- A lightweight launcher so `vllm`, `vllm --help`, `vllm aliases`, `vllm pull`, and local service management feel immediate
- Built-in short model aliases such as `deepseek-r1:8b`
- Managed local background services with `vllm ps`, `vllm stop`, and `vllm logs`
- A tracked backlog for deferred parity work in `docs/cli/local_runtime_followups.md`

## Quick Start

If you want the fastest path to a local vLLM workflow from this repository:

```bash
./scripts/install.sh
vllm aliases
vllm pull deepseek-r1:8b
vllm run deepseek-r1:8b
```

If you want a local API server instead:

```bash
vllm serve deepseek-r1:8b
vllm ps
```

## Core Flows

### Local Runtime

Use the new local commands when you want an easier terminal-first experience:

- `vllm pull <model>` downloads a model using a short alias or exact Hugging Face repo
- `vllm run <model>` opens a local shell chat or runs a one-shot prompt
- `vllm serve <model>` starts a managed local service
- `vllm aliases` lists the built-in easy-model shortcuts
- `vllm ps`, `vllm stop`, and `vllm logs` manage those local services

### Built-in Easy Models

This version ships a larger set of built-in aliases so you can pull common models without remembering full Hugging Face repo IDs.

Examples:

- `deepseek-r1:1.5b`, `deepseek-r1:7b`, `deepseek-r1:8b`, `deepseek-r1:14b`, `deepseek-r1:32b`, `deepseek-r1:70b`
- `llama3.2:1b-instruct`, `llama3.2:3b-instruct`, `llama3.1:8b-instruct`, `llama3.1:70b-instruct`, `llama3.3:70b-instruct`
- `qwen2.5:0.5b-instruct`, `qwen2.5:1.5b-instruct`, `qwen2.5:3b-instruct`, `qwen2.5:7b-instruct`, `qwen2.5:14b-instruct`, `qwen2.5:32b-instruct`, `qwen2.5:72b-instruct`
- `qwen2.5-coder:1.5b-instruct`, `qwen2.5-coder:7b-instruct`, `qwen2.5-coder:32b-instruct`
- `mistral:7b-instruct`, `ministral:8b-instruct`, `mistral-nemo:12b-instruct`, `mixtral:8x7b-instruct`
- `gemma2:2b-it`, `gemma2:9b-it`, `gemma2:27b-it`
- `phi3.5:mini-instruct`, `phi3.5:moe-instruct`, `phi4`
- `smollm2:360m-instruct`, `smollm2:1.7b-instruct`

Use `vllm aliases` for the full installed list.

### Existing vLLM Paths

The existing advanced vLLM workflows still exist:

- `pip install vllm` for the standard package install path
- `vllm serve` in foreground mode for the existing server-first workflow
- `vllm chat`, `vllm complete`, `vllm bench`, and `vllm run-batch` for the existing advanced CLI surface
- Python library usage via `LLM`, `AsyncLLMEngine`, and the OpenAI-compatible server stack

## Documentation

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [CLI Guide](./docs/cli/README.md)
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
