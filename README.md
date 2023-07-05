<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://vllm.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://github.com/vllm-project/vllm/discussions"><b>Discussions</b></a> |

</p>

---

*Latest News* 🔥
- [2023/06] Serving vLLM On any Cloud with SkyPilot. Check out a 1-click [example](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm) to start the vLLM demo, and the [blog post](https://blog.skypilot.co/serving-llm-24x-faster-on-the-cloud-with-vllm-and-skypilot/) for the story behind vLLM development on the clouds.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

---

vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Optimized CUDA kernels

vLLM is flexible and easy to use with:

- Seamless integration with popular HuggingFace models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server

vLLM seamlessly supports many Huggingface models, including the following architectures:

- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- LLaMA (`lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)

Install vLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

## Getting Started

Visit our [documentation](https://vllm.readthedocs.io/en/latest/) to get started.
- [Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Supported Models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)

## Performance

vLLM outperforms HuggingFace Transformers (HF) by up to 24x and Text Generation Inference (TGI) by up to 3.5x, in terms of throughput.
For details, check out our [blog post](https://vllm.ai).

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a10g_n1_dark.png">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a10g_n1_light.png" width="45%">
  </picture>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a100_n1_dark.png">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a100_n1_light.png" width="45%">
  </picture>
  <br>
  <em> Serving throughput when each request asks for 1 output completion. </em>
</p>

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a10g_n3_dark.png">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a10g_n3_light.png" width="45%">
  </picture>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a100_n3_dark.png">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/figures/perf_a100_n3_light.png" width="45%">
  </picture>  <br>
  <em> Serving throughput when each request asks for 3 output completions. </em>
</p>

## Contributing

We welcome and value any contributions and collaborations.
Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.
