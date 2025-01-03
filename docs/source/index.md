# Welcome to vLLM!

```{figure} ./assets/logos/vllm-logo-text-light.png
:align: center
:alt: vLLM
:class: no-scaled-link
:width: 60%
```

```{raw} html
<p style="text-align:center">
<strong>Easy, fast, and cheap LLM serving for everyone
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
```

vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantization: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular HuggingFace models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism and pipeline parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs, Gaudi® accelerators and GPUs, PowerPC CPUs, TPU, and AWS Trainium and Inferentia Accelerators.
- Prefix caching support
- Multi-lora support

For more information, check out the following:

- [vLLM announcing blog post](https://vllm.ai) (intro to PagedAttention)
- [vLLM paper](https://arxiv.org/abs/2309.06180) (SOSP 2023)
- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference) by Cade Daniel et al.
- [vLLM Meetups](#meetups)

## Documentation

```{toctree}
:caption: Getting Started
:maxdepth: 1

getting_started/installation/index
getting_started/quickstart
getting_started/examples/examples_index
getting_started/troubleshooting
getting_started/faq
```

```{toctree}
:caption: Serving
:maxdepth: 1

serving/openai_compatible_server
serving/deploying_with_docker
serving/deploying_with_k8s
serving/deploying_with_helm
serving/deploying_with_nginx
serving/distributed_serving
serving/metrics
serving/integrations
serving/tensorizer
serving/runai_model_streamer
```

```{toctree}
:caption: Models
:maxdepth: 1

models/supported_models
models/generative_models
models/pooling_models
models/adding_model
models/enabling_multimodal_inputs
```

```{toctree}
:caption: Usage
:maxdepth: 1

usage/lora
usage/multimodal_inputs
usage/tool_calling
usage/structured_outputs
usage/spec_decode
usage/compatibility_matrix
usage/performance
usage/engine_args
usage/env_vars
usage/usage_stats
usage/disagg_prefill
```

```{toctree}
:caption: Quantization
:maxdepth: 1

quantization/supported_hardware
quantization/auto_awq
quantization/bnb
quantization/gguf
quantization/int8
quantization/fp8
quantization/fp8_e5m2_kvcache
quantization/fp8_e4m3_kvcache
```

```{toctree}
:caption: Automatic Prefix Caching
:maxdepth: 1

automatic_prefix_caching/apc
automatic_prefix_caching/details
```

```{toctree}
:caption: Performance
:maxdepth: 1

performance/benchmarks
```

% Community: User community resources

```{toctree}
:caption: Community
:maxdepth: 1

community/meetups
community/sponsors
```

% API Documentation: API reference aimed at vllm library usage

```{toctree}
:caption: API Documentation
:maxdepth: 2

dev/sampling_params
dev/pooling_params
dev/offline_inference/offline_index
dev/engine/engine_index
```

% Design: docs about vLLM internals

```{toctree}
:caption: Design
:maxdepth: 2

design/arch_overview
design/huggingface_integration
design/plugin_system
design/input_processing/model_inputs_index
design/kernel/paged_attention
design/multimodal/multimodal_index
design/multiprocessing
```

% For Developers: contributing to the vLLM project

```{toctree}
:caption: For Developers
:maxdepth: 2

contributing/overview
contributing/profiling/profiling_index
contributing/dockerfile/dockerfile
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
