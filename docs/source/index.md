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
:caption: Models
:maxdepth: 1

models/generative_models
models/pooling_models
models/supported_models
models/extensions/index
```

```{toctree}
:caption: Features
:maxdepth: 1

features/quantization/index
features/lora
features/tool_calling
features/structured_outputs
features/automatic_prefix_caching
features/disagg_prefill
features/spec_decode
features/compatibility_matrix
```

```{toctree}
:caption: Inference and Serving
:maxdepth: 1

serving/offline_inference
serving/openai_compatible_server
serving/multimodal_inputs
serving/distributed_serving
serving/metrics
serving/engine_args
serving/env_vars
serving/usage_stats
serving/integrations/index
```

```{toctree}
:caption: Deployment
:maxdepth: 1

deployment/docker
deployment/k8s
deployment/nginx
deployment/frameworks/index
deployment/integrations/index
```

```{toctree}
:caption: Performance
:maxdepth: 1

performance/optimization
performance/benchmarks
```

% Community: User community resources

```{toctree}
:caption: Community
:maxdepth: 1

community/meetups
community/sponsors
```

```{toctree}
:caption: API Reference
:maxdepth: 2

api/offline_inference/index
api/engine/index
api/multimodal/index
api/params
```

% Design Documents: Details about vLLM internals

```{toctree}
:caption: Design Documents
:maxdepth: 2

design/arch_overview
design/huggingface_integration
design/plugin_system
design/kernel/paged_attention
design/input_processing/model_inputs_index
design/automatic_prefix_caching
design/multiprocessing
```

% Developer Guide: How to contribute to the vLLM project

```{toctree}
:caption: Developer Guide
:maxdepth: 2

contributing/overview
contributing/profiling/profiling_index
contributing/dockerfile/dockerfile
contributing/model/index
contributing/vulnerability_management
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
