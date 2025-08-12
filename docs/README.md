# Welcome to vLLM

<figure markdown="span">
  ![](./assets/logos/vllm-logo-text-light.png){ align="center" alt="vLLM" class="no-scaled-link" width="60%" }
</figure>

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

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
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
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs, Gaudi® accelerators and GPUs, IBM Power CPUs, TPU, and AWS Trainium and Inferentia Accelerators.
- Prefix caching support
- Multi-lora support

For more information, check out the following:

- [vLLM announcing blog post](https://vllm.ai) (intro to PagedAttention)
- [vLLM paper](https://arxiv.org/abs/2309.06180) (SOSP 2023)
- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference) by Cade Daniel et al.
- [vLLM Meetups][meetups]
