
# Nightly benchmark

This benchmark aims to:

- Provide performance clarity: Provide clarity on which one (vllm, tensorrt-llm, lmdeploy and SGLang) leads in performance in what workload.
- Be reproducible: one can run the exact same set of benchmarking commands inside the exact same docker by following reproducing instructions.

Latest results: [results link](https://blog.vllm.ai/2024/09/05/perf-update.html), scroll to the end.

Latest reproduction guide: [github issue link](https://github.com/vllm-project/vllm/issues/8176)

## Setup

- Docker images:
    - vLLM: `vllm/vllm-openai:v0.6.2`
    - SGLang: `lmsysorg/sglang:v0.3.2-cu121`
    - LMDeploy: `openmmlab/lmdeploy:v0.6.1-cu12`
    - TensorRT-LLM: `nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3`
        - *NOTE: we use r24.07 as the current implementation only works for this version. We are going to bump this up.*
    - Check [nightly-pipeline.yaml](nightly-pipeline.yaml) for the concrete docker images, specs and commands we use for the benchmark.
- Hardware
    - 8x Nvidia A100 GPUs
- Workload:
    - Dataset
        - ShareGPT dataset
        - Prefill-heavy dataset (in average 462 input tokens, 16 tokens as output)
        - Decode-heavy dataset (in average 462 input tokens, 256 output tokens)
        - Check [nightly-tests.json](tests/nightly-tests.json) for the concrete configuration of datasets we use.
    - Models: llama-3 8B, llama-3 70B.
        - We do not use llama 3.1 as it is incompatible with trt-llm r24.07. ([issue](https://github.com/NVIDIA/TensorRT-LLM/issues/2105)).
    - Average QPS (query per second): 2, 4, 8, 16, 32 and inf.
        - Queries are randomly sampled, and arrival patterns are determined via Poisson process, but all with fixed random seed.
    - Evaluation metrics: Throughput (higher the better), TTFT (time to the first token, lower the better), ITL (inter-token latency, lower the better).

## Known issues

- TRT-LLM crashes with Llama 3.1 8B [issue](https://github.com/NVIDIA/TensorRT-LLM/issues/2105).
- TGI does not support `ignore-eos` flag.
