
# Nightly benchmark

This benchmark aims to:
- Provide performance clarity: Provide clarity on which one (vllm, tensorrt-llm, lmdeploy and tgi) leads in performance in what workload.
- Be reproducible: one can run the exact same set of benchmarking commands inside the exact same docker by following reproducing instructions in [reproduce.md]().


## Setup

- Docker images
  - vllm/vllm-openai:v0.5.0.post1
  - nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
  - openmmlab/lmdeploy:v0.5.0
  - ghcr.io/huggingface/text-generation-inference:2.1
- Hardware
  - 8x Nvidia A100 GPUs
- Workload
  - Input length: randomly sample 500 prompts from ShareGPT dataset (with fixed random seed).
  - Output length: the corresponding output length of these 500 prompts.
  - Models: llama-3 8B, llama-3 70B, mixtral 8x7B.
    - We do not use llama 3.1 as it is incompatible with trt-llm r24.07. ([issue](https://github.com/NVIDIA/TensorRT-LLM/issues/2105)).
  - Average QPS (query per second): 2, 4, 8 and inf.
    - Queries are randomly sampled, and arrival patterns are determined via Poisson process, but all with fixed random seed.
  - Evaluation metrics: Throughput (higher the better), TTFT (time to the first token, lower the better), ITL (inter-token latency, lower the better).

## Plots

In the following plots, the dot shows the mean and the error bar shows the standard error of the mean. Value 0 means that the corresponding benchmark crashed.

<img src="artifact://nightly_results_sharegpt.png" alt="Benchmarking results" height=250 >

<img src="artifact://nightly_results_sonnet_2048_128.png" alt="Benchmarking results" height=250 >

<img src="artifact://nightly_results_sonnet_128_2048.png" alt="Benchmarking results" height=250 >

## Results

{nightly_results_benchmarking_table}


## Known issues

- TRT-LLM crashes with Llama 3.1 8B [issue](https://github.com/NVIDIA/TensorRT-LLM/issues/2105).