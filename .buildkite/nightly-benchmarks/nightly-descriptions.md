
# Nightly benchmark

The main goal of this benchmarking is two-fold:
- Performance clarity: Provide clarity on which one (vllm, tensorrt-llm, lmdeploy and tgi) leads in performance in what workload.
- Reproducible: one can run the exact same set of benchmarking commands inside the exact same docker by following reproducing instructions in [reproduce.md]().


## Docker images

We benchmark vllm, tensorrt-llm, lmdeploy and tgi using the following docker images:
- vllm/vllm-openai:v0.5.0.post1
- nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
- openmmlab/lmdeploy:v0.5.0
- ghcr.io/huggingface/text-generation-inference:2.1

<!-- Please check <a href="artifact://workspace/build/buildkite/vllm/performance-benchmark/.buildkite/nightly-benchmarks/nightly-pipeline.yaml">nightly-pipeline.yaml</a> artifact for more details on how we deploy the docker images. -->


## Hardware

One AWS node with 8x NVIDIA A100 GPUs.


## Workload description

We benchmark vllm, tensorrt-llm, lmdeploy and tgi using the following workload:

- Input length: randomly sample 500 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 500 prompts.
- Models: llama-3 8B, llama-3 70B, mixtral 8x7B.
- Average QPS (query per second): 4 for the small model (llama-3 8B) and 2 for other two models. For each QPS, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
- Evaluation metrics: Throughput (higher the better), TTFT (time to the first token, lower the better), ITL (inter-token latency, lower the better).

<!-- Check <a href="artifact://workspace/build/buildkite/vllm/performance-benchmark/.buildkite/nightly-benchmarks/tests/nightly-tests.json">nightly-tests.json</a> artifact for more details. -->

## Plots

In the following plots, the dot shows the mean and the error bar shows the standard error of the mean. Value 0 means that the corresponding benchmark crashed.

<img src="artifact://nightly_results.png" alt="Benchmarking results" height=250 >

## Results

{nightly_results_benchmarking_table}
