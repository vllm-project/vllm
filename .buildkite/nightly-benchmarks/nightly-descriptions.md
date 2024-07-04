
# Nightly benchmark

The main goal of this benchmarking is two-fold:
- Performance clarity: Provide clarity on which one (vllm, tensorrt-llm, lmdeploy and tgi) leads in performance in what workload.
- Reproducible: one can run the exact same set of benchmarking commands inside the exact same docker by following reproducing instructions in [reproduce.md]().


## Versions

We benchmark vllm, tensorrt-llm, lmdeploy and tgi using the following docker images:
- vllm/vllm-openai:v0.5.0.post1
- nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
- openmmlab/lmdeploy:v0.5.0
- ghcr.io/huggingface/text-generation-inference:2.1

Check <a href="artifact://workspace/build/buildkite/vllm/performance-benchmark/.buildkite/nightly-benchmarks/nightly-pipeline.yaml">nightly-pipeline.yaml</a> artifact for more details.


## Workload description

We benchmark vllm, tensorrt-llm, lmdeploy and tgi using the following workload:

- Input length: randomly sample 1000 prompts from ShareGPT dataset (with fixed random seed).
- Output length: the corresponding output length of these 1000 prompts.
- Batch size: dynamically determined by vllm and the arrival pattern of the requests.
- Average QPS (query per second): 4 for 8B model and 2 for larger models. For each QPS, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
- Models: llama-3 8B, llama-3 70B, mixtral 8x7B.
- Evaluation metrics: Throughput, TTFT (time to the first token, with mean and std), ITL (inter-token latency, with mean and std).

Check <a href="artifact://workspace/build/buildkite/vllm/performance-benchmark/.buildkite/nightly-benchmarks/tests/nightly-tests.json">nightly-tests.json</a> artifact for more details.


## Known crashes

- TGI v2.1 crashes when running mixtral model, see [TGI Issue #2122](https://github.com/huggingface/text-generation-inference/issues/2122)



## Plots

In the following plots, the error bar shows the standard error of the mean. Value 0 means that the corresponding benchmark crashed.

<img src="artifact://nightly_results.png" alt="Benchmarking results" height=250 >

## Results

{nightly_results_benchmarking_table}
