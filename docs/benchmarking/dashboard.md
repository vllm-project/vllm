# Performance Dashboard

The performance dashboard is used to confirm whether new changes improve/degrade performance under various workloads.
It is updated by triggering benchmark runs on every commit with both the `perf-benchmarks` and `ready` labels, and when a PR is merged into vLLM.

The results are automatically published to the public [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm).

## Manually Trigger the benchmark

Use [vllm-ci-test-repo images](https://gallery.ecr.aws/q9t5s3a7/vllm-ci-test-repo) with vLLM benchmark suite.
For x86 CPU environment, please use the image with "-cpu" postfix. For AArch64 CPU environment, please use the image with "-arm64-cpu" postfix.

Here is an example for docker run command for CPU. For GPUs skip setting the `ON_CPU` env var.

```bash
export VLLM_COMMIT=1da94e673c257373280026f75ceb4effac80e892 # use full commit hash from the main branch
export HF_TOKEN=<valid Hugging Face token>
if [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]]; then
  IMG_SUFFIX="arm64-cpu"
else
  IMG_SUFFIX="cpu"
fi
docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_ARM64_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}-${IMG_SUFFIX}
```

Then, run below command inside the docker instance.

```bash
bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

When run, benchmark script generates results under **benchmark/results** folder, along with the benchmark_results.md and benchmark_results.json.

### Runtime environment variables

- `ON_CPU`: set the value to '1' on Intel® Xeon® and Arm® Neoverse™ Processors. Default value is 0.
- `SERVING_JSON`: JSON file to use for the serving tests. Default value is empty string (use default file).
- `LATENCY_JSON`: JSON file to use for the latency tests. Default value is empty string (use default file).
- `THROUGHPUT_JSON`: JSON file to use for the throughout tests. Default value is empty string (use default file).
- `REMOTE_HOST`: IP for the remote vLLM service to benchmark. Default value is empty string.
- `REMOTE_PORT`: Port for the remote vLLM service to benchmark. Default value is empty string.

For more results visualization, check the [visualizing the results](https://github.com/intel-ai-tce/vllm/blob/more_cpu_models/.buildkite/nightly-benchmarks/README.md#visualizing-the-results).

More information on the performance benchmarks and their parameters can be found in [Benchmark README](https://github.com/intel-ai-tce/vllm/blob/more_cpu_models/.buildkite/nightly-benchmarks/README.md) and [performance benchmark description](../../.buildkite/performance-benchmarks/performance-benchmarks-descriptions.md).

## Continuous Benchmarking

The continuous benchmarking provides automated performance monitoring for vLLM across different models and GPU devices. This helps track vLLM's performance characteristics over time and identify any performance regressions or improvements.

### How It Works

The continuous benchmarking is triggered via a [GitHub workflow CI](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) in the PyTorch infrastructure repository, which runs automatically every 4 hours. The workflow executes three types of performance tests:

- **Serving tests**: Measure request handling and API performance
- **Throughput tests**: Evaluate token generation rates
- **Latency tests**: Assess response time characteristics

### Benchmark Configuration

The benchmarking currently runs on a predefined set of models configured in the [vllm-benchmarks directory](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks). To add new models for benchmarking:

1. Navigate to the appropriate GPU directory in the benchmarks configuration
2. Add your model specifications to the corresponding configuration files
3. The new models will be included in the next scheduled benchmark run
