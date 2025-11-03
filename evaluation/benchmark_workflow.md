# vLLM Benchmark Workflow

## Current status

Manual GitHub Action workflows have been added to the ROCm/vLLM repository to build and test Aiter and vLLM at specific commits. These workflows run text-based performance evaluations, comparing our results on MI308 against the standard baseline.

## Steps

1. Navigate to the [vLLM Benchmark Workflow GitHub Action page](https://github.com/ROCm/vllm/actions/workflows/vllm_benchmark.yaml).

   To trigger the workflow, click the "Run workflow" button at the top right corner of the Actions page. By default, it will build and test both Aiter and vLLM using their respective `dev/perf` branches; however, you can specify a particular commit or branch to test if desired. The default model is `deepseekr1_ptpc_fp8`, but we plan to support all models listed under the evaluation directory.

2. The workflow will perform the following steps:
   - Clone the Aiter and vLLM repositories at the specified commits or branches.
   - Download the `rocm/vllm-dev:nightly` base image.
   - Build a new Docker image that includes the current versions of Aiter and vLLM.
   - Run evaluation tests for the selected models.
   - Compare the test results against the baseline.
   - Upload the evaluation results as workflow artifacts for further analysis and inspection.
