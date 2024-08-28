.. _benchmarks:

Benchmark suites of vLLM
========================



vLLM contains two sets of benchmarks:

+ **Performance benchmarks**: benchmark vLLM's performance under various workloads at a high frequency (when a pull request (PR for short) of vLLM is being merged). See `vLLM performance dashboard <https://perf.vllm.ai>`_ for the latest performance results.

+ **Nightly benchmarks**: compare vLLM's performance against alternatives (tgi, trt-llm, and lmdeploy) when there are major updates of vLLM (e.g., bumping up to a new version). The latest results are available in the `vLLM GitHub README <https://github.com/vllm-project/vllm/blob/main/README.md>`_.


Trigger a benchmark
-------------------

The performance benchmarks and nightly benchmarks can be triggered by submitting a PR to vLLM, and label the PR with `perf-benchmarks` and `nightly-benchmarks`.


.. note::

   Please refer to `vLLM performance benchmark descriptions <https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/performance-benchmarks-descriptions.md>`_ and `vLLM nightly benchmark descriptions <https://github.com/vllm-project/vllm/blob/main/.buildkite/nightly-benchmarks/nightly-descriptions.md>`_ for detailed descriptions on benchmark environment, workload and metrics.
