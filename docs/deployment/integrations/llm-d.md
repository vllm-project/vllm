# llm-d

[llm-d](https://llm-d.ai/) is a Kubernetes-native distributed inference framework for serving large language models at scale, with vLLM as its primary inference engine. llm-d coordinates a fleet of vLLM instances across a cluster so that performance holds up under real production traffic, achieving the fastest "time to state-of-the-art (SOTA) performance" for key OSS models across most hardware accelerators.

It is a [CNCF Sandbox project](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/) founded by Red Hat, Google Cloud, IBM Research, CoreWeave, and NVIDIA.

## What llm-d adds to vLLM

A single vLLM server is fast, but at scale the picture changes: across many replicas, cache locality breaks under round-robin load balancing, long prompts inflate time-to-first-token, and accelerators sit underused. llm-d adds the cluster-level layer that vLLM does not aim to provide on its own:

- **[Prefix-aware routing](https://llm-d.ai/docs/guides/precise-prefix-cache-aware).** Instead of round-robin, llm-d reads vLLM's KV-cache events and routes each request to the replica that already holds its prefix, reusing cache instead of recomputing it.
- **[Distributed KV-cache management](https://llm-d.ai/docs/guides#advanced-kv-cache-management).** A global index tracks which token blocks live on which replica, and [tiered offloading](https://llm-d.ai/docs/guides/tiered-prefix-cache) spills cache to CPU memory or local SSD, extending the working set beyond accelerator HBM.
- **[Prefill/decode disaggregation](https://llm-d.ai/docs/guides/pd-disaggregation).** Prompt processing and token generation run on separate vLLM workers, with KV-cache moved over the vLLM [NIXL connector](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/), lowering TTFT and steadying per-token latency on long prompts.
- **[Wide expert-parallelism](https://llm-d.ai/docs/guides/wide-expert-parallelism).** Serve large Mixture-of-Experts models such as DeepSeek-R1 and GPT-OSS across nodes with combined data and expert parallelism, for more KV-cache capacity and throughput.
- **SLO-aware [autoscaling](https://llm-d.ai/docs/guides/workload-autoscaling) and [flow control](https://llm-d.ai/docs/guides/flow-control).** Scale vLLM pools on real inference signals (queue depth, true demand) rather than raw GPU utilization, with multi-tenant fairness and priority dispatch.

These are composable. Most teams start by adding prefix-aware routing over an existing vLLM pool, then layer in the rest as specific bottlenecks appear.

## Performance

Representative benchmarked results across accelerators:

- **3x higher output throughput** and **2x faster TTFT** from prefix-aware routing vs round-robin (Llama 3.1 70B, AMD MI300X)
- **Up to 70% higher tokens/sec** from prefill/decode disaggregation (GPT-OSS, NVIDIA B200)
- **13.9x throughput** from hierarchical KV offloading at high concurrency vs GPU-only (NVIDIA H100)

See the [full list](https://github.com/llm-d/llm-d#performance-highlights) and reproducible benchmarks on [Prism](https://prism.llm-d.ai/).

## Get started

1. Deploy the [Optimized Baseline](https://llm-d.ai/docs/guides/optimized-baseline) with the [Quickstart](https://llm-d.ai/docs/getting-started/quickstart). It stands up an intelligent router over a vLLM pool on Kubernetes in a tested configuration.
2. Browse the [well-lit path guides](https://llm-d.ai/docs/guides), each a tested recipe for one of the capabilities above, and add the optimization that fits your workload.
3. Read the [Introduction](https://llm-d.ai/docs/getting-started) and [Architecture overview](https://llm-d.ai/docs/architecture) to see how the pieces wrap your vLLM deployment.

You can also deploy vLLM with llm-d via [KServe's LLMInferenceService](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-overview).

Questions and contributions are welcome on [GitHub](https://github.com/llm-d/llm-d) and [Slack](https://llm-d.ai/slack).
