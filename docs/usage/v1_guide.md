# vLLM V1

**We have started the process of deprecating V0. Please read [RFC #18571](https://github.com/vllm-project/vllm/issues/18571) for more details.**

V1 is now enabled by default for all supported use cases, and we will gradually enable it for every use case we plan to support. Please share any feedback on [GitHub](https://github.com/vllm-project/vllm) or in the [vLLM Slack](https://inviter.co/vllm-slack).

To disable V1, please set the environment variable as: `VLLM_USE_V1=0`, and send us a GitHub issue sharing the reason!

## Why vLLM V1?

vLLM V0 successfully supported a wide range of models and hardware, but as new features were developed independently, the system grew increasingly complex. This complexity made it harder to integrate new capabilities and introduced technical debt, revealing the need for a more streamlined and unified design.

Building on V0’s success, vLLM V1 retains the stable and proven components from V0
(such as the models, GPU kernels, and utilities). At the same time, it significantly
re-architects the core systems, covering the scheduler, KV cache manager, worker,
sampler, and API server, to provide a cohesive, maintainable framework that better
accommodates continued growth and innovation.

Specifically, V1 aims to:

- Provide a **simple, modular, and easy-to-hack codebase**.
- Ensure **high performance** with near-zero CPU overhead.
- **Combine key optimizations** into a unified architecture.
- Require **zero configs** by enabling features/optimizations by default.

We see significant performance improvements from upgrading to V1 core engine, in
particular for long context scenarios. Please see performance benchmark (To be
added).

For more details, check out the vLLM V1 blog post [vLLM V1: A Major
Upgrade to vLLM’s Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (published Jan 27, 2025).

This living user guide outlines a few known **important changes and limitations** introduced by vLLM V1. The team has been working actively to bring V1 as the default engine, therefore this guide will be updated constantly as more features get supported on vLLM V1.

### Supports Overview
#### Hardware

| Hardware | Status                                   |
|----------|------------------------------------------|
| **NVIDIA** | <nobr>🚀 Natively Supported</nobr>         |
| **AMD**    | <nobr>🚧 WIP</nobr>           |
| **TPU**    | <nobr>🚧 WIP</nobr>           |
| **CPU**    | <nobr>🚧 WIP</nobr>           |

#### Feature / Model

| Feature / Model | Status |
|-----------------|-----------------------------------------------------------------------------------|
| **Prefix Caching**                    | <nobr>🚀 Optimized</nobr>                                                        |
| **Chunked Prefill**                    | <nobr>🚀 Optimized</nobr>                                                        |
| **LoRA**                                    | <nobr>🚀 Optimized</nobr>                                                         |
| **Logprobs Calculation**                    | <nobr>🟢 Functional</nobr>                                                        |
| **Multimodal Models**                       | <nobr>🟢 Functional</nobr>                                                        |
| **FP8 KV Cache**                            | <nobr>🟢 Functional on Hopper devices ([PR #15191](https://github.com/vllm-project/vllm/pull/15191))</nobr>|
| **Spec Decode**                             | <nobr>🚧 WIP ([PR #13933](https://github.com/vllm-project/vllm/pull/13933))</nobr>|
| **Prompt Logprobs with Prefix Caching**     | <nobr>🟡 Planned ([RFC #13414](https://github.com/vllm-project/vllm/issues/13414))</nobr>|
| **Structured Output Alternative Backends**  | <nobr>🟢 Functional</nobr>                                                        |
| **Embedding Models**                        | <nobr>🚧 WIP ([PR #16188](https://github.com/vllm-project/vllm/pull/16188))</nobr> |
| **Mamba Models**                            | <nobr>🟡 Planned</nobr>                                                           |
| **Encoder-Decoder Models**                  | <nobr>🟠 Delayed</nobr>                                                           |
| **Request-level Structured Output Backend** | <nobr>🔴 Deprecated</nobr>                                                        |
| **best_of**                                 | <nobr>🔴 Deprecated ([RFC #13361](https://github.com/vllm-project/vllm/issues/13361))</nobr>|
| **Per-Request Logits Processors**           | <nobr>🔴 Deprecated ([RFC #13360](https://github.com/vllm-project/vllm/pull/13360))</nobr> |
| **GPU <> CPU KV Cache Swapping**            | <nobr>🔴 Deprecated</nobr>                                                        |

- **🚀 Optimized**: Nearly fully optimized, with no further work currently planned.
- **🟢 Functional**: Fully operational, with ongoing optimizations.  
- **🚧 WIP**: Under active development.  
- **🟡 Planned**: Scheduled for future implementation (some may have open PRs/RFCs).  
- **🟠 Delayed**: Temporarily dropped in V1 but planned to be re-introduced later.
- **🔴 Deprecated**: Not planned for V1 unless there is strong demand.

**Note**: vLLM V1’s unified scheduler treats both prompt and output tokens the same
way by using a simple dictionary (e.g., `{request_id: num_tokens}`) to dynamically
allocate a fixed token budget per request, enabling features like chunked prefills,
prefix caching, and speculative decoding without a strict separation between prefill
and decode phases.

### Semantic Changes and Deprecated Features

#### Logprobs

vLLM V1 supports logprobs and prompt logprobs. However, there are some important semantic
differences compared to V0:

**Logprobs Calculation**

Logprobs in V1 are now returned immediately once computed from the model’s raw output (i.e.
before applying any logits post-processing such as temperature scaling or penalty
adjustments). As a result, the returned logprobs do not reflect the final adjusted
probabilities used during sampling.

Support for logprobs with post-sampling adjustments is in progress and will be added in future updates.

**Prompt Logprobs with Prefix Caching**

Currently prompt logprobs are only supported when prefix caching is turned off via `--no-enable-prefix-caching`. In a future release, prompt logprobs will be compatible with prefix caching, but a recomputation will be triggered to recover the full prompt logprobs even upon a prefix cache hit. See details in [RFC #13414](https://github.com/vllm-project/vllm/issues/13414).

#### Deprecated Features

As part of the major architectural rework in vLLM V1, several legacy features have been deprecated.

**Sampling features**

- **best_of**: This feature has been deprecated due to limited usage. See details at [RFC #13361](https://github.com/vllm-project/vllm/issues/13361).
- **Per-Request Logits Processors**: In V0, users could pass custom
  processing functions to adjust logits on a per-request basis. In vLLM V1, this
  feature has been deprecated. Instead, the design is moving toward supporting **global logits
  processors**, a feature the team is actively working on for future releases. See details at [RFC #13360](https://github.com/vllm-project/vllm/pull/13360).

**KV Cache features**

- **GPU <> CPU KV Cache Swapping**: with the new simplified core architecture, vLLM V1 no longer requires KV cache swapping
to handle request preemptions.

**Structured Output features**

- **Request-level Structured Output Backend**: Deprecated, alternative backends
  (outlines, guidance) with fallbacks is WIP.
### Feature & Model Support in Progress

Although we have re-implemented and partially optimized many features and models from V0 in vLLM V1, optimization work is still ongoing for some, and others remain unsupported.

#### Features to Be Optimized

These features are already supported in vLLM V1, but their optimization is still
in progress.

- **Spec Decode**: Currently, only ngram-based spec decode is supported in V1. There
  will be follow-up work to support other types of spec decode (e.g., see [PR #13933](https://github.com/vllm-project/vllm/pull/13933)). We will prioritize the support for Eagle, MTP compared to draft model based spec decode.

- **Multimodal Models**: V1 is almost fully compatible with V0 except that interleaved modality input is not supported yet.
  See [here](https://github.com/orgs/vllm-project/projects/8) for the status of upcoming features and optimizations.

#### Models to Be Supported

vLLM V1 currently excludes model architectures with the `SupportsV0Only` protocol,
and the majority fall into the following categories. V1 support for these models will be added eventually.

**Embedding Models**  
The initial support will be provided by [PR #16188](https://github.com/vllm-project/vllm/pull/16188).

Later, we will consider using [hidden states processor](https://github.com/vllm-project/vllm/issues/12249), which is based on [global logits processor](https://github.com/vllm-project/vllm/pull/13360) to enable simultaneous generation and embedding using the same engine instance in V1.

**Mamba Models**  
Models using selective state-space mechanisms (instead of standard transformer attention)
are not yet supported (e.g., `MambaForCausalLM`, `JambaForCausalLM`).

**Encoder-Decoder Models**  
vLLM V1 is currently optimized for decoder-only transformers. Models requiring
  cross-attention between separate encoder and decoder are not yet supported (e.g., `BartForConditionalGeneration`, `MllamaForConditionalGeneration`).

For a complete list of supported models, see the [list of supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).
