# vLLM V1

!!! announcement

    We have fully deprecated V0. Please read [RFC #18571](https://github.com/vllm-project/vllm/issues/18571) for more details.

V1 is now enabled by default for all supported use cases, and we will gradually enable it for every use case we plan to support. Please share any feedback on [GitHub](https://github.com/vllm-project/vllm) or in the [vLLM Slack](https://inviter.co/vllm-slack).

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

## Current Status

For each item, our progress towards V1 support falls into one of the following states:

- **🚀 Optimized**: Nearly fully optimized, with no further work currently planned.
- **🟢 Functional**: Fully operational, with ongoing optimizations.
- **🚧 WIP**: Under active development.
- **🟡 Planned**: Scheduled for future implementation (some may have open PRs/RFCs).
- **🟠 Delayed**: Temporarily dropped in V1 but planned to be re-introduced later.
- **🔴 Deprecated**: Not planned for V1 unless there is strong demand.

!!! note
    vLLM V1’s unified scheduler treats both prompt and output tokens the same
    way by using a simple dictionary (e.g., `{request_id: num_tokens}`) to dynamically
    allocate a fixed token budget per request, enabling features like chunked prefills,
    prefix caching, and speculative decoding without a strict separation between prefill
    and decode phases.

The V1 scheduler supports multiple scheduling policies, including First-Come,
First-Served (FCFS) and priority-based scheduling (where requests are processed
based on assigned priority, with FCFS as a tie-breaker), configurable via the
`--scheduling-policy` argument.

### Hardware

| Hardware   | Status                                        |
|------------|-----------------------------------------------|
| **NVIDIA** | <nobr>🚀</nobr>                               |
| **AMD**    | <nobr>🟢</nobr>                               |
| **INTEL GPU**    | <nobr>🟢</nobr>                               |
| **TPU**    | <nobr>🟢</nobr>                               |
| **CPU**    | <nobr>🟢 (x86\_64/aarch64) 🟡 (MacOS) </nobr> |

!!! note

    More hardware platforms may be supported via plugins, e.g.:

    - [vllm-ascend](https://github.com/vllm-project/vllm-ascend)
    - [vllm-spyre](https://github.com/vllm-project/vllm-spyre)
    - [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
    - [vllm-openvino](https://github.com/vllm-project/vllm-openvino)

    Please check their corresponding repositories for more details.

### Models

| Model Type                  | Status                                                                             |
|-----------------------------|------------------------------------------------------------------------------------|
| **Decoder-only Models**     | <nobr>🚀 Optimized</nobr>                                                          |
| **Encoder-Decoder Models**  | <nobr>🟢 Whisper only</nobr>                                                       |
| **Embedding Models**        | <nobr>🟢 Functional</nobr>                                                         |
| **Mamba Models**            | <nobr>🟢 (Mamba-2), 🟢 (Mamba-1)</nobr>                                            |
| **Multimodal Models**       | <nobr>🟢 Functional</nobr>                                                         |

See below for the status of models that are not yet supported or have more features planned in V1.

#### Embedding Models

The initial basic support is now functional.

Later, we will consider using [hidden states processor](https://github.com/vllm-project/vllm/issues/12249),
which is based on [global logits processor](https://github.com/vllm-project/vllm/pull/13360)
to enable simultaneous generation and embedding using the same engine instance in V1.

#### Mamba Models

Models using selective state-space mechanisms instead of standard transformer attention are supported.
Models that use Mamba-2 and Mamba-1 layers (e.g., `Mamba2ForCausalLM`, `MambaForCausalLM`,`FalconMambaForCausalLM`) are supported.

Hybrid models that combine Mamba-2 and Mamba-1 layers with standard attention layers are also supported (e.g., `BambaForCausalLM`,
`Zamba2ForCausalLM`, `NemotronHForCausalLM`, `FalconH1ForCausalLM` and `GraniteMoeHybridForCausalLM`, `JambaForCausalLM`, `Plamo2ForCausalLM`).

Hybrid models with mechanisms different to Mamba are also supported (e.g, `MiniMaxText01ForCausalLM`, `MiniMaxM1ForCausalLM`, `Lfm2ForCausalLM`).

Please note that prefix caching is not yet supported for any of the above models.

#### Encoder-Decoder Models

Whisper is supported. Other models requiring cross-attention between separate
encoder and decoder (e.g., `BartForConditionalGeneration`,
`MllamaForConditionalGeneration`) are not supported.

### Features

| Feature                                     | Status                                                                            |
|---------------------------------------------|-----------------------------------------------------------------------------------|
| **Prefix Caching**                          | <nobr>🚀 Optimized</nobr>                                                         |
| **Chunked Prefill**                         | <nobr>🚀 Optimized</nobr>                                                         |
| **LoRA**                                    | <nobr>🚀 Optimized</nobr>                                                         |
| **Logprobs Calculation**                    | <nobr>🟢 Functional</nobr>                                                        |
| **FP8 KV Cache**                            | <nobr>🟢 Functional on Hopper devices (<https://github.com/vllm-project/vllm/pull/15191>)</nobr>|
| **Spec Decode**                             | <nobr>🚀 Optimized</nobr>                                                         |
| **Prompt Logprobs with Prefix Caching**     | <nobr>🟡 Planned ([RFC #13414](https://github.com/vllm-project/vllm/issues/13414))</nobr>|
| **Structured Output Alternative Backends**  | <nobr>🟢 Functional</nobr>                                                        |
| **Request-level Structured Output Backend** | <nobr>🔴 Deprecated</nobr>                                                        |
| **best_of**                                 | <nobr>🔴 Deprecated ([RFC #13361](https://github.com/vllm-project/vllm/issues/13361))</nobr>|
| **Per-Request Logits Processors**           | <nobr>🔴 Deprecated ([RFC #13360](https://github.com/vllm-project/vllm/pull/13360))</nobr> |
| **GPU <> CPU KV Cache Swapping**            | <nobr>🔴 Deprecated</nobr>                                                        |

!!! note

    vLLM V1’s unified scheduler treats both prompt and output tokens the same
    way by using a simple dictionary (e.g., `{request_id: num_tokens}`) to dynamically
    allocate a fixed token budget per request, enabling features like chunked prefills,
    prefix caching, and speculative decoding without a strict separation between prefill
    and decode phases.

#### Semantic Changes to Logprobs

vLLM V1 supports logprobs and prompt logprobs. However, there are some important semantic
differences compared to V0:

##### Logprobs Calculation

By default, logprobs in V1 are now returned immediately once computed from the model’s raw output (i.e.
before applying any logits post-processing such as temperature scaling or penalty
adjustments). As a result, the returned logprobs do not reflect the final adjusted
probabilities used during sampling.

You can adjust this behavior by setting the `--logprobs-mode` flag.
Four modes are supported: `raw_logprobs` (default), `processed_logprobs`, `raw_logits`, `processed_logits`.
Raw means the values before applying any logit processors, like bad words.
Processed means the values after applying all processors, including temperature and top_k/top_p.

##### Prompt Logprobs with Prefix Caching

Logprobs are not cached. For a request requiring prompt logprobs, the engine will ignore the prefix cache and recompute the prefill of full prompt to generate the logprobs.

#### Deprecated Features

As part of the major architectural rework in vLLM V1, several legacy features have been deprecated.

##### Sampling features

- **best_of**: This feature has been deprecated due to limited usage. See details at [RFC #13361](https://github.com/vllm-project/vllm/issues/13361).
- **Per-Request Logits Processors**: In V0, users could pass custom
  processing functions to adjust logits on a per-request basis. In vLLM V1, this
  feature has been deprecated. Instead, we now support **global logits processors**
  which are set at startup time, see [RFC #17799](https://github.com/vllm-project/vllm/issues/17799).

##### KV Cache features

- **GPU <> CPU KV Cache Swapping**: with the new simplified core architecture, vLLM V1 no longer requires KV cache swapping
to handle request preemptions.

##### Structured Output features

- **Request-level Structured Output Backend**: Deprecated, alternative backends (outlines, guidance) with fallbacks is supported now.
