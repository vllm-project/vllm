# Features

## Compatibility Matrix

The tables below show mutually exclusive features and the support on some hardware.

The symbols used have the following meanings:

- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD

!!! note
    Check the ❌ or 🟠 with links to see tracking issue for unsupported feature/hardware combination.

### Feature x Feature

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| Feature | [CP](../configuration/optimization.md#chunked-prefill) | [APC](automatic_prefix_caching.md) | [LoRA](lora.md) | [SD](spec_decode.md) | CUDA graph | [pooling](../models/pooling_models.md) | <abbr title="Encoder-Decoder Models">enc-dec</abbr> | <abbr title="Logprobs">logP</abbr> | <abbr title="Prompt Logprobs">prmpt logP</abbr> | <abbr title="Async Output Processing">async output</abbr> | multi-step | <abbr title="Multimodal Inputs">mm</abbr> | best-of | beam-search | [prompt-embeds](prompt_embeds.md) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [CP](../configuration/optimization.md#chunked-prefill) | ✅ | | | | | | | | | | | | | | |
| [APC](automatic_prefix_caching.md) | ✅ | ✅ | | | | | | | | | | | | | |
| [LoRA](lora.md) | ✅ | ✅ | ✅ | | | | | | | | | | | | |
| [SD](spec_decode.md) | ✅ | ✅ | ❌ | ✅ | | | | | | | | | | | |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | |
| [pooling](../models/pooling_models.md) | 🟠\* | 🟠\* | ✅ | ❌ | ✅ | ✅ | | | | | | | | | |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr> | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ✅ | ✅ | ✅ | | | | | | | | |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | | |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | | | | | | |
| <abbr title="Async Output Processing">async output</abbr> | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | | | | | |
| multi-step | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | | | | |
| [mm](multimodal_inputs.md) | ✅ | ✅ | [🟠](https://github.com/vllm-project/vllm/pull/4194)<sup>^</sup> | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | | |
| best-of | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ✅ | ✅ | | |
| beam-search | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ❔ | ✅ | ✅ | |
| [prompt-embeds](prompt_embeds.md) | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❔ | ❔ | ❌ | ❔ | ❔ | ✅ |

\* Chunked prefill and prefix caching are only applicable to last-token pooling.  
<sup>^</sup> LoRA is only applicable to the language backbone of multimodal models.

### Feature x Hardware

| Feature                                                   | Volta               | Turing    | Ampere    | Ada    | Hopper     | CPU                | AMD    | Intel GPU |
|-----------------------------------------------------------|---------------------|-----------|-----------|--------|------------|--------------------|--------| ------------|
| [CP](../configuration/optimization.md#chunked-prefill)                                     | [❌](https://github.com/vllm-project/vllm/issues/2729) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |
| [APC](automatic_prefix_caching.md)                        | [❌](https://github.com/vllm-project/vllm/issues/3687) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |
| [LoRA](lora.md)                                           | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |
| [SD](spec_decode.md)                                      | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ✅     | [🟠](https://github.com/vllm-project/vllm/issues/26963)       |
| CUDA graph                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ✅     | [❌](https://github.com/vllm-project/vllm/issues/26970)        |
| [pooling](../models/pooling_models.md)                    | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr>       | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❌     | ✅        |
| [mm](multimodal_inputs.md)                                | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | [🟠](https://github.com/vllm-project/vllm/issues/26965)       |
| [prompt-embeds](prompt_embeds.md)                         | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❔     | ✅       |
| <abbr title="Logprobs">logP</abbr>                        | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |
| <abbr title="Prompt Logprobs">prmpt logP</abbr>           | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |
| <abbr title="Async Output Processing">async output</abbr> | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ❌     | ✅        |
| multi-step                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | [❌](https://github.com/vllm-project/vllm/issues/8477) | ✅     | ✅        |
| best-of                                                   | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |
| beam-search                                               | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅        |

!!! note
    For information on feature support on Google TPU, please refer to the [TPU-Inference Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) documentation.
