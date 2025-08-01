# Compatibility Matrix

The tables below show mutually exclusive features and the support on some hardware.

The symbols used have the following meanings:

- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD

!!! note
    Check the ❌ or 🟠 with links to see tracking issue for unsupported feature/hardware combination.

## Feature x Feature

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

| Feature | [CP][chunked-prefill] | [APC](automatic_prefix_caching.md) | [LoRA](lora.md) | [SD](spec_decode.md) | CUDA graph | [pooling](../models/pooling_models.md) | <abbr title="Encoder-Decoder Models">enc-dec</abbr> | <abbr title="Logprobs">logP</abbr> | <abbr title="Prompt Logprobs">prmpt logP</abbr> | <abbr title="Async Output Processing">async output</abbr> | multi-step | <abbr title="Multimodal Inputs">mm</abbr> | best-of | beam-search |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [CP][chunked-prefill] | ✅ | | | | | | | | | | | | | |
| [APC](automatic_prefix_caching.md) | ✅ | ✅ | | | | | | | | | | | | |
| [LoRA](lora.md) | ✅ | ✅ | ✅ | | | | | | | | | | | |
| [SD](spec_decode.md) | ✅ | ✅ | ❌ | ✅ | | | | | | | | | | |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | |
| [pooling](../models/pooling_models.md) | 🟠\* | 🟠\* | ✅ | ❌ | ✅ | ✅ | | | | | | | | |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr> | ❌ | [❌](gh-issue:7366) | ❌ | [❌](gh-issue:7366) | ✅ | ✅ | ✅ | | | | | | | |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | | | | | |
| <abbr title="Async Output Processing">async output</abbr> | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | | | | |
| multi-step | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | | | |
| [mm](multimodal_inputs.md) | ✅ | ✅ | [🟠](gh-pr:4194)<sup>^</sup> | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | |
| best-of | ✅ | ✅ | ✅ | [❌](gh-issue:6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](gh-issue:7968) | ✅ | ✅ | |
| beam-search | ✅ | ✅ | ✅ | [❌](gh-issue:6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](gh-issue:7968) | ❔ | ✅ | ✅ |

\* Chunked prefill and prefix caching are only applicable to last-token pooling.  
<sup>^</sup> LoRA is only applicable to the language backbone of multimodal models.

[](){ #feature-x-hardware }

## Feature x Hardware

| Feature                                                   | Volta               | Turing    | Ampere    | Ada    | Hopper     | CPU                | AMD    | TPU |
|-----------------------------------------------------------|---------------------|-----------|-----------|--------|------------|--------------------|--------|-----|
| [CP][chunked-prefill]                                     | [❌](gh-issue:2729) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅ |
| [APC](automatic_prefix_caching.md)                        | [❌](gh-issue:3687) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅ |
| [LoRA](lora.md)                                           | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ✅ |
| [SD](spec_decode.md)                                      | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ |
| CUDA graph                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ✅     | ❌ |
| [pooling](../models/pooling_models.md)                    | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr>       | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❌     | ❌ |
| [mm](multimodal_inputs.md)                                | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ |
| <abbr title="Logprobs">logP</abbr>                        | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ |
| <abbr title="Prompt Logprobs">prmpt logP</abbr>           | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ |
| <abbr title="Async Output Processing">async output</abbr> | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ❌     | ❌ |
| multi-step                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | [❌](gh-issue:8477) | ✅     | ❌ |
| best-of                                                   | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ |
| beam-search                                               | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     | ❌ |

!!! note
    Please refer to [Feature support through NxD Inference backend][feature-support-through-nxd-inference-backend] for features supported on AWS Neuron hardware
