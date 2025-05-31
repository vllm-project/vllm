---
title: Compatibility Matrix
---
[](){ #compatibility-matrix }

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

| Feature | [CP][chunked-prefill] | [APC][automatic-prefix-caching] | [LoRA][lora-adapter] | <abbr title="Prompt Adapter">prmpt adptr</abbr> | [SD][spec-decode] | CUDA graph | <abbr title="Pooling Models">pooling</abbr> | <abbr title="Encoder-Decoder Models">enc-dec</abbr> | <abbr title="Logprobs">logP</abbr> | <abbr title="Prompt Logprobs">prmpt logP</abbr> | <abbr title="Async Output Processing">async output</abbr> | multi-step | <abbr title="Multimodal Inputs">mm</abbr> | best-of | beam-search |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [CP][chunked-prefill] | ✅ | | | | | | | | | | | | | | |
| [APC][automatic-prefix-caching] | ✅ | ✅ | | | | | | | | | | | | | |
| [LoRA][lora-adapter] | ✅ | ✅ | ✅ | | | | | | | | | | | | |
| <abbr title="Prompt Adapter">prmpt adptr</abbr> | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | |
| [SD][spec-decode] | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | | | | | |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | |
| <abbr title="Pooling Models">pooling</abbr> | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | | | | | | | | |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr> | ❌ | [❌](gh-issue:7366) | ❌ | ❌ | [❌](gh-issue:7366) | ✅ | ✅ | ✅ | | | | | | | |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | | | | | |
| <abbr title="Async Output Processing">async output</abbr> | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | | | | |
| multi-step | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | | | |
| <abbr title="Multimodal Inputs">mm</abbr> | ✅ | [🟠](gh-pr:8348) | [🟠](gh-pr:4194) | ❔ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | |
| best-of | ✅ | ✅ | ✅ | ✅ | [❌](gh-issue:6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](gh-issue:7968) | ✅ | ✅ | |
| beam-search | ✅ | ✅ | ✅ | ✅ | [❌](gh-issue:6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](gh-issue:7968) | ❔ | ✅ | ✅ |

[](){ #feature-x-hardware }

## Feature x Hardware

| Feature                                                   | Volta              | Turing   | Ampere   | Ada   | Hopper   | CPU                | AMD   |
|-----------------------------------------------------------|--------------------|----------|----------|-------|----------|--------------------|-------|
| [CP][chunked-prefill]                                     | [❌](gh-issue:2729) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| [APC][automatic-prefix-caching]                           | [❌](gh-issue:3687) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| [LoRA][lora-adapter]                                      | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="Prompt Adapter">prmpt adptr</abbr>           | ✅                  | ✅        | ✅        | ✅     | ✅        | [❌](gh-issue:8475) | ✅     |
| [SD][spec-decode]                                         | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| CUDA graph                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ✅     |
| <abbr title="Pooling Models">pooling</abbr>               | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❔     |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr>       | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❌     |
| <abbr title="Multimodal Inputs">mm</abbr>                 | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="Logprobs">logP</abbr>                        | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="Prompt Logprobs">prmpt logP</abbr>           | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="Async Output Processing">async output</abbr> | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ❌     |
| multi-step                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | [❌](gh-issue:8477) | ✅     |
| best-of                                                   | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| beam-search                                               | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |

!!! note
    Please refer to [Feature support through NxD Inference backend][feature-support-through-nxd-inference-backend] for features supported on AWS Neuron hardware
