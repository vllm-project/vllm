(compatibility-matrix)=

# Compatibility Matrix

The tables below show mutually exclusive features and the support on some hardware.

```{note}
Check the '✗' with links to see tracking issue for unsupported feature/hardware combination.
```

## Feature x Feature

```{raw} html
<style>
  /* Make smaller to try to improve readability  */
  td {
    font-size: 0.8rem;
    text-align: center;
  }

  th {
    text-align: center;
    font-size: 0.8rem;
  }
</style>
```

```{list-table}
   :header-rows: 1
   :stub-columns: 1
   :widths: auto

   * - Feature
     - [CP](#chunked-prefill)
     - [APC](#automatic-prefix-caching)
     - [LoRA](#lora-adapter)
     - <abbr title="Prompt Adapter">prmpt adptr</abbr>
     - [SD](#spec_decode)
     - CUDA graph
     - <abbr title="Pooling Models">pooling</abbr>
     - <abbr title="Encoder-Decoder Models">enc-dec</abbr>
     - <abbr title="Logprobs">logP</abbr>
     - <abbr title="Prompt Logprobs">prmpt logP</abbr>
     - <abbr title="Async Output Processing">async output</abbr>
     - multi-step
     - <abbr title="Multimodal Inputs">mm</abbr>
     - best-of
     - beam-search
     - <abbr title="Guided Decoding">guided dec</abbr>
   * - [CP](#chunked-prefill)
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - [APC](#automatic-prefix-caching)
     - ✅
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - [LoRA](#lora-adapter)
     - [✗](gh-pr:9057)
     - ✅
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - <abbr title="Prompt Adapter">prmpt adptr</abbr>
     - ✅
     - ✅
     - ✅
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - [SD](#spec_decode)
     - ✅
     - ✅
     - ✗
     - ✅
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - CUDA graph
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - <abbr title="Pooling Models">pooling</abbr>
     - ✗
     - ✗
     - ✗
     - ✗
     - ✗
     - ✗
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - <abbr title="Encoder-Decoder Models">enc-dec</abbr>
     - ✗
     - [✗](gh-issue:7366)
     - ✗
     - ✗
     - [✗](gh-issue:7366)
     - ✅
     - ✅
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - <abbr title="Logprobs">logP</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✗
     - ✅
     -
     -
     -
     -
     -
     -
     -
     -
   * - <abbr title="Prompt Logprobs">prmpt logP</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - [✗](gh-pr:8199)
     - ✅
     - ✗
     - ✅
     - ✅
     -
     -
     -
     -
     -
     -
     -
   * - <abbr title="Async Output Processing">async output</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✗
     - ✅
     - ✗
     - ✗
     - ✅
     - ✅
     -
     -
     -
     -
     -
     -
   * - multi-step
     - ✗
     - ✅
     - ✗
     - ✅
     - ✗
     - ✅
     - ✗
     - ✗
     - ✅
     - [✗](gh-issue:8198)
     - ✅
     -
     -
     -
     -
     -
   * - <abbr title="Multimodal Inputs">mm</abbr>
     - ✅
     -  [✗](gh-pr:8348)
     -  [✗](gh-pr:7199)
     - ?
     - ?
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ?
     -
     -
     -
     -
   * - best-of
     - ✅
     - ✅
     - ✅
     - ✅
     - [✗](gh-issue:6137)
     - ✅
     - ✗
     - ✅
     - ✅
     - ✅
     - ?
     - [✗](gh-issue:7968)
     - ✅
     -
     -
     -
   * - beam-search
     - ✅
     - ✅
     - ✅
     - ✅
     - [✗](gh-issue:6137)
     - ✅
     - ✗
     - ✅
     - ✅
     - ✅
     - ?
     - [✗](gh-issue:7968>)
     - ?
     - ✅
     -
     -
   * - <abbr title="Guided Decoding">guided dec</abbr>
     - ✅
     - ✅
     - ?
     - ?
     - ✅
     - ✅
     - ✗
     - ?
     - ✅
     - ✅
     - ✅
     - [✗](gh-issue:9893)
     - ?
     - ✅
     - ✅
     -

```

(feature-x-hardware)=

## Feature x Hardware

```{list-table}
   :header-rows: 1
   :stub-columns: 1
   :widths: auto

   * - Feature
     - Volta
     - Turing
     - Ampere
     - Ada
     - Hopper
     - CPU
     - AMD
   * - [CP](#chunked-prefill)
     - [✗](gh-issue:2729)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - [APC](#automatic-prefix-caching)
     - [✗](gh-issue:3687)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - [LoRA](#lora-adapter)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - <abbr title="Prompt Adapter">prmpt adptr</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - [✗](gh-issue:8475)
     - ✅
   * - [SD](#spec_decode)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - CUDA graph
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✗
     - ✅
   * - <abbr title="Pooling Models">pooling</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ?
   * - <abbr title="Encoder-Decoder Models">enc-dec</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✗
   * - <abbr title="Multimodal Inputs">mm</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - <abbr title="Logprobs">logP</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - <abbr title="Prompt Logprobs">prmpt logP</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - <abbr title="Async Output Processing">async output</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✗
     - ✗
   * - multi-step
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - [✗](gh-issue:8477)
     - ✅
   * - best-of
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - beam-search
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - <abbr title="Guided Decoding">guided dec</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
```
