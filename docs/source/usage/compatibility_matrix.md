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
     - [APC](#apc)
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
   * - [APC](#apc)
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
     - [✗](https://github.com/vllm-project/vllm/pull/9057)
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
     - [✗](https://github.com/vllm-project/vllm/issues/7366)
     - ✗
     - ✗
     - [✗](https://github.com/vllm-project/vllm/issues/7366)
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
     - [✗](https://github.com/vllm-project/vllm/pull/8199)
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
     - [✗](https://github.com/vllm-project/vllm/issues/8198)
     - ✅
     -
     -
     -
     -
     -
   * - <abbr title="Multimodal Inputs">mm</abbr>
     - ✅
     -  [✗](https://github.com/vllm-project/vllm/pull/8348)
     -  [✗](https://github.com/vllm-project/vllm/pull/7199)
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
     - [✗](https://github.com/vllm-project/vllm/issues/6137)
     - ✅
     - ✗
     - ✅
     - ✅
     - ✅
     - ?
     - [✗](https://github.com/vllm-project/vllm/issues/7968)
     - ✅
     -
     -
     -
   * - beam-search
     - ✅
     - ✅
     - ✅
     - ✅
     - [✗](https://github.com/vllm-project/vllm/issues/6137)
     - ✅
     - ✗
     - ✅
     - ✅
     - ✅
     - ?
     - [✗](https://github.com/vllm-project/vllm/issues/7968>)
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
     - [✗](https://github.com/vllm-project/vllm/issues/9893)
     - ?
     - ✅
     - ✅
     -

```

### Feature x Hardware

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
     - [✗](https://github.com/vllm-project/vllm/issues/2729)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - [APC](#apc)
     - [✗](https://github.com/vllm-project/vllm/issues/3687)
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
     - [✗](https://github.com/vllm-project/vllm/pull/4830)
     - ✅
   * - <abbr title="Prompt Adapter">prmpt adptr</abbr>
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - [✗](https://github.com/vllm-project/vllm/issues/8475)
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
     - [✗](https://github.com/vllm-project/vllm/issues/8477)
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
