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

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: auto

   * - Feature
     - [CP](#chunked-prefill)
     - [APC](#apc)
     - [LoRA](#lora)
     - :abbr:`prmpt adptr (Prompt Adapter)`
     - [SD](#spec-decode)
     - CUDA graph
     - :abbr:`pooling (Pooling Models)`
     - :abbr:`enc-dec (Encoder-Decoder Models)`
     - :abbr:`logP (Logprobs)`
     - :abbr:`prmpt logP (Prompt Logprobs)`
     - :abbr:`async output (Async Output Processing)`
     - multi-step
     - :abbr:`mm (Multimodal Inputs)`
     - best-of
     - beam-search
     - :abbr:`guided dec (Guided Decoding)`
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
   * - [LoRA](#lora)
     - `✗ <https://github.com/vllm-project/vllm/pull/9057>`__
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
   * - :abbr:`prmpt adptr (Prompt Adapter)`
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
   * - [SD](#spec-decode)
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
   * - :abbr:`pooling (Pooling Models)`
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
   * - :abbr:`enc-dec (Encoder-Decoder Models)`
     - ✗
     - `✗ <https://github.com/vllm-project/vllm/issues/7366>`__
     - ✗
     - ✗
     - `✗ <https://github.com/vllm-project/vllm/issues/7366>`__
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
   * - :abbr:`logP (Logprobs)`
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
   * - :abbr:`prmpt logP (Prompt Logprobs)`
     - ✅
     - ✅
     - ✅
     - ✅
     - `✗ <https://github.com/vllm-project/vllm/pull/8199>`__
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
   * - :abbr:`async output (Async Output Processing)`
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
     - `✗ <https://github.com/vllm-project/vllm/issues/8198>`__
     - ✅
     -
     -
     -
     -
     -
   * - :abbr:`mm (Multimodal Inputs)`
     - ✅
     -  `✗ <https://github.com/vllm-project/vllm/pull/8348>`__
     -  `✗ <https://github.com/vllm-project/vllm/pull/7199>`__
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
     - `✗ <https://github.com/vllm-project/vllm/issues/6137>`__
     - ✅
     - ✗
     - ✅
     - ✅
     - ✅
     - ?
     - `✗ <https://github.com/vllm-project/vllm/issues/7968>`__
     - ✅
     -
     -
     -
   * - beam-search
     - ✅
     - ✅
     - ✅
     - ✅
     - `✗ <https://github.com/vllm-project/vllm/issues/6137>`__
     - ✅
     - ✗
     - ✅
     - ✅
     - ✅
     - ?
     - `✗ <https://github.com/vllm-project/vllm/issues/7968>`__
     - ?
     - ✅
     -
     -
   * - :abbr:`guided dec (Guided Decoding)`
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
     - `✗ <https://github.com/vllm-project/vllm/issues/9893>`__
     - ?
     - ✅
     - ✅
     -

```

### Feature x Hardware

```{eval-rst}
.. list-table::
   :header-rows: 1
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
     - `✗ <https://github.com/vllm-project/vllm/issues/2729>`__
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - [APC](#apc)
     - `✗ <https://github.com/vllm-project/vllm/issues/3687>`__
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - [LoRA](#lora)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - `✗ <https://github.com/vllm-project/vllm/pull/4830>`__
     - ✅
   * - :abbr:`prmpt adptr (Prompt Adapter)`
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - `✗ <https://github.com/vllm-project/vllm/issues/8475>`__
     - ✅
   * - [SD](#spec-decode)
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
   * - :abbr:`pooling (Pooling Models)`
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ?
   * - :abbr:`enc-dec (Encoder-Decoder Models)`
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✗
   * - :abbr:`mm (Multimodal Inputs)`
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - :abbr:`logP (Logprobs)`
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - :abbr:`prmpt logP (Prompt Logprobs)`
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - :abbr:`async output (Async Output Processing)`
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
     - `✗ <https://github.com/vllm-project/vllm/issues/8477>`__
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
   * - :abbr:`guided dec (Guided Decoding)`
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
```