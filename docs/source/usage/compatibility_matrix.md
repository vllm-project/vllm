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
   :stub-columns: 1
   :widths: auto

   * - Feature
     - Chunked-Prefill
     - APC
     - LoRA
     - Prompt Adapter
     - Speculative Decoding
     - CUDA graph
     - Pooling Models
     - Encoder-Decoder
     - Logprobs
     - Prompt Logprobs
     - Async Output Processing
     - multi-step
     - Multimodal Inputs
     - best-of
     - beam-search
     - Guided Decoding
   * - Chunked-Prefill
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
   * - APC
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
   * - LoRA
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
   * - Prompt Adapter
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
   * - Speculative Decoding
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
   * - Pooling Models
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
   * - Encoder-Decoder
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
   * - Logprobs
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
   * - Prompt Logprobs
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
   * - Async Output Processing
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
   * - Multimodal Inputs
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
   * - Guided Decoding
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
   * - Chunked-Prefill
     - `✗ <https://github.com/vllm-project/vllm/issues/2729>`__
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - APC
     - `✗ <https://github.com/vllm-project/vllm/issues/3687>`__
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - LoRA
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - `✗ <https://github.com/vllm-project/vllm/pull/4830>`__
     - ✅
   * - Prompt Adapter
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - `✗ <https://github.com/vllm-project/vllm/issues/8475>`__
     - ✅
   * - Speculative Decoding
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
   * - Pooling Models
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ?
   * - Encoder-Decoder
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✗
   * - Multimodal Inputs
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - Logprobs
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - Prompt Logprobs
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - Async Output Processing
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
   * - Guided Decoding
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
```
