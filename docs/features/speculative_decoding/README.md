# Speculative Decoding

This document shows how to use [Speculative Decoding](https://arxiv.org/pdf/2302.01318) with vLLM to reduce inter-token latency under medium-to-low QPS (query per second), memory-bound workloads.

To train your own draft models for optimized speculative decoding, see [vllm-project/speculators](speculators.md) for seamless training and integration with vLLM.

## vLLM Speculation Methods

vLLM supports a variety of methods of speculative decoding. Model-based methods such as EAGLE, MTP, draft models, PARD and MLP provide the best latency reduction, while simpler methods such as n-gram and suffix decoding provide modest speedups without increasing workload during peak traffic.

- [EAGLE](eagle.md)
- [Multi-Token Prediction (MTP)](mtp.md)
- [Draft Model](draft_model.md)
- [Parallel Draft Model (PARD)](parallel_draft_model.md)
- [Multi-Layer Perceptron](mlp.md)
- [N-Gram](n_gram.md)
- [Suffix Decoding](suffix.md)

## Method Selection at a Glance

Use this qualitative table as a starting point for method selection. Real gains
depend on your model family, traffic pattern, hardware, and sampling settings.

| Method | Low QPS (latency focused) | High QPS (throughput focused) | Notes |
| --- | --- | --- | --- |
| EAGLE | High gain | Medium to high gain | Strong general-purpose model-based method. |
| MTP | High gain | Medium to high gain | Best when the target model has native MTP support. |
| Draft model | High gain | Medium gain | Needs a separate draft model. |
| Parallel Draft Model | High gain | Medium to high gain | Low draft model latency. |
| MLP speculator | Medium to high gain | Medium gain | Good when compatible MLP speculators are available. |
| N-gram | Low to medium gain | Medium gain | Lightweight and easy to enable. |
| Suffix decoding | Low to medium gain | Medium gain | No extra draft model; dynamic speculation depth. |

For reproducible measurements in your environment, use
[`examples/offline_inference/spec_decode.py`](../../../examples/offline_inference/spec_decode.py)
or the [benchmark CLI guide](../../benchmarking/cli.md).

## `--speculative-config` Reference

The `--speculative-config` flag (CLI) or `speculative_config` dict (Python API)
accepts a JSON object whose keys map to the fields of
[`SpeculativeConfig`](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py).
This section documents every user-facing key, its type, default value, and
allowed values so that users do not have to read the source code.

### Usage

=== "CLI (online serving)"

    ```bash
    vllm serve <model> \
        --speculative-config '{"model": "draft-model-name", "num_speculative_tokens": 5}'
    ```

=== "Python (offline inference)"

    ```python
    llm = LLM(
        model="target-model-name",
        speculative_config={
            "model": "draft-model-name",
            "num_speculative_tokens": 5,
        },
    )
    ```

### General keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `method` | `str` | Auto-detected | The speculative method to use. Accepted values: `"ngram"`, `"ngram_gpu"`, `"medusa"`, `"mlp_speculator"`, `"draft_model"`, `"suffix"`, `"eagle"`, `"eagle3"`, `"mtp"`, `"extract_hidden_states"`. When `model` is provided and `method` is omitted, the method is inferred from the draft model config (e.g. an EAGLE checkpoint is detected automatically). If `model` is not provided, `method` must be set explicitly. |
| `model` | `str` | `None` | The Hugging Face model name or local path for the draft model, EAGLE head, or speculator weights. Not required for `ngram`, `ngram_gpu`, `suffix`, or `mtp` methods (set automatically). |
| `num_speculative_tokens` | `int` | From draft config, or required | The number of tokens to speculate per step. Must be **> 0**. For models with an `n_predict` field in their config (e.g. MTP, EAGLE), this defaults to that value. For suffix decoding it acts as the *maximum* and defaults to `suffix_decoding_max_tree_depth` (24). |
| `enforce_eager` | `bool` | `None` | Override the target model's `enforce_eager` setting for the draft model. When `None`, inherits from the target model config. |

### Draft model keys

These keys apply when using a model-based speculative method (`draft_model`,
`eagle`, `eagle3`, `medusa`, `mlp_speculator`, `mtp`).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `draft_tensor_parallel_size` | `int` | Same as target TP | Tensor-parallel degree for the draft model. Must be **1** or equal to the target model's tensor parallel size. For `mlp_speculator` models this defaults to 1. |
| `quantization` | `str` | `None` | Quantization method used for the draft model weights. Accepted values: `"awq"`, `"fp8"`, `"ptpc_fp8"`, `"fbgemm_fp8"`, `"fp_quant"`, `"modelopt"`, `"modelopt_fp4"`, `"modelopt_mxfp8"`, `"modelopt_mixed"`, `"gguf"`, `"gptq_marlin"`, `"awq_marlin"`, `"gptq"`, `"compressed-tensors"`, `"bitsandbytes"`, `"experts_int8"`, `"quark"`, `"moe_wna16"`, `"torchao"`, `"inc"`, `"mxfp4"`, `"mxfp8"`, `"petit_nvfp4"`, `"cpu_awq"`. When `None`, the model weights are assumed unquantized. For `mtp`, if unset, the target model's quantization is used automatically. |
| `max_model_len` | `int` | `None` | Maximum sequence length for the draft model. Must be **>= 1**. When `None`, defaults to `min(draft_model_max_len, target_model_max_len)`. Useful for testing the ability to skip speculation for some sequences. |
| `revision` | `str` | `None` | The specific model version to use for the draft model (branch name, tag, or commit id). |
| `code_revision` | `str` | `None` | The specific code revision for the draft model on Hugging Face Hub (branch name, tag, or commit id). |

### Advanced control keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `disable_padded_drafter_batch` | `bool` | `False` | Disable input padding for speculative decoding. When `True`, speculative input batches can contain sequences of different lengths, which may only be supported by certain attention backends. Currently only affects the EAGLE method. |
| `use_local_argmax_reduction` | `bool` | `False` | Use vocab-parallel local argmax instead of all-gathering full logits for draft token generation. Reduces communication from O(vocab_size) to O(2 * tp_size) per token. Only applies to greedy draft selection in non-tree speculation. |
| `speculative_token_tree` | `str` | Auto-generated chain | Specifies the tree structure for speculative token generation as a Python list-of-tuples string, e.g. `"[(0,), (0, 0), (0, 1)]"`. When `None`, a simple chain of `num_speculative_tokens` tokens is generated. The tree is sorted breadth-first internally. |
| `parallel_drafting` | `bool` | `False` | Enable parallel drafting, where all speculative tokens are generated in parallel rather than sequentially. Requires that the speculative model was trained to support parallel drafting (e.g. [PARD](https://arxiv.org/pdf/2504.18583) models). Only compatible with EAGLE and draft model methods. |
| `rejection_sample_method` | `str` | `"strict"` | Whether to use strict (target and draft sampled tokens must match exactly) or probabilistic rejection sampling. Accepted values: `"strict"`, `"probabilistic"`. Both respect the target model distribution, but probabilistic sampling yields a higher acceptance rate at the cost of more memory to cache draft logits. |

### N-gram proposer keys

These keys apply only when `method` is `"ngram"` or `"ngram_gpu"`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt_lookup_max` | `int` | `5` | Maximum n-gram window size for matching against the prompt. Must be **>= 1**. |
| `prompt_lookup_min` | `int` | `5` | Minimum n-gram window size for matching against the prompt. Must be **>= 1** and **<= `prompt_lookup_max`**. |

!!! note
    If neither `prompt_lookup_max` nor `prompt_lookup_min` is provided, both
    default to 5. If only one is provided, the other is set to the same value.

### Suffix decoding keys

These keys apply only when `method` is `"suffix"`. Requires the
[Arctic Inference](https://github.com/snowflakedb/ArcticInference) package
(`pip install arctic-inference`).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `suffix_decoding_max_tree_depth` | `int` | `24` | Maximum depth of the suffix decoding global and prompt trees. Limits the sum of the prefix match and speculation lengths. Must be **>= 1**. |
| `suffix_decoding_max_cached_requests` | `int` | `10000` | Maximum number of requests to cache in the global suffix tree. If exceeded, evicts in FIFO order. Set to `0` to disable the global suffix tree (prompt trees are still used). Must be **>= 0**. |
| `suffix_decoding_max_spec_factor` | `float` | `1.0` | Controls speculation lengths based on the prefix match length: `max_spec_tokens = max_spec_factor * prefix_match_length`. Must be **>= 0**. |
| `suffix_decoding_min_token_prob` | `float` | `0.1` | Minimum token probability (based on frequency counts) for a token to be speculated. Must be in **[0, 1]**. |

### Examples

Draft model with quantization:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config '{
        "model": "TechxGenus/Meta-Llama-3-8B-GPTQ",
        "method": "draft_model",
        "num_speculative_tokens": 3,
        "quantization": "gptq"
    }'
```

EAGLE with tree-structured speculation:

```python
speculative_config={
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "method": "eagle",
    "num_speculative_tokens": 3,
    "speculative_token_tree": "[(0,), (0, 0), (0, 1), (0, 0, 0)]",
}
```

N-gram with custom window:

```python
speculative_config={
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 8,
    "prompt_lookup_min": 3,
}
```

Suffix decoding with tuned parameters:

```python
speculative_config={
    "method": "suffix",
    "num_speculative_tokens": 32,
    "suffix_decoding_max_tree_depth": 16,
    "suffix_decoding_min_token_prob": 0.05,
}
```

## Lossless guarantees of Speculative Decoding

In vLLM, speculative decoding aims to enhance inference efficiency while maintaining accuracy. This section addresses the lossless guarantees of
speculative decoding, breaking down the guarantees into three key areas:

1. **Theoretical Losslessness**
   \- Speculative decoding sampling is theoretically lossless up to the precision limits of hardware numerics. Floating-point errors might
   cause slight variations in output distributions, as discussed
   in [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318)

2. **Algorithmic Losslessness**
   \- vLLM’s implementation of speculative decoding is algorithmically validated to be lossless. Key validation tests include:

    > - **Rejection Sampler Convergence**: Ensures that samples from vLLM’s rejection sampler align with the target
    >   distribution. [View Test Code](https://github.com/vllm-project/vllm/blob/47b65a550866c7ffbd076ecb74106714838ce7da/tests/samplers/test_rejection_sampler.py#L252)
    > - **Greedy Sampling Equality**: Confirms that greedy sampling with speculative decoding matches greedy sampling
    >   without it. This verifies that vLLM's speculative decoding framework, when integrated with the vLLM forward pass and the vLLM rejection sampler,
    >   provides a lossless guarantee. Almost all of the tests in [tests/spec_decode/e2e](/tests/v1/spec_decode).
    >   verify this property using [this assertion implementation](https://github.com/vllm-project/vllm/blob/b67ae00cdbbe1a58ffc8ff170f0c8d79044a684a/tests/spec_decode/e2e/conftest.py#L291)

3. **vLLM Logprob Stability**
   \- vLLM does not currently guarantee stable token log probabilities (logprobs). This can result in different outputs for the
   same request across runs. For more details, see the FAQ section
   titled *Can the output of a prompt vary across runs in vLLM?* in the [FAQs](../../usage/faq.md).

While vLLM strives to ensure losslessness in speculative decoding, variations in generated outputs with and without speculative decoding
can occur due to following factors:

- **Floating-Point Precision**: Differences in hardware numerical precision may lead to slight discrepancies in the output distribution.
- **Batch Size and Numerical Stability**: Changes in batch size may cause variations in logprobs and output probabilities, potentially
  due to non-deterministic behavior in batched operations or numerical instability.

For mitigation strategies, please refer to the FAQ entry *Can the output of a prompt vary across runs in vLLM?* in the [FAQs](../../usage/faq.md).

## Known Feature Incompatibility

1. Pipeline parallelism is not composible with speculative decoding as of `vllm<=0.15.0`
2. Speculative decoding with a draft models is not supported in `vllm<=0.10.0`

## Resources for vLLM contributors

- [[vLLM Office Hours #40] Intro to Speculators](https://www.youtube.com/watch?v=2ISAr_JVGLs)
- [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [What is Lookahead Scheduling in vLLM?](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [Information on batch expansion](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [Dynamic speculative decoding](https://github.com/vllm-project/vllm/issues/4565)
