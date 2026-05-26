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
- [Custom Proposer Backend (Experimental)](#custom-proposer-backend-experimental)

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
| Custom Proposer | Varies | Varies | Bring your own proposer class (experimental). |

For reproducible measurements in your environment, use
[`examples/features/speculative_decoding/spec_decode_offline.py`](../../../examples/features/speculative_decoding/spec_decode_offline.py)
or the [benchmark CLI guide](../../benchmarking/cli.md).

## Custom Proposer Backend (Experimental)

You can plug in your own custom proposer class for speculative decoding by setting the method to `custom_class` and providing the full module path to your class.
Your custom class must accept a `VllmConfig` upon instantiation and implement a `propose` method.

**Example configuration:**

- `speculative_config.method = "custom_class"`
- `speculative_config.model = "your_module.YourCustomProposerClass"`

## `--speculative-config` schema

Use `--speculative-config` to pass speculative decoding settings as a JSON
object on the CLI:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```

The same keys are accepted from Python via `LLM(..., speculative_config={...})`.
The tables below highlight common user-facing keys accepted in this JSON
object; they are not an exhaustive schema reference.
For more details, see the generated [engine arguments reference](../../configuration/engine_args.md)
and the API docs for [vllm.config.SpeculativeConfig][].

### Common keys

These keys are commonly used across speculative decoding setups, though some
only apply to model-based methods such as `draft_model`, `mtp`, `eagle3`, and
`dflash`.

| Key | Type | Default | Allowed values / meaning |
| --- | --- | --- | --- |
| `method` | `string` | `None` | Speculation method. Common values include `draft_model`, `ngram`, `suffix`, `mtp`, `eagle3`, and `dflash`. If omitted, vLLM infers the method from the provided configuration when possible. |
| `model` | `string` | `None` | Draft model, EAGLE head, or auxiliary model identifier. For `ngram`, `ngram_gpu`, `suffix`, and `mtp`, this can often be omitted. |
| `num_speculative_tokens` | `integer > 0` | `None` | Number of speculative tokens to propose per step. Required for methods that do not infer it from model metadata. |
| `draft_tensor_parallel_size` | `integer >= 1` | `None` | Tensor parallel size for the draft model. |
| `max_model_len` | `integer >= 1` | `None` | Maximum context length for the draft model. |
| `parallel_drafting` | `boolean` | `false` | Enable parallel draft token generation. Only compatible with EAGLE and draft-model methods. |
| `rejection_sample_method` | `string` | `strict` | `strict`, `probabilistic`, or `synthetic`. |
| `synthetic_acceptance_rate` | `float` | `None` | Average acceptance rate to target when `rejection_sample_method` is `synthetic`. Valid range is `[0, 1]`. |

!!! note
    Gemma 4 assistant checkpoints are handled as Gemma 4 MTP speculators, not
    as generic draft models. Use `"method": "mtp"` with the assistant
    checkpoint in `model`, as shown in the [MTP guide](mtp.md#gemma-4-assistant-models).

    If startup logs show `SpeculativeConfig(method='draft_model', ...)` for a
    Gemma 4 assistant checkpoint, the installed vLLM version does not include
    Gemma 4 MTP support for that path. Upgrade to a version that includes
    Gemma 4 MTP support instead of forcing the assistant checkpoint through
    generic draft-model speculative decoding.

### Method-specific keys

#### N-gram

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt_lookup_max` | `integer >= 1` | `5` if both lookup bounds are omitted; otherwise mirrors `prompt_lookup_min` when omitted | Maximum n-gram window size. |
| `prompt_lookup_min` | `integer >= 1` | `5` if both lookup bounds are omitted; otherwise mirrors `prompt_lookup_max` when omitted | Minimum n-gram window size. |

Example:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

#### Suffix decoding

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `suffix_decoding_max_tree_depth` | `integer` | `24` | Maximum combined prefix-match and speculation tree depth. |
| `suffix_decoding_max_cached_requests` | `integer` | `10000` | Maximum number of requests cached in the global suffix tree. Set `0` to disable the global cache. |
| `suffix_decoding_max_spec_factor` | `float` | `1.0` | Caps speculative length as a multiple of prefix-match length. |
| `suffix_decoding_min_token_prob` | `float` | `0.1` | Minimum estimated token probability required to speculate a token. |

Example:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "suffix",
    "num_speculative_tokens": 8,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_max_cached_requests": 10000,
    "suffix_decoding_max_spec_factor": 1.0,
    "suffix_decoding_min_token_prob": 0.1
  }'
```

### Notes

- `--speculative-config` expects a JSON object on the CLI. In YAML config
  files, use a nested mapping instead of an escaped JSON string.
- `tensor_parallel_size` is not a valid key in `speculative_config`. Use
  `draft_tensor_parallel_size` instead.
- Keys such as `temperature` and `top_p` are sampling parameters, not
  `--speculative-config` fields.
- Internal fields such as `target_model_config`, `draft_model_config`,
  `target_parallel_config`, `draft_parallel_config`, and `draft_load_config`
  are populated by vLLM and are not intended to be set by users.

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
