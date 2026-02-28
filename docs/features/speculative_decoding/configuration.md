# Configuration Reference

This page documents all accepted keys for the `--speculative-config` CLI argument (or the `speculative_config` parameter in the Python API).

## Usage

=== "CLI (Online Serving)"

    ```bash
    vllm serve <model> --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
    ```

=== "Python (Offline Inference)"

    ```python
    from vllm import LLM

    llm = LLM(
        model="<model>",
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
        },
    )
    ```

## General Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | `str` | `None` | Speculative method to use. See [Supported Methods](#supported-methods) below. If `model` is provided, the method is auto-detected when possible. If `model` is not provided, this is required. |
| `num_speculative_tokens` | `int` | `None` | Number of speculative tokens to generate per step. Required unless the draft model config provides a default. Must be > 0. |
| `model` | `str` | `None` | Name or path of the draft model, EAGLE head, or additional weights. |
| `enforce_eager` | `bool` | `None` | Override the target model's `enforce_eager` setting for the draft model. |

## Draft Model Parameters

These parameters apply when using model-based methods (`draft_model`, `eagle`, `eagle3`, `mlp_speculator`, or MTP variants).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `draft_tensor_parallel_size` | `int` | `None` | Tensor parallelism degree for the draft model. Must be 1 or equal to the target model's tensor parallel size. |
| `quantization` | `str` | `None` | Quantization method used for draft model weights (e.g., `"awq"`, `"gptq"`, `"fp8"`). Only applies to draft model-based methods. |
| `max_model_len` | `int` | `None` | Maximum model length of the draft model. Used to determine when to skip speculation for long sequences. |
| `revision` | `str` | `None` | Specific model version for the draft model (branch name, tag, or commit id). |
| `code_revision` | `str` | `None` | Specific code revision for the draft model on Hugging Face Hub. |
| `draft_load_config` | `dict` | `None` | Load configuration for the draft model. If not specified, uses the target model's load config. |

## N-Gram Parameters

These parameters apply when `method` is set to `"ngram"`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `prompt_lookup_max` | `int` | `None` | Maximum n-gram token window size. If unset, defaults to the value of `prompt_lookup_min` if provided, otherwise 5. |
| `prompt_lookup_min` | `int` | `None` | Minimum n-gram token window size. If unset, defaults to the value of `prompt_lookup_max` if provided, otherwise 5. |

## Suffix Decoding Parameters

These parameters apply when `method` is set to `"suffix"`. Requires [Arctic Inference](https://github.com/snowflakedb/ArcticInference) (`pip install arctic-inference`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `suffix_decoding_max_tree_depth` | `int` | `24` | Maximum depth of the suffix decoding global and prompt trees. Limits the sum of prefix match and speculation lengths. |
| `suffix_decoding_max_cached_requests` | `int` | `10000` | Maximum number of requests cached in the global suffix tree. Exceeding this triggers FIFO eviction. Set to 0 to disable the global suffix tree (prompt trees are still used). |
| `suffix_decoding_max_spec_factor` | `float` | `1.0` | Controls speculation length relative to prefix match: `max_spec_tokens = max_spec_factor * prefix_match_length`. |
| `suffix_decoding_min_token_prob` | `float` | `0.1` | Minimum estimated token probability (based on frequency counts) required to speculate a token. |

## Advanced Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `disable_padded_drafter_batch` | `bool` | `False` | Disable input padding for speculative decoding. When `True`, speculative batches can contain sequences of different lengths. Currently only affects the EAGLE method. |
| `use_local_argmax_reduction` | `bool` | `False` | Use vocab-parallel local argmax instead of all-gathering full logits for draft token generation. Reduces communication from O(vocab_size) to O(2 * tp_size) per token. Only applies to greedy draft selection in non-tree speculation. |
| `speculative_token_tree` | `str` | `None` | Tree structure specification for speculative token generation. |
| `parallel_drafting` | `bool` | `False` | Generate all speculative tokens in parallel rather than sequentially. Requires the speculative model to be trained for parallel drafting. Only compatible with EAGLE and draft model methods. |

## Supported Methods

The `method` parameter accepts the following values:

| Method | Description | Requires `model`? |
|--------|-------------|-------------------|
| `"draft_model"` | Uses a smaller model to generate draft tokens. See [Draft Models](draft_model.md). | Yes |
| `"eagle"` | Uses an EAGLE draft model. See [EAGLE](eagle.md). | Yes |
| `"eagle3"` | Uses an Eagle3 draft model. See [EAGLE](eagle.md). | Yes |
| `"ngram"` | Matches n-grams in the prompt to generate proposals. See [N-Gram](n_gram.md). | No |
| `"mlp_speculator"` | Uses an MLP-based draft model. See [MLP](mlp.md). | Yes |
| `"suffix"` | Uses suffix tree matching against prompt and past generations. See [Suffix Decoding](suffix.md). | No |
| `"medusa"` | Uses Medusa-style multi-head prediction. | Yes |
| `"deepseek_mtp"`, `"qwen3_5_mtp"`, ... | Model-specific MTP (Multi-Token Prediction) methods. Auto-detected from model architecture. | No |
