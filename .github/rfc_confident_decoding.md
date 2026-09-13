# [RFC]: Confident Decoding — entropy-guided intermediate-layer logits selection

## Summary

We propose adding **Confident Decoding** to vLLM: a training-free decoding strategy that dynamically selects logits from near-final intermediate layers instead of always using the final layer. Layer selection is guided by prediction entropy (conservative backward search for the first entropy valley closest to the final layer).

This is the inference implementation accompanying our paper:

> **Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding**  
> Xuanming Zhang, Sining Zhoubian, Yuxuan Chen, et al.  
> arXiv: https://arxiv.org/abs/2606.21906

The paper shows consistent gains on challenging reasoning benchmarks (GPQA-Diamond, Omni-MATH, HLE) across dense and MoE LLMs, with zero memory overhead and <2% latency increase.

## Motivation

Standard autoregressive decoding always uses logits from the **final** transformer layer. Our work reveals a recurring **Guess-Refine-Perturb** dynamic: early layers form coarse guesses, intermediate layers refine reasoning-relevant semantics, and final layers can perturb refined predictions toward generic or alignment-preferred tokens.

Confident Decoding bypasses harmful final-layer perturbations by selecting the most confident near-final layer at each token position.

## Proposed API

Enabled via `--additional-config` (no recompilation required):

```bash
vllm serve /path/to/model \
  --additional-config '{
    "enable_multi_layer_entropy_selection": true,
    "select_method": "trough",
    "p": 1.0,
    "trough_max_backtrack_layers": 10
  }'
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_multi_layer_entropy_selection` | bool | `false` | Global switch |
| `select_method` | str | `"trough"` | Layer selection strategy (`trough`, `trough-m1`, `last-m8`, etc.) |
| `p` | float | `1.0` | Probability of using selected-layer logits; `0.0` = standard final-layer decoding |
| `trough_max_backtrack_layers` | int | `0` | Max layers to backtrack; `>0` uses this value; `<0` unlimited |
| `trough_backtrack_ratio` | float | `0.0` | Used when `trough_max_backtrack_layers == 0` |
| `trough_log_interval` | int | `0` | Periodic selection statistics logging; `0` disables |

## Technical Design

### CUDA graph compatibility

vLLM's CUDA graph capture covers the outer model wrapper. Mutating Python attributes or dynamic buffers inside compiled forward causes stale state on graph replay. Our design:

1. **Inner model** (compiled): collects raw candidate-layer hidden states only (similar to existing `aux_hidden_states`).
2. **Outer CausalLM wrapper** (eager): applies final norm to collected states and stores in shape-keyed buffers.
3. **Model runner** (eager): sets `_last_seq_len` and `_last_logits_indices` before `compute_logits`, keyed by CUDA graph batch shape.
4. **`compute_logits`**: batched `lm_head` over `[L*B, H]`, entropy computation, backward trough scan, gather selected logits.

### Selection algorithm (default `trough`)

1. Compute logits for all candidate layers in one batched `lm_head` call.
2. Compute per-token, per-layer prediction entropy.
3. Scan backward from the final candidate layer; continue while entropy decreases.
4. Stop at the first entropy increase (entropy valley closest to final layer).
5. Optionally mix with final-layer logits via probability `p`.

### Limitations (initial PR)

- Pipeline parallelism (`pp > 1`): disabled for correctness.
- Additional `lm_head` + entropy compute overhead per token (bounded by backtrack window).
- Some multimodal entry points (e.g. Qwen3-VL with deepstack) require separate integration.

## Proposed PR plan

We plan to land this in phases to keep reviews manageable:

| Phase | Scope |
| --- | --- |
| **PR 1** | `trough_utils.py`, v1 worker hooks, **Llama + Qwen3**, unit tests, `docs/features/confident_decoding.md` |
| **PR 2** | Gemma, Mixtral, MoE variants |
| **PR 3** | Multimodal entry points, DeepSeek, GPT-OSS |

Estimated core PR size: ~800 LOC (excluding model-family expansions).

## Validation plan

- **Unit tests**: `vectorized_entropy_select` layer selection logic; `compute_trough_layer_range`; `prepare_trough_layer_states` alignment.
- **Regression**: `p=0.0` produces identical outputs to standard final-layer decoding.
- **Deterministic check**: `select_method=last-mk` selects fixed layer index.
- **Integration**: small-model e2e generation smoke test.

## Questions for maintainers

1. Is `additional_config` the preferred configuration surface, or should this become a first-class `VllmConfig` field?
2. Should we rename `enable_multi_layer_entropy_selection` → `enable_confident_decoding` for clarity (with backward-compatible alias)?
3. Any concerns about the CUDA graph buffer design vs. alternative approaches?

## References

- Paper: https://arxiv.org/abs/2606.21906
- Implementation prototype (vLLM 0.19.1 fork): https://github.com/zhoubiansining/vllm (branch `confident-dev`)
