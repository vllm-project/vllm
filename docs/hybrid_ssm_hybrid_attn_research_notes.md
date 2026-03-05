#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

### Hybrid SSM + Hybrid Attention Research Prototype

This document summarizes the current research prototype for combining
sliding-window attention with a lightweight SSM history branch inside vLLM.

- **Core components**:
  - `HybridSSMAdapter` (`vllm/model_executor/layers/hybrid_ssm_adapter.py`):
    - Exposes a Mamba-style KV cache spec (`MambaSpec`) so it can obtain its
      own SSM state pool from the v1 KV manager.
    - Provides `forward_history_branch_prefill` and
      `forward_history_branch_decode`, which the hybrid backend calls to
      obtain an SSM contribution over the same flattened token set as
      Triton sliding-window attention.
  - `HybridAttentionLayer` (`vllm/model_executor/layers/hybrid_attn_layer.py`):
    - Thin wrapper around `Attention` that forces the
      `HybridAttentionBackend` and owns a `HybridSSMAdapter`.
  - `HybridAttentionBackend` / `HybridAttentionImpl`
    (`vllm/v1/attention/backends/hybrid_attn.py`):
    - Delegates the main KV attention to Triton (`TritonAttentionImpl`).
    - Invokes the layer’s `ssm_adapter` and fuses its output into the first
      `num_actual_tokens` positions of the attention output.

- **Prefix-sum SSM mode (no new kernels)**:
  - `HybridSSMAdapter` now has a small experimentation knob:
    - Environment variable `VLLM_HYBRID_SSM_MODE`:
      - `"disabled"` (default): history branch returns zeros (no-op).
      - `"prefix_sum"`: history branch computes a prefix sum along the token
        dimension for the active tokens:
        - Prefill: first `num_prefill_tokens`.
        - Decode: first `num_decode_tokens` or, if unavailable,
          `num_actual_tokens`.
  - The prefix-sum rule is:
    - Shape-agnostic beyond the leading token dimension (works for both
      `[T, H]` and `[T, num_heads, head_size]`).
    - Implemented in pure PyTorch, avoiding any new CUDA kernels while
      still providing a non-trivial, history-dependent signal for
      experimentation and tests.

- **Unit tests and synthetic evaluation**:
  - `tests/v1/attention/test_hybrid_attention.py`:
    - Validates that `HybridSSMAdapter` matches Mamba state shape/dtype.
    - Adds `test_hybrid_ssm_adapter_prefix_sum_mode`, which:
      - Enables `VLLM_HYBRID_SSM_MODE=prefix_sum`.
      - Checks that the decode history branch returns the correct prefix sum
        on a simple 1D sequence.
    - Keeps `test_hybrid_attention_impl_fuses_ssm_output`, which verifies that
      `HybridAttentionImpl` correctly fuses an adapter-provided tensor.
  - `tests/v1/attention/test_hybrid_synthetic_eval.py`:
    - Defines a small `_PrefixSumAdapter` and `_ToyLayer` to exercise
      `HybridAttentionImpl` end-to-end on a synthetic long-range task where
      the correct output is the prefix sum of per-token values.
    - Uses a stub Triton impl that simply copies `query` into `output` so
      the effect of the SSM fusion can be observed directly.

- **How to experiment**:
  - To enable the prefix-sum SSM mode globally for hybrid SSM adapters:
    - Set `VLLM_HYBRID_SSM_MODE=prefix_sum` in the environment before
      importing vLLM.
  - Run the v1 unit tests that cover the hybrid components, for example:
    - `tests/v1/attention/test_hybrid_attention.py`
    - `tests/v1/attention/test_hybrid_synthetic_eval.py`
  - For throughput / latency experiments, you can:
    - Use `vllm bench throughput` or long-document scripts in `benchmarks/`
      and compare runs with:
      - `VLLM_HYBRID_SSM_MODE=disabled` (no-op SSM) vs.
      - `VLLM_HYBRID_SSM_MODE=prefix_sum` (analytic SSM history).

- **Limitations and future directions**:
  - The current SSM branch is *not trained* and uses a hand-crafted
    prefix-sum rule purely for wiring and research validation.
  - No new CUDA kernels are introduced; performance characteristics are
    therefore not representative of a highly optimized SSM implementation.
  - Next steps when training becomes available:
    - Replace the analytic prefix-sum with a learnable Mamba-style SSM that
      reuses existing Mamba kernels and state layout.
    - Integrate hybrid attention into real long-context or streaming models
      (e.g., Step3Text variants or multimodal/video models) and train on
      tasks where long-range temporal memory is critical.


