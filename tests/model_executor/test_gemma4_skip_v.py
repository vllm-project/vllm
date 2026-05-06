# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Gemma4 k_eq_v optimization.

The optimization passes ``v_head_size=0`` to ``QKVParallelLinear`` on
k_eq_v global-attention layers so the packed weight matrix excludes the V
slot entirely.  V is then derived in ``forward()`` by running the pre-norm
K tensor through ``v_norm`` (which has no learnable scale) — mathematically
identical to the old approach of loading K weights into the V projection
slot.

Tests:
  - ``v_head_size=0`` drops the V slot from the packed weight without
    changing any other behaviour of ``QKVParallelLinear``.
  - Deriving V via ``v_norm(k_pre)`` is bit-exact equivalent to the old
    duplicated-weight approach.
"""

import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear


HIDDEN = 64
NUM_Q = 4
NUM_KV = 2
HEAD_DIM = 16


def _make_qkv(*, v_head_size: int | None) -> QKVParallelLinear:
    return QKVParallelLinear(
        hidden_size=HIDDEN,
        head_size=HEAD_DIM,
        total_num_heads=NUM_Q,
        total_num_kv_heads=NUM_KV,
        bias=False,
        quant_config=None,
        prefix="test.qkv_proj",
        disable_tp=True,
        v_head_size=v_head_size,
    )


class TestQKVParallelLinearVHeadSizeZero:
    """``v_head_size=0`` must drop V from the packed weight/output."""

    def test_weight_shape_without_v(self, dist_init):
        proj = _make_qkv(v_head_size=0)
        q_size = NUM_Q * HEAD_DIM
        kv_size = NUM_KV * HEAD_DIM
        assert proj.weight.shape == (q_size + kv_size, HIDDEN)

    def test_weight_shape_with_v(self, dist_init):
        proj = _make_qkv(v_head_size=None)
        q_size = NUM_Q * HEAD_DIM
        kv_size = NUM_KV * HEAD_DIM
        assert proj.weight.shape == (q_size + 2 * kv_size, HIDDEN)

    def test_forward_output_shape_without_v(self, dist_init):
        proj = _make_qkv(v_head_size=0)
        x = torch.randn(3, HIDDEN)
        out, _ = proj(x)
        q_size = NUM_Q * HEAD_DIM
        kv_size = NUM_KV * HEAD_DIM
        assert out.shape == (3, q_size + kv_size)

    def test_memory_saved_equals_v_slot(self, dist_init):
        proj_full = _make_qkv(v_head_size=None)
        proj_skip = _make_qkv(v_head_size=0)
        saved = proj_full.weight.numel() - proj_skip.weight.numel()
        assert saved == NUM_KV * HEAD_DIM * HIDDEN


class TestGemma4KeqVMathEquivalence:
    """Verify v_norm(k_pre) == v_old when W_v is a clone of W_k."""

    EPS = 1e-6

    def _make_norms(self, head_dim: int, seed: int = 42):
        torch.manual_seed(seed)
        k_norm = RMSNorm(head_dim, eps=self.EPS)
        v_norm = RMSNorm(head_dim, eps=self.EPS, has_weight=False)
        return k_norm, v_norm

    def test_v_from_v_norm_k_pre_equals_old_v_gemm(self, default_vllm_config):
        hidden, num_kv, head_dim, batch = 64, 2, 16, 5
        torch.manual_seed(0)
        W_k = torch.randn(num_kv * head_dim, hidden)
        k_norm, v_norm = self._make_norms(head_dim)
        x = torch.randn(batch, hidden)

        # Old path: W_v == W_k (weight duplication) → v_pre == k_pre.
        W_v = W_k.clone()
        k_pre = x @ W_k.T
        v_pre = x @ W_v.T
        k_old = k_norm(k_pre.unflatten(-1, (num_kv, head_dim))).flatten(-2, -1)
        v_old = v_norm(v_pre.unflatten(-1, (num_kv, head_dim))).flatten(-2, -1)

        # New path: derive V from the same k_pre via v_norm.
        k_pre_4d = (x @ W_k.T).unflatten(-1, (num_kv, head_dim))
        k_new = k_norm(k_pre_4d).flatten(-2, -1)
        v_new = v_norm(k_pre_4d).flatten(-2, -1)

        assert torch.allclose(k_old, k_new)
        assert torch.allclose(v_old, v_new)

    def test_k_and_v_differ_after_norms(self, default_vllm_config):
        """Sanity: K and V must differ when k_norm.weight != 1."""
        hidden, num_kv, head_dim, batch = 64, 2, 16, 5
        k_norm, v_norm = self._make_norms(head_dim)
        with torch.no_grad():
            k_norm.weight.fill_(2.0)

        W_k = torch.randn(num_kv * head_dim, hidden)
        x = torch.randn(batch, hidden)
        k_pre = (x @ W_k.T).unflatten(-1, (num_kv, head_dim))
        k = k_norm(k_pre).flatten(-2, -1)
        v = v_norm(k_pre).flatten(-2, -1)

        assert not torch.allclose(k, v)
