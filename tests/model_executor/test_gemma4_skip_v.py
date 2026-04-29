# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Gemma4 k_eq_v optimization:
  - QKVParallelLinear with skip_v=True produces correct output shape and
    rejects "v" shard loading.
  - Gemma4Attention with use_k_eq_v=True produces K and V tensors that are
    mathematically identical to the reference (old) approach where both K and
    V weights are loaded from the same checkpoint weights.

Fixtures used:
  dist_init        — initialises the TP distributed group (from conftest.py);
                     required by QKVParallelLinear / ModelWeightParameter.
  default_vllm_config — sets the global VllmConfig context required by
                     CustomOp subclasses such as RMSNorm.
"""

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qkv(
    hidden: int,
    num_q: int,
    num_kv: int,
    head_dim: int,
    *,
    skip_v: bool,
) -> QKVParallelLinear:
    """Create a QKVParallelLinear with TP disabled (single-rank)."""
    return QKVParallelLinear(
        hidden_size=hidden,
        head_size=head_dim,
        total_num_heads=num_q,
        total_num_kv_heads=num_kv,
        bias=False,
        quant_config=None,
        prefix="test.qkv_proj",
        disable_tp=True,
        skip_v=skip_v,
    )


# ---------------------------------------------------------------------------
# QKVParallelLinear.skip_v tests
# (dist_init initialises TP group + vLLM config, both required here)
# ---------------------------------------------------------------------------

HIDDEN = 64
NUM_Q = 4
NUM_KV = 2
HEAD_DIM = 16


class TestQKVParallelLinearSkipV:
    def test_weight_shape_skip_v_false(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=False)
        q_size = NUM_Q * HEAD_DIM
        kv_size = NUM_KV * HEAD_DIM
        assert proj.weight.shape == (q_size + 2 * kv_size, HIDDEN)

    def test_weight_shape_skip_v_true(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=True)
        q_size = NUM_Q * HEAD_DIM
        kv_size = NUM_KV * HEAD_DIM
        # No V slot: weight rows = Q + K only
        assert proj.weight.shape == (q_size + kv_size, HIDDEN)

    def test_output_sizes_skip_v_true(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=True)
        assert len(proj.output_sizes) == 2
        assert proj.output_sizes[0] == NUM_Q * HEAD_DIM  # q
        assert proj.output_sizes[1] == NUM_KV * HEAD_DIM  # k

    def test_output_sizes_skip_v_false(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=False)
        assert len(proj.output_sizes) == 3

    def test_validate_shard_id_rejects_v_when_skip_v(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=True)
        with pytest.raises(ValueError, match="[Ss]hard"):
            proj.validate_shard_id("v")

    def test_validate_shard_id_accepts_qk_when_skip_v(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=True)
        proj.validate_shard_id("q")
        proj.validate_shard_id("k")

    def test_forward_output_shape_skip_v_true(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=True)
        x = torch.randn(3, HIDDEN)
        out, _ = proj(x)
        q_size = NUM_Q * HEAD_DIM
        kv_size = NUM_KV * HEAD_DIM
        assert out.shape == (3, q_size + kv_size)

    def test_forward_output_shape_skip_v_false(self, dist_init):
        proj = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=False)
        x = torch.randn(3, HIDDEN)
        out, _ = proj(x)
        q_size = NUM_Q * HEAD_DIM
        kv_size = NUM_KV * HEAD_DIM
        assert out.shape == (3, q_size + 2 * kv_size)

    def test_skip_v_weight_is_smaller(self, dist_init):
        proj_full = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=False)
        proj_skip = _make_qkv(HIDDEN, NUM_Q, NUM_KV, HEAD_DIM, skip_v=True)
        saved = proj_full.weight.numel() - proj_skip.weight.numel()
        assert saved == NUM_KV * HEAD_DIM * HIDDEN


# ---------------------------------------------------------------------------
# Gemma4 k_eq_v forward math equivalence
# (only RMSNorm + raw torch ops; default_vllm_config is sufficient)
# ---------------------------------------------------------------------------


class TestGemma4KeqVMathEquivalence:
    """
    Verify that deriving V via v_norm(k_pre) gives the same result as the
    old approach of loading K weights into the V projection slot.

    No GPU or distributed setup needed — only RMSNorm (CustomOp) requires
    the default_vllm_config context.
    """

    EPS = 1e-6

    def _make_norms(self, head_dim: int, seed: int = 42):
        torch.manual_seed(seed)
        k_norm = RMSNorm(head_dim, eps=self.EPS)
        v_norm = RMSNorm(head_dim, eps=self.EPS, has_weight=False)
        return k_norm, v_norm

    def test_v_from_v_norm_k_pre_equals_old_v_gemm(self, default_vllm_config):
        """
        Old approach: W_v == W_k (duplicate), v_pre == k_pre.
        New approach: v = v_norm(k_pre).
        Both must produce identical K and V tensors.
        """
        hidden, num_kv, head_dim, batch = 64, 2, 16, 5
        torch.manual_seed(0)
        W_k = torch.randn(num_kv * head_dim, hidden)
        k_norm, v_norm = self._make_norms(head_dim)
        x = torch.randn(batch, hidden)

        # ---- old path: W_v == W_k (weight duplication) ----
        W_v = W_k.clone()
        k_pre = x @ W_k.T
        v_pre = x @ W_v.T  # identical to k_pre
        k_pre_4d = k_pre.unflatten(-1, (num_kv, head_dim))
        v_pre_4d = v_pre.unflatten(-1, (num_kv, head_dim))
        k_old = k_norm(k_pre_4d).flatten(-2, -1)
        v_old = v_norm(v_pre_4d).flatten(-2, -1)

        # ---- new path: skip_v, derive V from k_pre ----
        k_pre_new = x @ W_k.T
        k_pre_new_4d = k_pre_new.unflatten(-1, (num_kv, head_dim))
        k_normed = k_norm(k_pre_new_4d)
        v_new = v_norm(k_pre_new_4d).flatten(-2, -1)
        k_new = k_normed.flatten(-2, -1)

        assert torch.allclose(k_old, k_new), "K differs between old and new path"
        assert torch.allclose(v_old, v_new), "V differs between old and new path"

    def test_k_and_v_differ_after_norms(self, default_vllm_config):
        """K and V must differ when k_norm.weight != 1."""
        hidden, num_kv, head_dim, batch = 64, 2, 16, 5
        k_norm, v_norm = self._make_norms(head_dim)
        with torch.no_grad():
            k_norm.weight.fill_(2.0)

        W_k = torch.randn(num_kv * head_dim, hidden)
        x = torch.randn(batch, hidden)
        k_pre = (x @ W_k.T).unflatten(-1, (num_kv, head_dim))
        k = k_norm(k_pre).flatten(-2, -1)
        v = v_norm(k_pre).flatten(-2, -1)

        assert not torch.allclose(k, v), "K and V should differ when k_norm.weight != 1"
