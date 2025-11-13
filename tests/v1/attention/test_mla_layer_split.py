# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MLAAttention split orchestration in the layer.

These tests verify:
- decode-only path calls forward_decode
- prefill-only path calls forward_prefill
- mixed path splits correctly and concatenates outputs
- clamping of n_decode_tokens to total tokens
"""

import types

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.attention.backends.abstract import MLAAttentionImpl
from vllm.attention.layer import MLAAttention
from vllm.config.vllm import set_current_vllm_config
from vllm.forward_context import set_forward_context


class _DummyMLAImpl(MLAAttentionImpl):
    accept_output_buffer: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype: str,
        logits_soft_cap,
        attn_type: str,
        kv_sharing_target_layer_name,
        q_lora_rank,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj,
        indexer=None,
    ):
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.head_size = head_size

    def supports_compiled_split(self) -> bool:
        return True

    def forward_prefill(
        self,
        layer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # Return 2s with expected output width
        return torch.full(
            (q.shape[0], self.num_heads * self.v_head_dim),
            2.0,
            dtype=q.dtype,
            device=q.device,
        )

    def forward_decode(
        self,
        layer,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # Return 1s with expected output width
        return torch.full(
            (q.shape[0], self.num_heads * self.v_head_dim),
            1.0,
            dtype=q.dtype,
            device=q.device,
        )

    def forward(  # pragma: no cover
        self,
        layer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise AssertionError(
            "Unified forward should not be called in compiled split tests"
        )


class _FakeBackend:
    @staticmethod
    def get_impl_cls():
        return _DummyMLAImpl

    accept_output_buffer = True

    @staticmethod
    def get_name() -> str:
        # Use an existing enum name to satisfy code paths
        return "TRITON_MLA"


def _make_layer(
    monkeypatch, num_heads=2, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4
):
    # Force selector to return our fake backend
    import vllm.attention.selector as selector

    monkeypatch.setattr(
        selector, "get_attn_backend", lambda *args, **kwargs: _FakeBackend
    )

    # Minimal kv_b_proj argument, not used by dummy impl
    kv_b_proj = object()

    layer = MLAAttention(
        num_heads=num_heads,
        scale=1.0,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=None,
        kv_lora_rank=4,
        kv_b_proj=kv_b_proj,  # dummy
        prefix="L0",
    )
    # Overwrite impl with dummy directly to be explicit
    layer.impl = _DummyMLAImpl(
        num_heads=num_heads,
        head_size=layer.head_size,
        scale=layer.scale,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=layer.kv_cache_dtype,
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=layer.kv_lora_rank,
        qk_nope_head_dim=layer.qk_nope_head_dim,
        qk_rope_head_dim=layer.qk_rope_head_dim,
        qk_head_dim=layer.qk_nope_head_dim + layer.qk_rope_head_dim,
        v_head_dim=layer.v_head_dim,
        kv_b_proj=kv_b_proj,
    )
    # Re-enable compiled split based on dummy impl capability
    layer._use_compiled_split = True
    return layer


def _make_metadata(total: int, n_decode: int, device: torch.device):
    # Minimal metadata object with required fields
    meta = types.SimpleNamespace()
    meta.num_actual_tokens = total
    meta.num_decode_tokens = n_decode
    meta.num_decodes = 1 if n_decode > 0 else 0
    meta.num_prefills = 1 if total > n_decode else 0
    meta.slot_mapping = torch.arange(total, dtype=torch.int64, device=device)
    return meta


@pytest.mark.parametrize("total,n_decode", [(5, 5), (6, 0), (7, 3), (4, 9)])
def test_mla_layer_compiled_split(monkeypatch, total, n_decode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    vllm_config = create_vllm_config(dtype=dtype)
    with set_current_vllm_config(vllm_config):
        layer = _make_layer(monkeypatch)
        # Inputs (content unused by dummy impl)
        q = torch.randn(total, 1, dtype=dtype, device=device)
        kv_c_normed = torch.randn(total, 1, dtype=dtype, device=device)
        k_pe = torch.randn(total, 1, 1, dtype=dtype, device=device)
        kv_cache = torch.tensor([], device=device)

        meta = _make_metadata(total, n_decode, device)
        # forward_context expects a dict keyed by layer_name
        attn_md = {"L0": meta}
        # Use compiled split path by entering via direct call and setting context
        with set_forward_context(attn_md, vllm_config=vllm_config, num_tokens=total):
            out = layer.forward_impl(
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                kv_cache=kv_cache,
                attn_metadata=meta,
                allow_compiled_split=True,
            )

        # Validate output shape
        assert out.shape == (meta.num_actual_tokens, layer.num_heads * layer.v_head_dim)

        # Clamp n_decode to total for expected behavior
        n_dec = max(0, min(n_decode, total))
        if n_dec == total:
            # All decode => ones
            assert torch.allclose(out, torch.ones_like(out))
        elif n_dec == 0:
            # All prefill => twos
            assert torch.allclose(out, torch.full_like(out, 2.0))
        else:
            # Mixed => first n_dec rows ones, rest twos
            assert torch.allclose(out[:n_dec], torch.ones_like(out[:n_dec]))
            assert torch.allclose(out[n_dec:], torch.full_like(out[n_dec:], 2.0))
