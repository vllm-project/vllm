# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.deepseek_v4.amd.rocm import (
    _plain_fp8_sparse_mla,
    _wo_a_block_scale_to_e8m0,
)
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    _q_rope_fusion_min_tokens,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention


def test_wo_a_block_scale_to_e8m0_from_float():
    scale = torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float32)

    encoded = _wo_a_block_scale_to_e8m0(scale)

    assert encoded is not None
    torch.testing.assert_close(
        encoded,
        torch.tensor([[126, 127, 128, 129]], dtype=torch.uint8),
    )
    assert encoded.is_contiguous()


def test_wo_a_block_scale_to_e8m0_preserves_encoded_scales():
    raw = torch.tensor([[125, 127, 131]], dtype=torch.uint8)
    encoded = raw.view(torch.float8_e8m0fnu)

    converted = _wo_a_block_scale_to_e8m0(encoded)

    assert converted is not None
    torch.testing.assert_close(converted, raw)


@pytest.mark.parametrize(
    "scale",
    [
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[-1.0, 1.0]]),
        torch.tensor([[0.75, 1.0]]),
        torch.tensor([[float("inf"), 1.0]]),
        torch.ones(1, dtype=torch.int32),
    ],
)
def test_wo_a_block_scale_to_e8m0_rejects_invalid_scales(scale: torch.Tensor):
    assert _wo_a_block_scale_to_e8m0(scale) is None


@pytest.fixture
def aiter_fuses_q_rope(monkeypatch):
    def set_probe(value: bool) -> None:
        monkeypatch.setattr(
            rocm_aiter_ops, "sparse_mla_fuses_q_rope", staticmethod(lambda: value)
        )

    return set_probe


@pytest.mark.parametrize(
    ("use_aiter", "kill_switch", "probe", "min_tokens", "expected"),
    [
        (True, "1", True, None, 2048),
        (True, "1", True, "512", 512),
        (True, "0", True, None, None),
        (True, "1", False, None, None),
        (False, "1", True, None, None),
    ],
)
def test_q_rope_fusion_min_tokens_gates(
    monkeypatch, aiter_fuses_q_rope, use_aiter, kill_switch, probe, min_tokens, expected
):
    aiter_fuses_q_rope(probe)
    monkeypatch.setenv("VLLM_ROCM_FUSE_Q_ROPE", kill_switch)
    if min_tokens is not None:
        monkeypatch.setenv("VLLM_ROCM_FUSE_Q_ROPE_MIN_TOKENS", min_tokens)

    assert _q_rope_fusion_min_tokens(use_aiter) == expected


def test_fuses_q_rope_in_attention_uses_step_token_threshold():
    layer = object.__new__(DeepseekV41ROCMAiterMLAAttention)

    layer._q_rope_fusion_min_tokens = None
    assert not layer.fuses_q_rope_in_attention(1 << 20)

    layer._q_rope_fusion_min_tokens = 2048
    assert not layer.fuses_q_rope_in_attention(2047)
    assert layer.fuses_q_rope_in_attention(2048)
    assert not DeepseekV4Attention.fuses_q_rope_in_attention(layer, 1 << 20)


def _plain_fp8_layer() -> SimpleNamespace:
    return SimpleNamespace(
        scale=0.5,
        head_dim=512,
        attn_sink=torch.zeros(8),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(16, 64)),
        _flashinfer_fp8_q_scale=torch.ones(1),
        _flashinfer_fp8_kv_scale=torch.ones(1),
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["raw_bf16", "fp8"]
)
def test_plain_fp8_sparse_mla_hands_raw_q_to_the_kernel_to_rotate(monkeypatch, q_dtype):
    calls: list[dict] = []
    monkeypatch.setattr(
        rocm_aiter_ops,
        "triton_sparse_mla_fwd",
        staticmethod(lambda *args, **kwargs: calls.append(kwargs)),
    )
    layer = _plain_fp8_layer()
    positions = torch.arange(4)
    indptr = torch.tensor([0, 1])

    _plain_fp8_sparse_mla(
        layer,
        torch.zeros(4, 8, 512).to(q_dtype),
        torch.empty(4, 8, 512, dtype=torch.bfloat16),
        torch.empty(1),
        indptr,
        indptr,
        None,
        None,
        None,
        positions=positions,
    )

    (kwargs,) = calls
    assert kwargs["dot_precision"] == "fp8"
    assert kwargs["inv_rope_positions"] is positions
    if q_dtype == torch.bfloat16:
        assert kwargs["q_rope_positions"] is positions
        assert kwargs["q_rope_cos_sin_cache"] is layer.rotary_emb.cos_sin_cache
    else:
        assert "q_rope_positions" not in kwargs


@pytest.fixture
def fake_aiter_sparse_mla(monkeypatch):
    triton_calls: list[dict] = []
    flydsl_calls: list[tuple] = []

    def sparse_mla_fwd(q, kv_buffer, kv_indptr, kv_indices, sm_scale, **kwargs):
        triton_calls.append(kwargs)
        return None, None

    def flydsl_sparse_mla_fwd(*args, **kwargs):
        flydsl_calls.append(args)
        return False, None

    module = ModuleType("aiter.ops.triton.attention.sparse_mla")
    module.__dict__["sparse_mla_fwd"] = sparse_mla_fwd
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        rocm_aiter_ops, "_flydsl_sparse_mla_fwd", staticmethod(flydsl_sparse_mla_fwd)
    )
    return triton_calls, flydsl_calls


def _call_triton_sparse_mla_fwd(**kwargs) -> None:
    rocm_aiter_ops.triton_sparse_mla_fwd(
        torch.zeros(1, 8, 512, dtype=torch.bfloat16),
        torch.empty(1),
        torch.empty(1, 8, 512, dtype=torch.bfloat16),
        0.5,
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        qk_rope_head_dim=0,
        **kwargs,
    )


def test_triton_sparse_mla_fwd_without_q_rope_keeps_the_old_call(
    fake_aiter_sparse_mla,
):
    triton_calls, flydsl_calls = fake_aiter_sparse_mla

    _call_triton_sparse_mla_fwd()

    (kwargs,) = triton_calls
    assert len(flydsl_calls) == 1
    assert not any(key.startswith("q_rope") or key == "q_norm_eps" for key in kwargs)
    assert kwargs["q_scale"] is None
    assert kwargs["dot_precision"] == "bf16"


def test_triton_sparse_mla_fwd_forwards_q_rope_and_skips_flydsl(
    fake_aiter_sparse_mla,
):
    triton_calls, flydsl_calls = fake_aiter_sparse_mla
    positions, cache, q_scale = torch.zeros(1), torch.zeros(1), torch.ones(1)

    _call_triton_sparse_mla_fwd(
        q_scale=q_scale, q_rope_positions=positions, q_rope_cos_sin_cache=cache
    )

    (kwargs,) = triton_calls
    assert not flydsl_calls
    assert kwargs["q_rope_positions"] is positions
    assert kwargs["q_rope_cos_sin_cache"] is cache
    assert kwargs["q_scale"] is q_scale
    assert kwargs["dot_precision"] == "fp8"
