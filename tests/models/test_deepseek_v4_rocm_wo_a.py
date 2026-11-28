# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    _wo_a_block_scale_to_e8m0,
)


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


def _fp8_wo_a_layer(rotary_emb: RotaryEmbedding) -> DeepseekV4ROCMAiterMLAAttention:
    """A layer with only the state `_prepare_fp8_wo_a` reads."""
    layer = DeepseekV4ROCMAiterMLAAttention.__new__(DeepseekV4ROCMAiterMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.n_local_groups = 1
    layer.o_lora_rank = 128
    layer.wo_a = SimpleNamespace(
        weight=torch.zeros(128, 128, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.ones(1, 1),
    )
    layer.rotary_emb = rotary_emb
    return layer


def test_fp8_wo_a_rope_cos_sin_is_shared_and_refilled_in_place(
    default_vllm_config, monkeypatch: pytest.MonkeyPatch
):
    """Layers sharing a rotary module read one cos/sin pair instead of a copy
    each, and preparing again (a weight reload) refills it in place, so the
    addresses captured CUDA graphs read stay valid."""
    # `_prepare_fp8_wo_a` only checks that the AITER ops import.
    for name in (
        "aiter",
        "aiter.ops",
        "aiter.ops.batched_gemm_op_a8w8",
        "aiter.ops.inverse_rope_group_quant",
    ):
        module: Any = ModuleType(name)
        module.batched_gemm_a8w8_mxscale = None
        module.inverse_rope_group_quant = None
        monkeypatch.setitem(sys.modules, name, module)

    rotary_emb = RotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=False,
        dtype=torch.float32,
    )
    expected_cos, expected_sin = rotary_emb.cos_sin_cache.bfloat16().chunk(2, dim=-1)
    layer, other_layer = _fp8_wo_a_layer(rotary_emb), _fp8_wo_a_layer(rotary_emb)

    layer._prepare_fp8_wo_a()
    other_layer._prepare_fp8_wo_a()
    cos, sin = layer._wo_a_cos_cache, layer._wo_a_sin_cache
    assert other_layer._wo_a_cos_cache is cos and other_layer._wo_a_sin_cache is sin
    assert cos.is_contiguous() and sin.is_contiguous()
    torch.testing.assert_close(cos, expected_cos, atol=0, rtol=0)
    torch.testing.assert_close(sin, expected_sin, atol=0, rtol=0)

    cos.zero_()
    sin.zero_()
    layer._prepare_fp8_wo_a()
    assert layer._wo_a_cos_cache is cos and layer._wo_a_sin_cache is sin
    torch.testing.assert_close(cos, expected_cos, atol=0, rtol=0)
    torch.testing.assert_close(sin, expected_sin, atol=0, rtol=0)
