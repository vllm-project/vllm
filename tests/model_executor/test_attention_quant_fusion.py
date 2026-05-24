# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.attention.attention as attention_module
from vllm.model_executor.layers.quantization.utils.quant_fusion import (
    get_static_fp8_attn_output_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform


class _AttentionImpl:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported
        self.seen_quant_key = None

    def fused_output_quant_supported(self, quant_key):
        self.seen_quant_key = quant_key
        return self.supported


def _linear(activation_quant_key=kFp8StaticTensorSym):
    return SimpleNamespace(
        input_scale=torch.tensor([0.5]),
        input_quant_key=activation_quant_key,
    )


def _attention(supported: bool = True, fuse_attn_quant: bool = True):
    attn = object.__new__(attention_module.Attention)
    attn.calculate_kv_scales = False
    attn.query_quant = None
    attn.num_heads = 2
    attn.num_kv_heads = 2
    attn.head_size = 4
    attn.head_size_v = 4
    attn.use_direct_call = True
    attn.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=True)
    attn.kv_sharing_target_layer_name = None
    attn.layer_name = "test_attn"
    attn.impl = _AttentionImpl(supported=supported)
    attn.use_fused_attn_quant = fuse_attn_quant
    return attn


def test_static_fp8_attn_output_scale_selected_when_supported():
    attn = _attention()
    output_proj = _linear()

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is output_proj.input_scale
    assert attn.impl.seen_quant_key == kFp8StaticTensorSym


def test_static_fp8_attn_output_scale_skipped_without_input_quant_key():
    attn = _attention()
    output_proj = SimpleNamespace(input_scale=torch.tensor([0.5]))

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None


def test_static_fp8_attn_output_scale_skipped_when_flag_disabled():
    attn = _attention(fuse_attn_quant=False)
    output_proj = _linear()

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None
    assert attn.impl.seen_quant_key is None


def test_static_fp8_attn_output_scale_skipped_when_backend_unsupported():
    attn = _attention(supported=False)
    output_proj = _linear()

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None


def test_static_fp8_attn_output_scale_skipped_for_non_static_fp8_linear():
    attn = _attention()
    output_proj = _linear(activation_quant_key=kFp8DynamicTokenSym)

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None


def test_attention_forward_uses_fp8_output_when_scale_provided(monkeypatch):
    captured = {}

    def fake_unified_attention_with_output(
        query,
        key,
        value,
        output,
        layer_name,
        output_scale=None,
        output_block_scale=None,
        kv_cache_dummy_dep=None,
    ):
        captured["output_dtype"] = output.dtype
        captured["output_scale"] = output_scale

    monkeypatch.setattr(
        attention_module,
        "unified_attention_with_output",
        fake_unified_attention_with_output,
    )

    attn = _attention()
    q = torch.randn(3, 8)
    k = torch.randn(3, 8)
    v = torch.randn(3, 8)
    scale = torch.tensor([0.5])

    result = attention_module.Attention.forward(attn, q, k, v, output_scale=scale)

    assert captured["output_scale"] is scale
    assert captured["output_dtype"] == current_platform.fp8_dtype()
    assert result.dtype == current_platform.fp8_dtype()
