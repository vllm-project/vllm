# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer, KimiMLAAttention


class _DirectMLA(KimiMLAAttention):
    def __init__(self, result: torch.Tensor) -> None:
        nn.Module.__init__(self)
        self.result = result

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.result


def _make_layer(self_attn: nn.Module) -> KimiDecoderLayer:
    layer = object.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = self_attn
    return layer


def test_mla_self_attention_returns_projection_storage_directly() -> None:
    hidden_states = torch.randn(4, 8)
    projected = torch.randn_like(hidden_states)
    layer = _make_layer(_DirectMLA(projected))

    output = layer._run_self_attn(torch.arange(4), hidden_states)

    assert output is projected
    assert output.data_ptr() == projected.data_ptr()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("fused_decode", [False, True])
def test_mla_forward_preserves_output_shape_with_fp8_input(
    monkeypatch, quantized: bool, fused_decode: bool
) -> None:
    """Both MLA paths accept producer-quantized input after decode fusion."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )
    from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper

    hidden_states = torch.randn(4, 8, dtype=torch.bfloat16)
    if quantized:
        hidden_states = QuantizedActivation(
            hidden_states.to(kFp8DynamicTokenSym.dtype),
            torch.ones(4, 1),
            hidden_states.dtype,
            hidden_states.shape,
            kFp8DynamicTokenSym,
        )
    projected = torch.randn(4, 8, dtype=torch.bfloat16)
    paths = []

    def input_projection(value):
        assert value is hidden_states
        return torch.zeros(4, 8, dtype=torch.bfloat16), None

    def attention_output(path, output_shape):
        assert output_shape == projected.shape
        paths.append(path)
        return projected

    attention = SimpleNamespace(
        q_lora_rank=2,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        qk_head_dim=4,
        num_heads=2,
        v_head_dim=4,
        fused_qkv_a_proj=input_projection,
        q_a_layernorm=nn.Identity(),
        _normalize_q_kv=lambda q, kv: (q, kv),
        q_b_proj=lambda q: (torch.zeros(4, 8, dtype=q.dtype), None),
        rotary_emb=None,
        indexer=None,
        dcp_q_replicate=False,
        _fused_qk_prep=fused_decode,
        _fused_decode=lambda q, kv, pe, pos, shape, *args: attention_output(
            "fused", shape
        ),
        g_proj=None,
        o_proj=lambda value: (value, None),
    )
    if fused_decode:
        attention.mla_attn = SimpleNamespace(layer_name="test")
        monkeypatch.setattr(
            "vllm.models.kimi_k3.amd.mla.get_attention_context",
            lambda _: (
                SimpleNamespace(num_actual_tokens=4, num_decode_tokens=4),
                None,
                torch.zeros(1),
                torch.arange(4),
            ),
        )
    else:
        attention.mla_attn = lambda q, kv, pe, **kwargs: attention_output(
            "regular", kwargs["output_shape"]
        )

    output = KimiK3MultiHeadLatentAttentionWrapper.forward(
        attention, torch.arange(4), hidden_states
    )

    assert output is projected
    assert paths == ["fused" if fused_decode else "regular"]
