# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout dispatch tests for ``QwenGatedDeltaNetAttention._forward_core_rocm``.

The AITER fused reshape+conv kernel learned Qwen3.5's flat ``[q|k|v|z]``
packing in https://github.com/ROCm/aiter/pull/3251, so flat-layout models take
the decode fast path too instead of falling back to the generic path.

These tests run host-side on CPU: ``_forward_core_rocm`` is bound to a stub
layer whose ``_forward_core_decode_aiter``/``_forward_core`` record which
branch ran, so no GPU or AITER install is needed.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

PREFIX = "model.layers.0.linear_attn"
H = 2  # num key heads
HV = 4  # num value heads
K = 8  # head_k_dim
V = 8  # head_v_dim


def _make_metadata(
    *,
    num_prefills: int = 0,
    num_decodes: int = 2,
    spec_sequence_masks: torch.Tensor | None = None,
):
    """Only the fields the dispatch condition reads carry meaningful values."""
    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_prefills + num_decodes,
        spec_sequence_masks=spec_sequence_masks,
    )


def _make_layer(gqa_interleaved_layout: bool):
    """Stub layer running the real ``_forward_core_rocm``, recording dispatch."""
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.gqa_interleaved_layout = gqa_interleaved_layout
    layer.qkvz_layout = "interleaved" if gqa_interleaved_layout else "flat"

    layer.calls = []
    layer._forward_core_decode_aiter = lambda **kw: layer.calls.append("aiter")
    layer._forward_core = lambda **kw: layer.calls.append("generic")
    layer.prepare_gdn_attention_core_inputs = lambda qkvz, ba, n: (
        torch.zeros(n, H * K * 2 + HV * V),
        torch.zeros(n, HV, V),
        torch.zeros(n, HV),
        torch.zeros(n, HV),
    )
    layer._forward_core_rocm = types.MethodType(
        QwenGatedDeltaNetAttention._forward_core_rocm, layer
    )
    return layer


def _run(layer, meta) -> str:
    num_tokens = max(meta.num_actual_tokens, 1)
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core_rocm(
            qkvz=torch.zeros(num_tokens, 2 * H * K + 2 * HV * V),
            ba=torch.zeros(num_tokens, 2 * HV),
            z_out=torch.zeros(num_tokens, HV, V),
            core_attn_out=torch.zeros(num_tokens, HV, V),
        )
    assert len(layer.calls) == 1
    return layer.calls[0]


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
def test_pure_decode_takes_the_fast_path(gqa_interleaved_layout: bool) -> None:
    """Both packings reach the fast path; flat used to be guarded out."""
    layer = _make_layer(gqa_interleaved_layout)
    assert _run(layer, _make_metadata()) == "aiter"


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
def test_layout_string_matches_the_packing(gqa_interleaved_layout: bool) -> None:
    """The value handed to the kernel's ``qkvz_layout`` parameter."""
    layer = _make_layer(gqa_interleaved_layout)
    expected = "interleaved" if gqa_interleaved_layout else "flat"
    assert layer.qkvz_layout == expected


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
@pytest.mark.parametrize(
    "meta_kwargs",
    [
        {"num_prefills": 1},
        {"num_decodes": 0},
        {"spec_sequence_masks": torch.zeros(1, dtype=torch.bool)},
    ],
    ids=["has_prefill", "no_decode", "spec_decode"],
)
def test_non_pure_decode_batches_use_the_generic_path(
    gqa_interleaved_layout: bool, meta_kwargs: dict
) -> None:
    layer = _make_layer(gqa_interleaved_layout)
    assert _run(layer, _make_metadata(**meta_kwargs)) == "generic"
