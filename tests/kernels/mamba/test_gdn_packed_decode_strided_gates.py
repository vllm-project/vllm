# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed GDN decode path reads the b/a gates as strided views.

``forward_cuda`` used to copy b and a into contiguous tensors before the core
op on every step; the packed recurrent decode kernel takes a token stride for
both, so only the other paths copy them now.
"""

from __future__ import annotations

import types

import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def test_packed_decode_gets_strided_gates(monkeypatch) -> None:
    metadata = object.__new__(GDNAttentionMetadata)
    metadata.spec_sequence_masks = None
    metadata.num_prefills = 0
    metadata.num_decodes = 2
    context = types.SimpleNamespace(attn_metadata={"layer.0": metadata})
    monkeypatch.setattr(qwen_gdn_linear_attn, "get_forward_context", lambda: context)

    seen = {}

    def decode_non_spec(**kwargs):
        seen.update(kwargs)

    layer = types.SimpleNamespace(
        prefix="layer.0",
        enable_packed_recurrent_decode=True,
        _forward_core_decode_non_spec=decode_non_spec,
    )
    projection = torch.randn(2, 40)
    b, a = projection[:, 32:36], projection[:, 36:40]
    assert not b.is_contiguous()
    QwenGatedDeltaNetAttention._forward_core(
        layer,
        mixed_qkv=projection[:, :32],
        b=b,
        a=a,
        core_attn_out=torch.empty(2, 4, 8),
    )
    assert seen["b"] is b
    assert seen["a"] is a
