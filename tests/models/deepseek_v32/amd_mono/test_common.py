# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of mono/common.py: RoPE tables and per-layer attention metadata lookup."""

from types import SimpleNamespace as NS

import torch

from vllm.models.deepseek_v32.amd.mono import common as C


def _layer(rows=100, dtype=torch.float32, name="model.layers.3.self_attn.attn"):
    cache = torch.randn(rows, 64, dtype=dtype)
    return NS(self_attn=NS(rotary_emb=NS(cos_sin_cache=cache), layer_name=name))


def test_rope_tables():
    lay = _layer()
    cos, sin, src = C.rope_tables(lay, 40)
    c = lay.self_attn.rotary_emb.cos_sin_cache
    assert cos.shape == (40, 32) and sin.shape == (40, 32) and src == "torch.float32"
    assert cos.dtype is torch.bfloat16 and cos.is_contiguous() and sin.is_contiguous()
    assert torch.equal(cos, c[:40, :32].to(torch.bfloat16))
    assert torch.equal(sin, c[:40, 32:].to(torch.bfloat16))
    cos2, _, _ = C.rope_tables(lay)  # no slicing
    assert cos2.shape == (100, 32)


def test_layer_metadata():
    lay = _layer()
    md = NS(x=1)
    sm = torch.arange(4)
    name = lay.self_attn.layer_name
    fc = NS(attn_metadata={name: md}, slot_mapping={name: sm})
    assert C.layer_metadata(lay, fc=fc) == (md, sm)
    fc_list = NS(attn_metadata=[{name: md}], slot_mapping={name: sm})
    assert C.layer_metadata(lay, "first", fc=fc_list) == (md, sm)  # live-mode policy
    assert C.layer_metadata(lay, "none", fc=fc_list) == (None, None)
    empty = NS(attn_metadata=[], slot_mapping=None)
    assert C.layer_metadata(lay, "first", fc=empty) == (None, None)
    no_md = NS(attn_metadata=None, slot_mapping=sm)
    assert C.layer_metadata(lay, fc=no_md) == (None, None)
