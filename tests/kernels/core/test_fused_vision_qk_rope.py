# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed vision rotation must preserve native output and graph replay."""

import einops
import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.fused_vision_qk_rope import (
    _supported_layout,
    try_fused_vision_qk_rope,
)


def _inputs(seq, batch, strided=False, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(125)
    x = torch.randn(
        (seq, batch, 3456), dtype=torch.bfloat16, device=device, generator=generator
    )
    angle = torch.randn(
        (seq, 72 if strided else 36), device=device, generator=generator
    )
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    if strided:
        cos, sin = cos[:, ::2], sin[:, ::2]
    return x, cos, sin


def _fused(x, cos, sin):
    return try_fused_vision_qk_rope(
        x, cos, sin, 16, 72, is_neox_style=True, enable_fp32_compute=False
    )


def _original(x, cos, sin, rotary):
    qkv = einops.rearrange(x, "s b (three h d) -> b s three h d", three=3, h=16)
    qk = einops.rearrange(qkv[:, :, :2], "b s two h d -> (two b) s h d", two=2)
    return rotary(qk.contiguous(), cos, sin)


def test_metadata_fallback():
    x, cos, sin = _inputs(17, 1, device="cpu")
    assert _supported_layout(x, cos, sin, 16, 72)
    assert _fused(x, cos, sin) is None
    assert not _supported_layout(x[:, :, :-1], cos, sin, 16, 72)
    assert not _supported_layout(x.float(), cos, sin, 16, 72)
    assert not _supported_layout(x, cos[:-1], sin[:-1], 16, 72)
    assert not _supported_layout(x[:0], cos, sin, 16, 72)


def test_metadata_wide_offsets():
    x = torch.empty((400000, 2, 3456), dtype=torch.bfloat16, device="meta")
    cos = torch.empty((400000, 36), dtype=x.dtype, device="meta")
    assert x.numel() > 2**31
    assert _supported_layout(x, cos, cos, 16, 72)
    x = torch.empty((4000000, 1, 3456), dtype=x.dtype, device="meta")
    cos = torch.empty((4000000, 36), dtype=x.dtype, device="meta")
    assert not _supported_layout(x, cos, cos, 16, 72)


@pytest.mark.parametrize("seq,batch", [(17, 1), (257, 2), (8192, 1), (46080, 1)])
@pytest.mark.parametrize("strided", [False, True])
def test_native_parity(default_vllm_config, seq, batch, strided):
    x, cos, sin = _inputs(seq, batch, strided)
    original = x.clone()
    rotary = ApplyRotaryEmb(enforce_enable=True)
    actual = _fused(x, cos, sin)
    assert actual is not None
    torch.testing.assert_close(actual, _original(x, cos, sin, rotary), rtol=0, atol=0)
    assert torch.equal(x, original)


def test_changed_input_graph_replay(default_vllm_config):
    x, cos, sin = _inputs(8192, 1)
    rotary = ApplyRotaryEmb(enforce_enable=True)
    _fused(x, cos, sin)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _fused(x, cos, sin)
    changed = _inputs(8192, 1, strided=True)
    x.copy_(-changed[0])
    cos.copy_(changed[1])
    sin.copy_(changed[2])
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual, _original(x, cos, sin, rotary), rtol=0, atol=0)
