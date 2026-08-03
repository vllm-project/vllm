# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA3 WNA16 MoE decode-scratch buffers survive a second post-load pass.

RFC #48312 lists these under category 1 (storage identity): they are plain
layer attributes passed straight to the fused HIP kernel, so reload
copy-back never sees them, and post-load processing re-runs on every
reload.

gfx1100 is not available here, so this covers the allocation contract only
-- the buffer geometry helper is deliberately separated from the HIP path
so it can be driven on CPU. Graph-replay validation on RDNA3 hardware is
still outstanding.
"""

import pytest
import torch
from torch import nn

rdna3 = pytest.importorskip(
    "vllm.model_executor.layers.quantization.compressed_tensors"
    ".compressed_tensors_moe.compressed_tensors_moe_wna16_rdna3")

from vllm.model_executor.reload_arena import peek_reload_arena  # noqa: E402

BUFFERS = ("rdna3_w1_buf", "rdna3_act_buf", "rdna3_out_buf",
           "rdna3_empty_tw")

GEOMETRY = dict(n_gate_up=64, hidden_size=32, act_dtype=torch.float32,
                device=torch.device("cpu"))


def _allocate(layer):
    rdna3.allocate_rdna3_decode_buffers(layer, **GEOMETRY)


def test_second_post_load_pass_reuses_storage():
    """The bug: a reload's post-load pass rebinds every buffer, leaving any
    graph captured earlier pointing at freed memory."""
    layer = nn.Module()
    _allocate(layer)
    first = {name: getattr(layer, name).data_ptr() for name in BUFFERS}

    _allocate(layer)  # what reload triggers
    second = {name: getattr(layer, name).data_ptr() for name in BUFFERS}

    assert second == first


def test_buffers_stay_out_of_state_dict():
    """Arena slots must not become parameters or buffers: they are not
    checkpoint state and must not be touched by load/copy-back machinery."""
    layer = nn.Module()
    _allocate(layer)
    assert layer.state_dict() == {}
    assert list(layer.parameters()) == []
    assert list(layer.buffers()) == []


def test_buffers_are_declared_to_the_arena():
    """The commit gate can only verify storage the arena knows about."""
    layer = nn.Module()
    _allocate(layer)
    arena = peek_reload_arena(layer)
    assert arena is not None
    assert set(arena.slots()) == set(BUFFERS)


def test_commit_gate_sees_a_clean_reload():
    layer = nn.Module()
    _allocate(layer)
    arena = peek_reload_arena(layer)
    snap = arena.snapshot()
    _allocate(layer)
    assert arena.verify(snap) == []


def test_geometry_matches_the_apply_path():
    """apply() slices w1_buf by total_tokens and feeds act_buf to the gated
    activation, so the two must agree on rows and halve on columns."""
    layer = nn.Module()
    _allocate(layer)
    w1, act = layer.rdna3_w1_buf, layer.rdna3_act_buf
    assert w1.shape[0] == act.shape[0]
    assert w1.shape[1] == GEOMETRY["n_gate_up"]
    assert act.shape[1] == GEOMETRY["n_gate_up"] // 2
    assert layer.rdna3_out_buf.shape[1] == GEOMETRY["hidden_size"]
    assert layer.rdna3_empty_tw.numel() == 0


def test_incompatible_geometry_fails_closed():
    """A shape-changing reload is not an in-place update; refusing beats
    silently replacing storage a captured graph may hold."""
    layer = nn.Module()
    _allocate(layer)
    with pytest.raises(ValueError, match="incompatible spec"):
        rdna3.allocate_rdna3_decode_buffers(
            layer, **{**GEOMETRY, "n_gate_up": 128})
