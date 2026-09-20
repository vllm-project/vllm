# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DeepSeek padded attention-sink loader."""

import pytest
import torch

from vllm.models.deepseek_v4.sink import load_padded_attn_sink

PADDED_HEADS = 8
LOCAL_HEADS = 4
GLOBAL_HEADS = 16


def _make_param(padded_heads: int = PADDED_HEADS) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
        requires_grad=False,
    )


def _checkpoint() -> torch.Tensor:
    return torch.arange(GLOBAL_HEADS, dtype=torch.float32)


@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_load_padded_attn_sink_writes_local_heads_and_keeps_padding_at_neginf(
    tp_rank: int,
):
    """Every rank takes its own heads and keeps the padded tail disabled."""
    param = _make_param()
    loaded_weight = _checkpoint()
    head_start = LOCAL_HEADS * tp_rank
    head_end = head_start + LOCAL_HEADS

    load_padded_attn_sink(param, loaded_weight, head_start, head_end)

    expected_local = loaded_weight[head_start:head_end]
    assert torch.equal(param.data[:LOCAL_HEADS], expected_local)
    assert torch.isneginf(param.data[LOCAL_HEADS:]).all()


def test_load_padded_attn_sink_resets_polluted_padding_tail():
    """A reload must restore the padding tail even if it was polluted."""
    param = _make_param()
    with torch.no_grad():
        param.data[LOCAL_HEADS:] = 0.0

    load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)

    assert torch.isneginf(param.data[LOCAL_HEADS:]).all()


def test_load_padded_attn_sink_preserves_parameter_storage():
    """The helper must load in place so kernel tensor references stay valid."""
    param = _make_param()
    ptr = param.data.data_ptr()

    load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)

    assert param.data.data_ptr() == ptr


def test_load_padded_attn_sink_uses_current_weight_loader():
    """The load must be routed through the parameter's current loader.

    Layerwise reload replaces ``weight_loader`` with ``online_process_loader``;
    reading it dynamically is what makes a live update observable.
    """
    param = _make_param()
    calls = []

    def recording_loader(p, w):
        calls.append((p, w))

    param.weight_loader = recording_loader

    load_padded_attn_sink(param, _checkpoint(), 0, LOCAL_HEADS)

    assert len(calls) == 1
    called_param, called_weight = calls[0]
    assert called_param is param
    # The loader receives the full padded runtime representation so that the
    # element counts match and the replay re-establishes the padding.
    assert called_weight.shape == param.shape
    assert torch.equal(called_weight[:LOCAL_HEADS], _checkpoint()[:LOCAL_HEADS])
    assert torch.isneginf(called_weight[LOCAL_HEADS:]).all()


def test_load_padded_attn_sink_degenerate_head_range():
    """A rank with no heads keeps the whole sink disabled and stays loadable."""
    param = _make_param()

    load_padded_attn_sink(param, _checkpoint(), 0, 0)

    assert torch.isneginf(param.data).all()
