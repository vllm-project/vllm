# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for ``ResidualStream`` module wiring."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

import vllm.model_executor.layers.fusion.residual_stream as rs_mod
from vllm.model_executor.layers.fusion.residual_stream import ResidualStream
from vllm.model_executor.models.llama import LlamaDecoderLayer


@pytest.fixture(autouse=True)
def _stub_tp_world_size(monkeypatch):
    # The stream reads the TP world size at construction; stub it so these tests
    # need no distributed init.
    monkeypatch.setattr(rs_mod, "get_tensor_model_parallel_world_size", lambda: 1)


class _FakeLinear(nn.Module):
    """Stand-in for a consumer linear, carrying only ``reduce_results``."""

    def __init__(self, reduce_results: bool) -> None:
        super().__init__()
        self.reduce_results = reduce_results


def _vllm_config(pp_size: int = 1, lora: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        lora_config=object() if lora else None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
    )


def _stream(*, pp_size: int = 1, **overrides) -> ResidualStream:
    modules = dict(
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        qkv_proj=_FakeLinear(reduce_results=False),
        o_proj=_FakeLinear(reduce_results=False),
        gate_up_proj=_FakeLinear(reduce_results=False),
        down_proj=_FakeLinear(reduce_results=False),
    )
    modules.update(overrides)
    return ResidualStream(vllm_config=_vllm_config(pp_size=pp_size), **modules)


def test_captures_modules_directly():
    mods = dict(
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        qkv_proj=_FakeLinear(False),
        o_proj=_FakeLinear(False),
        gate_up_proj=_FakeLinear(False),
        down_proj=_FakeLinear(False),
    )
    stream = ResidualStream(vllm_config=_vllm_config(), **mods)
    for name, module in mods.items():
        assert getattr(stream, name) is module


@pytest.mark.parametrize(
    "reduce_results,pp_size,expected",
    [
        (False, 1, True),  # linears defer -> leaves a PARTIAL sum for the stream
        (True, 1, False),  # linears reduce themselves -> FULL
        (False, 2, False),  # pipeline parallel disables deferral
    ],
)
def test_defer_reads_reduce_results(reduce_results, pp_size, expected):
    stream = _stream(
        pp_size=pp_size,
        o_proj=_FakeLinear(reduce_results),
        down_proj=_FakeLinear(reduce_results),
    )
    assert stream.defer_attn is expected
    assert stream.defer_mlp is expected


def test_none_linears_do_not_defer():
    # An MoE mlp exposes no gate_up_proj/down_proj; getattr(None, ...) must fall
    # back to reduce_results=True so the stream treats the input as FULL.
    stream = _stream(qkv_proj=None, o_proj=None, gate_up_proj=None, down_proj=None)
    assert stream.defer_attn is False
    assert stream.defer_mlp is False


def _build_via_layer(*, self_attn, mlp, input_layernorm) -> ResidualStream:
    """Call ``LlamaDecoderLayer._build_residual_stream`` on a duck-typed layer.

    Exercises the model-side wiring (which attribute is qkv/gate_up/...) without
    constructing a real decoder layer.
    """
    fake_layer = SimpleNamespace(
        input_layernorm=input_layernorm,
        post_attention_layernorm=nn.Identity(),
        self_attn=self_attn,
        mlp=mlp,
    )
    return LlamaDecoderLayer._build_residual_stream(fake_layer, _vllm_config())


def test_build_wires_attention_and_mlp_leaves():
    qkv, o = _FakeLinear(False), _FakeLinear(False)
    gate_up, down = _FakeLinear(False), _FakeLinear(False)
    stream = _build_via_layer(
        self_attn=SimpleNamespace(qkv_proj=qkv, o_proj=o),
        mlp=SimpleNamespace(gate_up_proj=gate_up, down_proj=down),
        input_layernorm=nn.Identity(),
    )
    assert stream.qkv_proj is qkv
    assert stream.o_proj is o
    assert stream.gate_up_proj is gate_up
    assert stream.down_proj is down


def test_build_tolerates_moe_mlp_without_gate_up():
    # aria swaps in an MoE mlp after super().__init__ and rebuilds; the rebuilt
    # stream must capture None rather than the discarded base-LlamaMLP linears.
    attn = SimpleNamespace(qkv_proj=_FakeLinear(True), o_proj=_FakeLinear(True))
    stream = _build_via_layer(
        self_attn=attn,
        mlp=SimpleNamespace(),  # no gate_up_proj / down_proj
        input_layernorm=nn.Identity(),
    )
    assert stream.gate_up_proj is None
    assert stream.down_proj is None


def test_build_captures_swapped_identity_norm():
    # eagle replaces its layer-0 input_layernorm with Identity and rebuilds; the
    # rebuilt stream must hold the Identity, not the original RMSNorm.
    identity = nn.Identity()
    attn = SimpleNamespace(qkv_proj=_FakeLinear(True), o_proj=_FakeLinear(True))
    mlp = SimpleNamespace(gate_up_proj=_FakeLinear(True), down_proj=_FakeLinear(True))
    stream = _build_via_layer(
        self_attn=attn,
        mlp=mlp,
        input_layernorm=identity,
    )
    assert stream.input_layernorm is identity
