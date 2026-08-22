# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's attention sink fuser."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.transformers.fuser import get_fuser, get_fusers
from vllm.model_executor.models.transformers.fusers import QKVFuser, SinkFuser

from .test_linear import FakeAttention


class SinkAttention(FakeAttention):
    """GPT-OSS style: a learnable per-head sink passed to the interface."""

    sink_attr = "sinks"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        heads = self.q_proj.out_features // self.head_dim
        setattr(self, self.sink_attr, nn.Parameter(torch.zeros(heads)))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, None
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            scaling=self.scaling,
            s_aux=getattr(self, self.sink_attr, None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class SingularSinkAttention(SinkAttention):
    """Same, but the parameter is named `sink` rather than `sinks`."""

    sink_attr = "sink"


class MaybeSinkAttention(SinkAttention):
    """Sinks on some layers only, e.g. MiMo-V2-Flash's sliding window layers."""

    def __init__(self, sinks: bool = True, **kwargs):
        super().__init__(**kwargs)
        if not sinks:
            del self.sinks


def fake_config(tp_size: int = 1, dtype: torch.dtype = torch.bfloat16):
    """The `fuse` inputs a sink needs, without building an engine config."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        model_config=SimpleNamespace(dtype=dtype),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


@pytest.mark.parametrize(
    "cls,expected", [(SinkAttention, "sinks"), (SingularSinkAttention, "sink")]
)
def test_sink_is_found_by_kwarg_not_by_name(cls, expected):
    """The sink is whatever reaches the attention interface as `s_aux`, so the name
    the model gives the parameter does not matter."""
    with torch.device("meta"):
        module = cls()
        fuser = get_fuser(module, SinkFuser)
    assert isinstance(fuser, SinkFuser)
    assert fuser.sink_name == expected
    assert fuser.sink(module) is getattr(module, expected)


def test_attention_without_sink_is_not_matched():
    with torch.device("meta"):
        assert get_fuser(FakeAttention(), SinkFuser) is None


def test_sink_composes_with_projection_fusion():
    """Sinks are one attribute of an attention module, so the QKV fusion of the same
    module must still apply."""
    with torch.device("meta"):
        fusers = get_fusers(SinkAttention())
    assert [type(fuser) for fuser in fusers] == [QKVFuser, SinkFuser]


@pytest.mark.parametrize("sinks", [True, False])
def test_sink_is_optional_per_instance(sinks):
    """Layers of one class that leave the sink unset must not be handed one."""
    with torch.device("meta"):
        module = MaybeSinkAttention(sinks=sinks)
        fuser = get_fuser(module, SinkFuser)
    if not sinks:
        assert fuser is None
        return
    assert fuser.validate(module, fake_config())
    with torch.device("meta"):
        assert not fuser.validate(MaybeSinkAttention(sinks=False), fake_config())


@pytest.mark.parametrize("tp_size", [1, 2])
def test_fuse_makes_sink_loadable(tp_size):
    """`fuse` sizes the sink to this rank's heads and gives it a sharding loader.

    `Attention` is handed the parameter this creates, so the checkpoint has to load
    into it directly.
    """
    module = SinkAttention(heads=4)
    fuser = get_fuser(module, SinkFuser)
    original = module.sinks
    fused = fuser.fuse(module, "self_attn", fake_config(tp_size))

    sink = fuser.sink(fused)
    assert fused is module and sink is not original
    assert sink.shape == (original.numel() // tp_size,)
    assert sink.dtype == torch.bfloat16
    assert not sink.requires_grad
    assert hasattr(sink, "weight_loader")
