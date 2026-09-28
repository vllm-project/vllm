# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.passes.utility.xpu_gdn_output_alloc import (
    XpuGdnOutputAllocPass,
)
from vllm.config import VllmConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="vllm::gdn_attention_core_xpu is XPU only"
)


@pytest.fixture
def alloc_pass():
    import vllm._xpu_ops  # noqa: F401  (registers vllm::gdn_attention_core_xpu)

    return XpuGdnOutputAllocPass(VllmConfig())


def _graph(fill_value=0, extra_user=False):
    g = fx.Graph()
    qkvz, ba = g.placeholder("qkvz"), g.placeholder("ba")
    kw = {"dtype": torch.float16, "device": torch.device("xpu")}
    buf = g.call_function(torch.ops.aten.full.default, ([4, 16, 128], fill_value), kw)
    z = g.call_function(torch.ops.aten.empty.memory_format, ([4, 16, 128],), kw)
    af = g.call_function(
        auto_functionalized,
        (torch.ops.vllm.gdn_attention_core_xpu.default,),
        {
            "core_attn_out": buf,
            "z": z,
            "projected_states_qkvz": qkvz,
            "projected_states_ba": ba,
            "layer_name": "model.layers.0.linear_attn",
        },
    )
    g.output((af, buf) if extra_user else (af,))
    return g


def _targets(g):
    return [n.target for n in g.nodes if n.op == "call_function"]


def test_zero_fill_becomes_empty(alloc_pass):
    g = _graph()
    alloc_pass(g)
    assert alloc_pass.matched_count == 1
    assert torch.ops.aten.full.default not in _targets(g)


@pytest.mark.parametrize("fill_value,extra_user", [(1.0, False), (0, True)])
def test_other_buffers_unchanged(alloc_pass, fill_value, extra_user):
    g = _graph(fill_value, extra_user)
    alloc_pass(g)
    assert alloc_pass.matched_count == 0
    assert torch.ops.aten.full.default in _targets(g)
