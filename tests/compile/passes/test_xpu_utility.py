# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator

import pytest
import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

# Registers vllm::all_reduce.
import vllm.distributed.parallel_state  # noqa: F401
from vllm.compilation.passes.utility.xpu_utility import (
    XpuAllReduceInplacePass,
    XpuGdnOutputAllocPass,
)
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

ALL_REDUCE = torch.ops.vllm.all_reduce.default


@pytest.fixture
def ar_pass():
    import vllm._xpu_ops  # noqa: F401  (registers vllm::xpu_all_reduce_)

    return XpuAllReduceInplacePass(VllmConfig())


def _val(*shape):
    return torch.empty(*shape, dtype=torch.float16, device="meta")


def _ar_graph(producer: str, extra_user: bool = False):
    """Graph: x = <producer>(a, b); y = all_reduce(x); return y [, x]"""
    g = fx.Graph()
    a, b = g.placeholder("a"), g.placeholder("b")
    a.meta["val"], b.meta["val"] = _val(4, 8), _val(4, 8)
    if producer == "placeholder":
        x = a
    elif producer == "view":
        x = g.call_function(torch.ops.aten.view.default, (a, [8, 4]))
        x.meta["val"] = _val(8, 4)
    elif producer == "transpose":
        x = g.call_function(torch.ops.aten.permute.default, (a, [1, 0]))
        x.meta["val"] = _val(4, 8).t()
    else:
        x = g.call_function(torch.ops.aten.add.Tensor, (a, b))
        x.meta["val"] = _val(4, 8)
    y = g.call_function(ALL_REDUCE, (x, "tp:0"))
    y.meta["val"] = _val(*x.meta["val"].shape)
    g.output((y, x) if extra_user else (y,))
    return g


def _targets(g):
    return [n.target for n in g.nodes if n.op == "call_function"]


def test_rewrites_fresh_single_use_input(ar_pass):
    g = _ar_graph("add")
    ar_pass(g)
    assert ar_pass.matched_count == 1
    targets = _targets(g)
    assert ALL_REDUCE not in targets
    af = next(n for n in g.nodes if n.target is auto_functionalized)
    assert af.args[0] is torch.ops.vllm.xpu_all_reduce_.default
    assert af.kwargs["group_name"] == "tp:0"
    out = next(iter(g.output_node().args[0]))
    assert out.target is operator.getitem and out.args == (af, 1)


@pytest.mark.parametrize("producer", ["placeholder", "view", "transpose"])
def test_unsafe_inputs_unchanged(ar_pass, producer):
    g = _ar_graph(producer)
    ar_pass(g)
    assert ar_pass.matched_count == 0
    assert ALL_REDUCE in _targets(g)


def test_rewrites_result_of_multi_output_op(ar_pass):
    g = fx.Graph()
    a, b = g.placeholder("a"), g.placeholder("b")
    a.meta["val"], b.meta["val"] = _val(4, 8), _val(4, 8)
    # aten.var_mean returns two fresh tensors.
    vm = g.call_function(torch.ops.aten.var_mean.correction, (a, [1]))
    x = g.call_function(operator.getitem, (vm, 0))
    x.meta["val"] = _val(4)
    y = g.call_function(ALL_REDUCE, (x, "tp:0"))
    y.meta["val"] = _val(4)
    g.output((y,))
    ar_pass(g)
    assert ar_pass.matched_count == 1


def test_input_with_other_user_unchanged(ar_pass):
    g = _ar_graph("add", extra_user=True)
    ar_pass(g)
    assert ar_pass.matched_count == 0
    assert ALL_REDUCE in _targets(g)


def test_all_reduce_decode_ranges_only(ar_pass):
    assert ar_pass.is_applicable_for_range(Range(start=1, end=8))
    assert not ar_pass.is_applicable_for_range(Range(start=9, end=4096))


@pytest.fixture
def alloc_pass():
    import vllm._xpu_ops  # noqa: F401  (registers vllm::gdn_attention_core_xpu)

    return XpuGdnOutputAllocPass(VllmConfig())


def _gdn_graph(fill_value=0, extra_user=False):
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


def test_zero_fill_becomes_empty(alloc_pass):
    g = _gdn_graph()
    alloc_pass(g)
    assert alloc_pass.matched_count == 1
    assert torch.ops.aten.full.default not in _targets(g)


@pytest.mark.parametrize("fill_value,extra_user", [(1.0, False), (0, True)])
def test_other_buffers_unchanged(alloc_pass, fill_value, extra_user):
    g = _gdn_graph(fill_value, extra_user)
    alloc_pass(g)
    assert alloc_pass.matched_count == 0
    assert torch.ops.aten.full.default in _targets(g)
