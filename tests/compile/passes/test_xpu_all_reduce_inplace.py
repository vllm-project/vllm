# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator

import pytest
import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

# Registers vllm::all_reduce.
import vllm.distributed.parallel_state  # noqa: F401
from vllm.compilation.passes.utility.xpu_all_reduce_inplace import (
    XpuAllReduceInplacePass,
)
from vllm.config import VllmConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="vllm::xpu_all_reduce_ is XPU only"
)

ALL_REDUCE = torch.ops.vllm.all_reduce.default


@pytest.fixture
def ar_pass():
    import vllm._xpu_ops  # noqa: F401  (registers vllm::xpu_all_reduce_)

    return XpuAllReduceInplacePass(VllmConfig())


def _val(*shape):
    return torch.empty(*shape, dtype=torch.float16, device="meta")


def _graph(producer: str, extra_user: bool = False):
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
    g = _graph("add")
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
    g = _graph(producer)
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
    g = _graph("add", extra_user=True)
    ar_pass(g)
    assert ar_pass.matched_count == 0
    assert ALL_REDUCE in _targets(g)
