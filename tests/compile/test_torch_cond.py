# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for torch.cond support in vLLM's piecewise compilation backend.

After split_module, get_attr nodes from torch.cond branch subgraphs can be
interspersed with placeholder nodes. get_fake_args_from_graph must skip
get_attr nodes to collect all placeholder fake inputs correctly.
"""
import pytest
import torch
import torch.nn as nn

from vllm.compilation.backends import split_graph


class SingleCondModel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        x = self.fc1(x)

        def true_fn(x):
            return x * 2.0

        def false_fn(x):
            return x * 3.0

        x = torch.cond(x.shape[0] < 32, true_fn, false_fn, (x,))
        return self.fc2(x)


class MultiCondModel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, d)

    def forward(self, x):
        x = self.fc1(x)

        def true1(x):
            return x * 2.0

        def false1(x):
            return x * 3.0

        x = torch.cond(x.shape[0] < 32, true1, false1, (x,))
        x = self.fc2(x)

        def true2(x):
            return torch.relu(x)

        def false2(x):
            return torch.sigmoid(x)

        x = torch.cond(x.shape[0] < 16, true2, false2, (x,))
        return self.fc3(x)


def _get_split_submod(model_cls):
    """Trace model through dynamo, split, and return the submodule."""
    model = model_cls().cuda().eval()
    result = {}

    def capture_backend(gm, example_inputs):
        split_gm, items = split_graph(gm, [])
        for item in items:
            if not item.is_splitting_graph:
                result["submod"] = getattr(split_gm, item.submod_name)
                result["example_inputs"] = example_inputs
        return gm.forward

    compiled = torch.compile(model, backend=capture_backend, dynamic=True)
    compiled(torch.randn(16, 64, device="cuda"))
    return result["submod"], result["example_inputs"]


@pytest.mark.parametrize(
    "model_cls,expected_get_attrs",
    [
        (SingleCondModel, 2),
        (MultiCondModel, 4),
    ],
    ids=["single_cond", "multi_cond"],
)
def test_fakify_args_with_torch_cond(model_cls, expected_get_attrs):
    """get_fake_args_from_graph must collect all placeholders even when
    get_attr nodes from torch.cond branches are interspersed between them."""
    from torch._inductor.compile_fx import compile_fx

    from vllm.compilation.piecewise_backend import get_fake_args_from_graph

    submod, example_inputs = _get_split_submod(model_cls)

    get_attrs = [n for n in submod.graph.nodes if n.op == "get_attr"]
    all_ph = [n for n in submod.graph.nodes if n.op == "placeholder"]
    assert len(get_attrs) == expected_get_attrs

    fake = get_fake_args_from_graph(submod)
    assert len(fake) == len(all_ph)
    assert len(fake) == len(example_inputs)

    compiled = compile_fx(submod, fake)
    assert compiled is not None
