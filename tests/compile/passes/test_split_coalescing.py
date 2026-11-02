# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import pytest
import torch

import vllm
from tests.compile.backend import TestBackend
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import CompilationConfig, CompilationMode, PassConfig, VllmConfig
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


class SplitCoalescingModel(torch.nn.Module):
    """Model with 3 separate split_with_sizes calls on the same input,
    simulating the B200+FP8 graph where CSE fails to merge them."""

    def __init__(self, q_size: int, kv_size: int) -> None:
        super().__init__()
        self.q_size = q_size
        self.kv_size = kv_size

    def forward(self, qkv: torch.Tensor):
        q, _, _ = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        _, k, _ = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        _, _, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q + 1, k + 2, v + 3


class SliceCoalescingModel(torch.nn.Module):
    """Model with a complete QKV partition represented as contiguous slices."""

    def __init__(self, q_size: int, kv_size: int) -> None:
        super().__init__()
        self.q_size = q_size
        self.kv_size = kv_size

    def forward(self, qkv: torch.Tensor):
        q_end = self.q_size
        k_end = q_end + self.kv_size
        v_end = k_end + self.kv_size
        q = torch.ops.aten.slice.Tensor(qkv, -1, 0, q_end)
        k = torch.ops.aten.slice.Tensor(qkv, -1, q_end, k_end)
        v = torch.ops.aten.slice.Tensor(qkv, -1, k_end, v_end)
        return q + 1, k + 2, v + 3


def make_vllm_config() -> VllmConfig:
    return VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(),
        )
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_coalescing(dtype):
    torch.set_default_device(DEVICE_TYPE)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    q_size, kv_size = 2048, 512

    vllm_config = make_vllm_config()
    with vllm.config.set_current_vllm_config(vllm_config):
        coalesce_pass = SplitCoalescingPass(vllm_config)
        backend = TestBackend(coalesce_pass)

        model = SplitCoalescingModel(q_size, kv_size)

        T = 5
        qkv = torch.randn(T, q_size + 2 * kv_size)
        torch._dynamo.mark_dynamic(qkv, 0)

        result_eager = model(qkv)

        model_compiled = torch.compile(model, backend=backend)
        result_compiled = model_compiled(qkv)

        ATOL, RTOL = (2e-3, 2e-3)
        for eager, compiled in zip(result_eager, result_compiled):
            torch.testing.assert_close(eager, compiled, atol=ATOL, rtol=RTOL)

        assert backend.op_count(torch.ops.aten.split_with_sizes.default) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_slice_partition_canonicalization(dtype):
    torch.set_default_device(DEVICE_TYPE)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    q_size, kv_size = 2048, 512
    vllm_config = make_vllm_config()
    with vllm.config.set_current_vllm_config(vllm_config):
        coalesce_pass = SplitCoalescingPass(vllm_config)
        backend = TestBackend(coalesce_pass)
        model = SliceCoalescingModel(q_size, kv_size)

        qkv = torch.randn(5, q_size + 2 * kv_size)
        torch._dynamo.mark_dynamic(qkv, 0)
        result_eager = model(qkv)
        result_compiled = torch.compile(model, backend=backend, fullgraph=True)(qkv)

        for eager, compiled in zip(result_eager, result_compiled):
            torch.testing.assert_close(eager, compiled, atol=2e-3, rtol=2e-3)

        assert backend.op_count(torch.ops.aten.slice.Tensor, before=True) == 3
        assert backend.op_count(torch.ops.aten.slice.Tensor) == 0
        assert backend.op_count(torch.ops.aten.split_with_sizes.default) == 1

        split = next(
            node
            for node in backend.final_graph.nodes
            if node.target == torch.ops.aten.split_with_sizes.default
        )
        assert list(split.args[1]) == [q_size, kv_size, kv_size]
        assert split.args[2] == -1

        getitems = [
            node
            for node in split.users
            if node.op == "call_function" and node.target == operator.getitem
        ]
        assert sorted(node.args[1] for node in getitems) == [0, 1, 2]
        assert isinstance(split.meta["val"], list)
        for getitem in getitems:
            index = getitem.args[1]
            assert getitem.meta["val"] is split.meta["val"][index]


def make_slice_graph(
    ranges: list[tuple[int, int, int]],
) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    source = graph.placeholder("qkv")
    source.meta["val"] = torch.empty((2, 8), device="meta")

    slices = []
    for start, end, step in ranges:
        node = graph.call_function(
            torch.ops.aten.slice.Tensor,
            args=(source, -1, start, end, step),
        )
        node.meta["val"] = torch.empty(
            (2, len(range(start, min(end, 8), step))), device="meta"
        )
        slices.append(node)
    graph.output(tuple(slices))
    return graph


def test_slice_partition_accepts_safe_group():
    graph = make_slice_graph([(0, 4, 1), (4, 8, 1)])
    vllm_config = make_vllm_config()
    with vllm.config.set_current_vllm_config(vllm_config):
        SplitCoalescingPass(vllm_config)(graph)

    graph.lint()
    assert sum(node.target == torch.ops.aten.slice.Tensor for node in graph.nodes) == 0
    assert (
        sum(
            node.target == torch.ops.aten.split_with_sizes.default
            for node in graph.nodes
        )
        == 1
    )


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 3, 1), (4, 8, 1)],  # gap
        [(0, 5, 1), (4, 8, 1)],  # overlap
        [(1, 4, 1), (4, 8, 1)],  # missing prefix
        [(0, 4, 1), (4, 7, 1)],  # missing suffix
        [(0, 8, 1)],  # single full-range slice
        [(0, 8, 2)],  # non-unit step
    ],
)
def test_slice_partition_rejects_unsafe_groups(ranges):
    graph = make_slice_graph(ranges)
    vllm_config = make_vllm_config()
    with vllm.config.set_current_vllm_config(vllm_config):
        SplitCoalescingPass(vllm_config)(graph)

    slice_count = sum(
        node.target == torch.ops.aten.slice.Tensor for node in graph.nodes
    )
    split_count = sum(
        node.target == torch.ops.aten.split_with_sizes.default for node in graph.nodes
    )
    assert slice_count == len(ranges)
    assert split_count == 0


def test_nested_slice_partitions_use_live_source():
    """Use the rewired source when a slice partition consumes another slice."""
    graph = torch.fx.Graph()
    source = graph.placeholder("input")
    source.meta["val"] = torch.empty((2, 8), device="meta")

    outer_left = graph.call_function(
        torch.ops.aten.slice.Tensor, args=(source, -1, 0, 4)
    )
    outer_left.meta["val"] = torch.empty((2, 4), device="meta")
    outer_right = graph.call_function(
        torch.ops.aten.slice.Tensor, args=(source, -1, 4, 8)
    )
    outer_right.meta["val"] = torch.empty((2, 4), device="meta")
    inner_left = graph.call_function(
        torch.ops.aten.slice.Tensor, args=(outer_left, -1, 0, 2)
    )
    inner_left.meta["val"] = torch.empty((2, 2), device="meta")
    inner_right = graph.call_function(
        torch.ops.aten.slice.Tensor, args=(outer_left, -1, 2, 4)
    )
    inner_right.meta["val"] = torch.empty((2, 2), device="meta")
    graph.output((inner_left, inner_right, outer_right))

    vllm_config = make_vllm_config()
    with vllm.config.set_current_vllm_config(vllm_config):
        SplitCoalescingPass(vllm_config)(graph)

    graph.lint()
    assert sum(node.target == torch.ops.aten.slice.Tensor for node in graph.nodes) == 0
    splits = [
        node
        for node in graph.nodes
        if node.target == torch.ops.aten.split_with_sizes.default
    ]
    assert len(splits) == 2

    outer_split = next(node for node in splits if list(node.args[1]) == [4, 4])
    inner_split = next(node for node in splits if list(node.args[1]) == [2, 2])
    assert outer_split.args[0] is source
    assert inner_split.args[0].target == operator.getitem
    assert inner_split.args[0].args == (outer_split, 0)


def test_split_coalescing_preserves_different_dimensions():
    """Keep equal-sized splits on different dimensions as separate nodes."""
    graph = torch.fx.Graph()
    source = graph.placeholder("input")
    source.meta["val"] = torch.empty((8, 8), device="meta")

    outputs = []
    for dim in (0, 1):
        split = graph.call_function(
            torch.ops.aten.split_with_sizes.default,
            args=(source, [4, 4], dim),
        )
        for index in range(2):
            outputs.append(graph.call_function(operator.getitem, args=(split, index)))
    graph.output(tuple(outputs))

    vllm_config = make_vllm_config()
    with vllm.config.set_current_vllm_config(vllm_config):
        SplitCoalescingPass(vllm_config)(graph)

    splits = [
        node
        for node in graph.nodes
        if node.target == torch.ops.aten.split_with_sizes.default
    ]
    assert len(splits) == 2
    assert all(split.args[0] is source for split in splits)
    assert {split.args[2] for split in splits} == {0, 1}
