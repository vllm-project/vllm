# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host dispatch must preserve boundaries and separate execution phases."""

import itertools

import pytest

from vllm.kernel_buckets import (
    KernelPhase,
    MBucketTable,
    MRange,
    PhaseBucketTable,
    m_in_range,
    phase_from_counts,
    select_nearest_m,
)


def test_intervals_cover_boundaries_without_rounding_m():
    table = MBucketTable((1, 17, 1025, 4097), ("empty", "a", "b", "c", "d"))
    for m in (0, 1, 16, 17, 1024, 1025, 4096, 4097, 10000):
        interval = table.interval(m)
        assert interval.contains(m)
        assert table.select(m) == (
            "empty"
            if m == 0
            else "a"
            if m <= 16
            else "b"
            if m <= 1024
            else "c"
            if m <= 4096
            else "d"
        )
        assert m_in_range(m, start=interval.start, stop=interval.stop)
        if interval.start:
            assert not interval.contains(interval.start - 1)
        if interval.stop is not None:
            assert not interval.contains(interval.stop)


def test_point_conversion_merges_runs_but_retains_unmeasured_holes():
    points = {1: "small", 2: "small", 4: "large", 5: "large", 9: "small"}
    table = MBucketTable.from_points(points, "fallback")
    assert table.interval(1) == MRange(1, 3)
    assert table.interval(4) == MRange(4, 6)
    assert table.interval(9) == MRange(9, 10)
    for m in range(20):
        assert table.select(m) == points.get(m, "fallback")


def test_nearest_conversion_preserves_unsorted_insertion_ties():
    for rows in itertools.permutations((0, 3, 8, 16)):
        points = {m: str(m) for m in rows}
        table = MBucketTable.from_nearest(points)
        for m in range(40):
            expected = points[min(points, key=lambda tuned_m: abs(tuned_m - m))]
            assert table.select(m) == expected
            assert select_nearest_m(points, m) == expected


def test_same_m_uses_independent_prefill_and_decode_implementations():
    fallback = MBucketTable((), ("legacy",))
    policy = PhaseBucketTable(
        "test.operator",
        fallback=fallback,
        prefill=MBucketTable((1025,), ("prefill_a", "prefill_b")),
        decode=MBucketTable((17,), ("decode_a", "decode_b")),
    )
    assert policy.select(16, phase=KernelPhase.PREFILL) == "prefill_a"
    assert policy.select(16, phase=KernelPhase.DECODE) == "decode_a"
    assert policy.select(1025, phase=KernelPhase.PREFILL) == "prefill_b"
    # Mixed/verify/unknown do not inherit decode-only implementations.
    for phase in (KernelPhase.MIXED, KernelPhase.VERIFY, KernelPhase.UNKNOWN):
        assert policy.select(16, phase=phase) == "legacy"
    prefill_key = policy.key(16, phase=KernelPhase.PREFILL, specialization=("sm90",))
    decode_key = policy.key(16, phase=KernelPhase.DECODE, specialization=("sm90",))
    assert prefill_key != decode_key
    assert prefill_key == policy.key(
        17, phase=KernelPhase.PREFILL, specialization=("sm90",)
    )
    assert decode_key != policy.key(
        17, phase=KernelPhase.DECODE, specialization=("sm90",)
    )
    with pytest.raises(ValueError, match="known execution phase"):
        policy.key(16, phase=KernelPhase.UNKNOWN, specialization=())


@pytest.mark.parametrize(
    "prefill,decode,draft,expected",
    [
        (1, 0, False, KernelPhase.PREFILL),
        (0, 4096, False, KernelPhase.DECODE),
        (4096, 1, False, KernelPhase.MIXED),
        (0, 16, True, KernelPhase.VERIFY),
        (1, 1, True, KernelPhase.MIXED),
        (0, 0, False, KernelPhase.UNKNOWN),
    ],
)
def test_execution_metadata_defines_phase(prefill, decode, draft, expected):
    assert (
        phase_from_counts(
            prefill_tokens=prefill, decode_tokens=decode, has_draft_tokens=draft
        )
        == expected
    )


@pytest.mark.parametrize(
    "stops,values", [((17, 16), (1, 2, 3)), ((0,), (1, 2)), ((17,), (1,))]
)
def test_invalid_partitions_fail_at_construction(stops, values):
    with pytest.raises(ValueError):
        MBucketTable(stops, values)


def test_negative_m_is_rejected_by_table_selection():
    with pytest.raises(ValueError, match="nonnegative"):
        MBucketTable((), ("legacy",)).select(-1)


def test_compiled_cond_switches_at_runtime_bucket_boundary():
    """The shared predicate must preserve torch.cond's dynamic M branch."""
    import torch

    captured = []

    def backend(graph_module, _example_inputs):
        captured.append(graph_module)
        return graph_module.forward

    def dispatch(x):
        return torch.cond(
            m_in_range(x.shape[0], stop=17),
            lambda value: value + 1,
            lambda value: value - 1,
            (x,),
        )

    compiled = torch.compile(dispatch, backend=backend, fullgraph=True, dynamic=True)
    for m in (16, 17, 4, 1025):
        x = torch.zeros(m, 2)
        assert torch.equal(compiled(x), x + (1 if m <= 16 else -1))
    assert any(
        node.target is torch.ops.higher_order.cond
        for graph in captured
        for node in graph.graph.nodes
    )


def test_agent_key_preserves_original_multidimensional_buckets():
    """Equal row buckets must not collapse existing N/K bucket decisions."""
    policy = PhaseBucketTable(
        "test.operator",
        fallback=MBucketTable((65,), ("small", "large")),
        axis="M * num_kv_heads",
    )
    n_bucket = ("N", MRange(128, 257))
    k_bucket = ("K", MRange(4096, 8192))
    key = policy.key(
        32,
        phase=KernelPhase.PREFILL,
        specialization=("sm90", "bf16", n_bucket, k_bucket),
    )
    assert key.axis == "M * num_kv_heads"
    assert key != policy.key(
        32,
        phase=KernelPhase.PREFILL,
        specialization=("sm90", "bf16", ("N", MRange(257, None)), k_bucket),
    )
    assert key != policy.key(
        32,
        phase=KernelPhase.PREFILL,
        specialization=("sm90", "bf16", n_bucket, ("K", MRange(8192, None))),
    )


def test_current_phase_selection_keeps_distinct_phase_implementations():
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.forward_context import set_forward_context

    legacy = MBucketTable((), ("legacy",))
    policy = PhaseBucketTable(
        "test.operator",
        legacy,
        prefill=MBucketTable((), ("prefill",)),
        decode=MBucketTable((), ("decode",)),
    )
    shared = PhaseBucketTable("test.shared", legacy, prefill=legacy, decode=legacy)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    for phase in KernelPhase:
        with set_forward_context(None, config, kernel_phase=phase):
            assert policy.select_current(16) == policy.select(16, phase=phase)
            assert shared.select_current(16) == "legacy"


def test_compiled_table_selection_preserves_dynamic_range_guards():
    import torch

    table = MBucketTable((17, 65), (1, 2, 3))

    def dispatch(x):
        return x + table.select(x.shape[0])

    compiled = torch.compile(dispatch, backend="eager", dynamic=True, fullgraph=True)
    for m in (16, 17, 64, 65, 4):
        x = torch.zeros(m, 2)
        assert torch.equal(compiled(x), x + table.select(m))
