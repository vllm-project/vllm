# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE heterogeneous-TP mapping and head/state slicing."""

import itertools
import threading
from collections import OrderedDict

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import ROLE
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    HeadLayout,
    MambaStateLayout,
    compute_head_slice_offsets,
    compute_mamba_slice_offsets,
    get_layer_head_layout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    _rank_heads,
    compute_write_tp_mapping,
)
from vllm.v1.kv_cache_interface import AttentionSpec, compute_layout_strides
from vllm.v1.kv_cache_layout import KVCacheLayout

pytestmark = pytest.mark.cpu_test


def _targets(producer_tp_rank, producer_tp_size, consumer_tp_size, heads, is_mla):
    mapping = compute_write_tp_mapping(
        producer_tp_rank, producer_tp_size, consumer_tp_size, heads, is_mla, False
    )
    return [
        (t.tp_rank, t.local_head, t.remote_head, t.num_heads, t.num_writers)
        for t in mapping.targets
    ]


@pytest.mark.parametrize(
    ("producer", "consumer", "heads", "is_mla", "expected"),
    [
        # Equal TP: one-to-one.
        ((1, 2), 2, 8, False, [(1, 0, 0, 4, 1)]),
        # MLA fan-out: the latent goes whole to every target.
        ((1, 2), 4, 1, True, [(2, 0, 0, 1, 1), (3, 0, 0, 1, 1)]),
        # Split GQA fan-out: each target gets its 2 of the producer's 4 heads.
        ((1, 2), 4, 8, False, [(2, 0, 0, 2, 1), (3, 2, 0, 2, 1)]),
        # Replicated GQA fan-out: 2 heads over 2 -> 4 ranks, a head per pair.
        ((1, 2), 4, 2, False, [(2, 0, 0, 1, 1), (3, 0, 0, 1, 1)]),
        # Producer split, consumer replicated: 4 heads, TP2 -> TP8.
        (
            (0, 2),
            8,
            4,
            False,
            [(0, 0, 0, 1, 1), (1, 0, 0, 1, 1), (2, 1, 0, 1, 1), (3, 1, 0, 1, 1)],
        ),
        # Split GQA fan-in: producer 3 of TP4 fills heads 2..3 of decode rank 1.
        ((3, 4), 2, 8, False, [(1, 0, 2, 2, 2)]),
        # Replicated fan-in: TP8 producers share heads; only first holders write.
        ((0, 8), 2, 2, False, [(0, 0, 0, 1, 1)]),
        ((1, 8), 2, 2, False, []),
        # MLA fan-in: one producer per consumer writes the latent.
        ((2, 4), 2, 1, True, [(1, 0, 0, 1, 1)]),
        ((3, 4), 2, 1, True, []),
    ],
)
def test_write_tp_mapping_targets(producer, consumer, heads, is_mla, expected):
    producer_tp_rank, producer_tp_size = producer
    assert (
        _targets(producer_tp_rank, producer_tp_size, consumer, heads, is_mla)
        == expected
    )


def test_mla_fan_in_with_mamba_still_writes_state_from_every_producer():
    mapping = compute_write_tp_mapping(3, 4, 2, 1, True, True)
    (target,) = mapping.targets
    assert (target.tp_rank, target.num_heads, target.num_writers) == (1, 0, 2)
    assert (target.remote_slice, target.remote_slices) == (1, 2)


@pytest.mark.parametrize(
    ("producer_tp", "consumer_tp", "heads", "is_mla", "match"),
    [
        (2, 3, 8, False, "multiple of the other"),
        (4, 6, 1, True, "multiple of the other"),
        (2, 4, 6, False, "cannot shard 6 KV heads over tp_size=4"),
        (8, 2, 3, False, "cannot shard 3 KV heads over tp_size=8"),
    ],
)
def test_write_tp_mapping_rejects_unmappable_topologies(
    producer_tp, consumer_tp, heads, is_mla, match
):
    with pytest.raises(ValueError, match=match):
        compute_write_tp_mapping(0, producer_tp, consumer_tp, heads, is_mla, False)


_TP_SIZES = (1, 2, 4, 8)
_TOPOLOGIES = [
    (p, d, h, mla, mamba)
    for p, d in itertools.product(_TP_SIZES, _TP_SIZES)
    for h in (1, 2, 4, 8, 16)
    for mla in (False, True)
    for mamba in (False, True)
    if (mla and h == 1)
    or (
        not mla
        and all((t <= h and h % t == 0) or (t > h and t % h == 0) for t in (p, d))
    )
]


@pytest.mark.parametrize(
    ("producer_tp", "consumer_tp", "heads", "mla", "mamba"), _TOPOLOGIES
)
def test_every_consumer_head_and_state_slice_is_written_exactly_once(
    producer_tp, consumer_tp, heads, mla, mamba
):
    """Across all producers, each consumer rank receives every global head it
    holds exactly once, from the producer holding that head, plus every state
    slice exactly once, and expects as many write_done as producers write."""
    heads_written: dict[int, list[int]] = {d: [] for d in range(consumer_tp)}
    slices_written: dict[int, list[int]] = {d: [] for d in range(consumer_tp)}
    writers: dict[int, set[int]] = {d: set() for d in range(consumer_tp)}
    advertised: dict[int, set[int]] = {d: set() for d in range(consumer_tp)}
    for p in range(producer_tp):
        mapping = compute_write_tp_mapping(
            p, producer_tp, consumer_tp, heads, mla, mamba
        )
        p_heads = _rank_heads(p, producer_tp, 1 if mla else heads)
        for t in mapping.targets:
            d_heads = _rank_heads(t.tp_rank, consumer_tp, 1 if mla else heads)
            for i in range(t.num_heads):
                # The global head sent is the global head it lands on.
                assert p_heads.start + t.local_head + i == (
                    d_heads.start + t.remote_head + i
                )
                heads_written[t.tp_rank].append(d_heads.start + t.remote_head + i)
            if mamba:
                assert t.local_slices == 1 or t.remote_slices == 1
                slices_written[t.tp_rank].append(t.remote_slice)
            writers[t.tp_rank].add(p)
            advertised[t.tp_rank].add(t.num_writers)

    for d in range(consumer_tp):
        d_heads = _rank_heads(d, consumer_tp, 1 if mla else heads)
        assert sorted(heads_written[d]) == list(d_heads)
        if mamba:
            ratio = max(1, producer_tp // consumer_tp)
            assert sorted(slices_written[d]) == list(range(ratio))
        assert advertised[d] == {len(writers[d])}


def _view(buffer: torch.Tensor, spec, num_blocks, layout) -> torch.Tensor:
    """A byte view [B, H, N, C] of ``buffer`` laid out as ``layout``."""
    strides = compute_layout_strides(spec, num_blocks, 1, layout)[1:]
    shape = (num_blocks, spec.num_heads, spec.num_states, spec.state_content_size_bytes)
    return torch.as_strided(buffer, shape, strides)


def _spec(num_kv_heads: int) -> AttentionSpec:
    return AttentionSpec(
        block_size=4, num_kv_heads=num_kv_heads, head_size=2, dtype=torch.bfloat16
    )


def _apply(offsets, src: torch.Tensor, dst: torch.Tensor) -> None:
    for local, remote, size in zip(*offsets):
        dst[remote : remote + size] = src[local : local + size]


_LAYOUTS = (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)


@pytest.mark.parametrize(
    ("local_layout", "remote_layout"), list(itertools.product(_LAYOUTS, _LAYOUTS))
)
@pytest.mark.parametrize(("producer_tp", "consumer_tp"), [(1, 2), (2, 4), (2, 1)])
def test_head_slices_land_byte_exact(
    local_layout, remote_layout, producer_tp, consumer_tp
):
    total_heads, num_blocks = 4, 3
    torch.manual_seed(0)
    full = torch.randint(0, 255, (num_blocks, total_heads, 4, 8), dtype=torch.uint8)
    local_blocks, remote_blocks = [0, 2], [2, 1]
    expected = {}
    received = {}
    consumers = {}
    for d in range(consumer_tp):
        spec = _spec(total_heads // consumer_tp)
        buffer = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.uint8)
        consumers[d] = (buffer, _view(buffer, spec, num_blocks, remote_layout), spec)
    for p in range(producer_tp):
        spec = _spec(total_heads // producer_tp)
        buffer = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.uint8)
        view = _view(buffer, spec, num_blocks, local_layout)
        view.copy_(full[:, _rank_heads(p, producer_tp, total_heads)])
        local = get_layer_head_layout("l", view, {"l": spec})
        mapping = compute_write_tp_mapping(
            p, producer_tp, consumer_tp, total_heads, False, False
        )
        for t in mapping.targets:
            remote_buffer, remote_view, remote_spec = consumers[t.tp_rank]
            remote = get_layer_head_layout("l", remote_view, {"l": remote_spec})
            offsets = compute_head_slice_offsets(
                local,
                HeadLayout.from_wire(remote.to_wire()),
                local_blocks,
                remote_blocks,
                t.local_head,
                t.remote_head,
                t.num_heads,
            )
            _apply(offsets, buffer, remote_buffer)
    for d, (_, remote_view, _) in consumers.items():
        d_heads = _rank_heads(d, consumer_tp, total_heads)
        expected[d] = full[local_blocks][:, d_heads]
        received[d] = remote_view[remote_blocks]
        assert torch.equal(received[d], expected[d])
        # Blocks nobody wrote stay untouched.
        assert not remote_view[0].any()


def test_head_slices_of_separate_kv_planes():
    """Legacy [2, num_blocks, block_size, heads, head_dim] caches move K and V."""
    spec = _spec(4)
    torch.manual_seed(0)
    local = torch.randint(0, 255, (2, 3, 4, 4, 4), dtype=torch.uint8)
    remote = torch.zeros((2, 3, 4, 2, 4), dtype=torch.uint8)
    local_layout = get_layer_head_layout("l", local, {"l": spec})
    remote_layout = get_layer_head_layout("l", remote, {"l": _spec(2)})
    offsets = compute_head_slice_offsets(
        local_layout,
        remote_layout,
        [1],
        [2],
        local_head=2,
        remote_head=0,
        num_heads=2,
    )
    _apply(offsets, local.view(-1), remote.view(-1))
    assert torch.equal(remote[:, 2], local[:, 1, :, 2:4])
    assert not remote[:, :2].any()


def test_token_major_head_slices_merge_to_one_run_per_token():
    spec = _spec(4)
    buffer = torch.zeros(spec.page_size_bytes * 2, dtype=torch.uint8)
    local = get_layer_head_layout(
        "l", _view(buffer, spec, 2, KVCacheLayout.LBNHC), {"l": spec}
    )
    remote_spec = _spec(2)
    remote_buffer = torch.zeros(remote_spec.page_size_bytes * 2, dtype=torch.uint8)
    remote = get_layer_head_layout(
        "l",
        _view(remote_buffer, remote_spec, 2, KVCacheLayout.LBNHC),
        {"l": remote_spec},
    )
    _, _, sizes = compute_head_slice_offsets(local, remote, [0], [1], 2, 0, 2)
    # Two heads of 8 bytes for each of the block's 4 tokens.
    assert sizes == [16] * 4


def _mamba_layout(proj, ssm, conv_stride, ssm_stride):
    return MambaStateLayout(conv_stride, ssm_stride, ssm, tuple(proj))


def _fill(layout: MambaStateLayout, slots, num_slots, seed):
    torch.manual_seed(seed)
    conv = torch.zeros(layout.conv_slot_stride * num_slots, dtype=torch.uint8)
    ssm = torch.zeros(layout.ssm_slot_stride * num_slots, dtype=torch.uint8)
    for slot in slots:
        base = slot * layout.conv_slot_stride
        n = sum(layout.conv_proj_bytes)
        conv[base : base + n] = torch.randint(1, 255, (n,), dtype=torch.uint8)
        base = slot * layout.ssm_slot_stride
        ssm[base : base + layout.ssm_bytes] = torch.randint(
            1, 255, (layout.ssm_bytes,), dtype=torch.uint8
        )
    return conv, ssm


def _move_state(offsets, src, dst):
    local, remote, sizes, n_conv = offsets
    _apply((local[:n_conv], remote[:n_conv], sizes[:n_conv]), src[0], dst[0])
    _apply((local[n_conv:], remote[n_conv:], sizes[n_conv:]), src[1], dst[1])


@pytest.mark.parametrize("replicated_bc", [False, True])
def test_mamba_state_fans_out_and_gathers_back(replicated_bc):
    # x, B, C sub-projections and the SSM state, TP1 vs TP2 pages.
    big = _mamba_layout((8, 4, 4), 16, conv_stride=40, ssm_stride=24)
    small = _mamba_layout(
        (4, 4, 4) if replicated_bc else (4, 2, 2), 8, conv_stride=20, ssm_stride=12
    )
    full = _fill(big, [1], 3, seed=0)

    # Fan-out: one TP1 producer slot 1 -> slot 2 of both TP2 consumers.
    shards = []
    for j in range(2):
        shard = (
            torch.zeros(20 * 3, dtype=torch.uint8),
            torch.zeros(12 * 3, dtype=torch.uint8),
        )
        offsets = compute_mamba_slice_offsets(big, small, [1], [2], j, 2, 0, 1)
        _move_state(offsets, full, shard)
        shards.append(shard)
    for j, (conv, ssm) in enumerate(shards):
        x = full[0][40:48].view(2, 4)[j]
        b = full[0][48:52] if replicated_bc else full[0][48:52].view(2, 2)[j]
        c = full[0][52:56] if replicated_bc else full[0][52:56].view(2, 2)[j]
        assert torch.equal(conv[40 : 40 + 4], x)
        assert torch.equal(conv[44 : 44 + b.numel()], b)
        assert torch.equal(conv[44 + b.numel() : 44 + 2 * b.numel()], c)
        assert torch.equal(ssm[24 : 24 + 8], full[1][24:40].view(2, 8)[j])

    # Fan-in: both TP2 producers write their slot 2 back into one TP1 slot 0.
    gathered = (
        torch.zeros(40 * 3, dtype=torch.uint8),
        torch.zeros(24 * 3, dtype=torch.uint8),
    )
    for j, shard in enumerate(shards):
        offsets = compute_mamba_slice_offsets(small, big, [2], [0], 0, 1, j, 2)
        _move_state(offsets, shard, gathered)
    assert torch.equal(gathered[0][:16], full[0][40:56])
    assert torch.equal(gathered[1][:16], full[1][24:40])


def test_mamba_state_rejects_mismatched_shards():
    big = _mamba_layout((8, 4, 4), 16, 40, 24)
    with pytest.raises(ValueError, match="cannot map"):
        compute_mamba_slice_offsets(
            big, _mamba_layout((3, 2, 2), 8, 20, 12), [0], [0], 0, 2, 0, 1
        )


def _consumer_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_write_cache_req_ids = []
    wrapper.failed_write_cache_req_ids = []
    wrapper._write_done_counts = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def test_consumer_completes_after_every_writer_reported(monkeypatch):
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    wrapper = _consumer_wrapper()
    message = {"type": "write_done", "transfer_id": "t", "num_writers": 2}

    wrapper._handle_structured_message(message)
    assert wrapper.pop_finished_write_req_ids() == set()

    wrapper._handle_structured_message(message)
    assert wrapper.pop_finished_write_req_ids() == {"t"}
    assert wrapper._write_done_counts == {}

    # A peer that does not send num_writers is a single writer.
    wrapper._handle_structured_message({"type": "write_done", "transfer_id": "u"})
    assert wrapper.pop_finished_write_req_ids() == {"u"}


def test_consumer_write_failure_drops_partial_write_count(monkeypatch):
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    wrapper = _consumer_wrapper()
    wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "t", "num_writers": 2}
    )
    wrapper._handle_structured_message({"type": "write_failed", "transfer_id": "t"})
    assert wrapper._write_done_counts == {}
    assert wrapper.pop_failed_write_req_ids() == {"t"}
