# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NIXL source-rank selection and descriptor lists.

These tests verify that the NIXL worker derives source ranks, byte ranges and
descriptor lists from the KV cache specs. No GPU or NIXL required.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.kv_connector.unit.utils import (
    region_pairs_by_index,
    remote_layer_spec,
    remote_stage,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.validation import (
    validate_csa_linear_tp_layout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ComponentSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

# ======================================================================
# Test fixtures / helpers
# ======================================================================

_ENGINE = "remote-engine"
_HEAD_SIZE = 8


def _sharded_state_spec(size: int = 400) -> MambaSpec:
    """One state of `size` bytes sharded across TP."""
    state = ComponentSpec(block_size=4, num_heads=size, state_content_bytes=1)
    return MambaSpec(
        block_size=4,
        shapes=((size,),),
        dtypes=(torch.int8,),
        component_specs=(("0", state),),
    )


def _spec(total_kv_heads: int | None, tp_size: int) -> AttentionSpec:
    """GQA spec of a rank at `tp_size`, or MLA's replicated latent (None)."""
    if total_kv_heads is None:
        return MLAAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=_HEAD_SIZE,
            dtype=torch.float16,
            max_tp_shards=1,
        )
    return AttentionSpec(
        block_size=4,
        num_kv_heads=max(1, total_kv_heads // tp_size),
        head_size=_HEAD_SIZE,
        dtype=torch.float16,
    )


def _worker(
    tp_rank: int,
    tp_size: int,
    layers: dict[str, KVCacheSpec | int | None],
    dcp_size: int = 1,
) -> NixlConnectorWorker:
    """Worker with one region and cache group per layer: a spec, or the total
    KV heads of a GQA layer (None: MLA)."""
    worker = object.__new__(NixlConnectorWorker)
    worker.transfer_tp_rank = tp_rank
    worker.transfer_tp_size = tp_size
    worker.transfer_topo = TransferTopology(
        tp_rank=tp_rank,
        tp_size=tp_size,
        block_size=4,
        engine_id="local-engine",
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=1,
        attn_backends=[],
        dcp_size=dcp_size,
    )
    worker._layer_specs = {
        name: layer if isinstance(layer, KVCacheSpec) else _spec(layer, tp_size)
        for name, layer in layers.items()
    }
    worker.kv_cache_config = SimpleNamespace(
        transfer_group_index_by_layer={name: i for i, name in enumerate(layers)}
    )
    worker.region_members = [[name] for name in layers]
    worker._region_by_layer = {name: i for i, name in enumerate(layers)}
    worker.block_len_per_layer = [
        spec.page_size_bytes for spec in worker._layer_specs.values()
    ]
    worker.block_stride_per_layer = list(worker.block_len_per_layer)
    worker.region_num_blocks = [8] * len(layers)
    worker._physical_blocks_per_group = [1] * len(layers)
    worker._group_spec_types = tuple(
        type(spec) for spec in worker._layer_specs.values()
    )
    worker.source_ranks = {}
    worker._remote_block_splits = {}
    worker.kv_cache_layout = "LBHNC"
    worker.use_host_buffer = False
    worker._layer_views = {}
    worker._test_layers = layers
    return worker


def _remote_meta(
    worker: NixlConnectorWorker, remote_tp_size: int, block_size: int = 4
) -> NixlAgentMetadata:
    """Metadata of a remote at `remote_tp_size` holding the worker's layers.
    Its views come from `fake_remote_layer_meta_tensors`."""
    heads = {
        name: _spec(layer, remote_tp_size).num_kv_heads
        for name, layer in worker._test_layers.items()
        if not isinstance(layer, MambaSpec)
    }
    worker._fake_remote_kv_heads = {_ENGINE: heads}
    block_lens = [
        remote_layer_spec(
            spec, worker.transfer_tp_size, remote_tp_size, heads.get(name)
        ).page_size_bytes
        * block_size
        // 4
        for name, spec in worker._layer_specs.items()
    ]
    return NixlAgentMetadata(
        engine_id=_ENGINE,
        agent_metadata=b"",
        kv_caches_base_addr=[0x10_000 * (i + 1) for i in range(len(block_lens))],
        device_id=0,
        num_blocks=8,
        block_lens=block_lens,
        block_strides=block_lens,
        kv_cache_layout="LBHNC",
        block_size=block_size,
        attn_backend_name="",
        physical_blocks_per_group=[1] * len(block_lens),
        region_members=worker.region_members,
    )


def _map(
    worker: NixlConnectorWorker,
    remote_tp_size: int,
    remote_dcp_size: int = 1,
    block_size_ratio: int = 1,
) -> tuple[int, ...]:
    """Map `worker` to a remote at `remote_tp_size` holding the same
    layers, as the handshake does, and return the ranks it reads from."""
    num_layers = len(worker.region_members)
    stage = remote_stage(worker, _ENGINE)
    stage.pairs = pairs = region_pairs_by_index(num_layers, range(num_layers))
    stage.byte_ranges = ranges = worker._tp_byte_ranges(
        _remote_meta(worker, remote_tp_size, 4 // block_size_ratio),
        list(zip(pairs.local, pairs.group)),
        remote_tp_size,
        remote_dcp_size,
        receiving=worker._TRANSFER_MODE == "pull",
    )
    # One cache group per layer.
    worker.source_ranks[_ENGINE] = [tuple(sorted(r)) for r in ranges]
    return tuple(sorted(set().union(*ranges)))


_GQA = {"model.layers.0.self_attn": 8}
_MLA = {"model.layers.0.self_attn": None}

# ======================================================================
# TP mapping structure tests
# ======================================================================


@pytest.mark.parametrize(
    "tp_rank,tp_size,remote_tp_size,expected_ranks",
    [(1, 2, 2, (1,)), (2, 4, 2, (1,)), (0, 1, 2, (0, 1))],
)
def test_gqa_source_ranks(tp_rank, tp_size, remote_tp_size, expected_ranks):
    worker = _worker(tp_rank, tp_size, _GQA)
    assert _map(worker, remote_tp_size) == expected_ranks


@pytest.mark.parametrize(
    "tp_rank,tp_size,remote_tp_size,draft_kv_heads,expected_head",
    [
        (3, 8, 1, 8, 3),  # one draft head per decode rank
        (5, 8, 2, 8, 1),  # rank 5 reads P rank 1, second head slot
        (5, 16, 1, 8, 2),  # D_TP > heads: two ranks share each head
    ],
)
def test_gqa_draft_under_mla_target_reads_its_heads(
    tp_rank, tp_size, remote_tp_size, draft_kv_heads, expected_head
):
    """Each region is sliced by its own spec: the MLA target's latent moves
    whole, while a GQA draft's rank reads its heads where the source holds
    them."""
    layers = {"model.layers.0.self_attn": None, "draft_model.layers.0": draft_kv_heads}
    worker = _worker(tp_rank, tp_size, layers)
    (source,) = _map(worker, remote_tp_size)
    mla_ranges, draft_ranges = (
        ranges[source] for ranges in remote_stage(worker, _ENGINE).byte_ranges
    )

    head_bytes = worker.block_len_per_layer[1]
    assert mla_ranges == [(0, 0, worker.block_len_per_layer[0])]
    assert draft_ranges == [(0, expected_head * head_bytes, head_bytes)]


@pytest.mark.parametrize(
    "tp_rank,tp_size,remote_tp_size,dcp_size,remote_dcp_size,expected_ranks",
    [
        (0, 1, 8, 1, 8, tuple(range(8))),
        (0, 4, 4, 1, 4, (0, 1, 2, 3)),
        (2, 4, 4, 4, 4, (2,)),
        (0, 2, 4, 2, 4, (0, 2)),
        (3, 4, 2, 4, 2, (1,)),
    ],
)
def test_mla_dcp_source_ranks(
    tp_rank,
    tp_size,
    remote_tp_size,
    dcp_size,
    remote_dcp_size,
    expected_ranks,
):
    worker = _worker(tp_rank, tp_size, _MLA, dcp_size=dcp_size)
    assert _map(worker, remote_tp_size, remote_dcp_size) == expected_ranks


def test_push_to_another_dcp_size_is_refused():
    """A push pairs the two ranks' blocks of a request in order, which is wrong
    unless both sides share the blocks out alike. It handshakes as if the
    remote had its DCP size, so a remote too small for that is refused first."""
    worker = _worker(0, 4, _MLA, dcp_size=4)
    worker._TRANSFER_MODE = "push"
    with pytest.raises(NotImplementedError, match="another DCP size"):
        _map(worker, remote_tp_size=4, remote_dcp_size=1)
    with pytest.raises(NotImplementedError, match="cannot have DCP size 4"):
        worker.transfer_topo.handshake_target_ranks(remote_tp_size=2, remote_dcp_size=4)


# Every (TP size, DCP size); a DCP group may be smaller than the TP size.
_TP_AND_DCP_SIZES = [
    (tp, dcp) for tp in (1, 2, 4, 8) for dcp in (1, 2, 4, 8) if tp % dcp == 0
]


@pytest.mark.parametrize("tp_size,dcp_size", _TP_AND_DCP_SIZES)
@pytest.mark.parametrize("remote_tp_size,remote_dcp_size", _TP_AND_DCP_SIZES)
def test_every_remote_rank_hears_from_the_ranks_it_waits_for(
    tp_size, dcp_size, remote_tp_size, remote_dcp_size
):
    """A producer rank frees a request's blocks once every consumer rank
    that handshook with it has notified it: every remote rank must be some
    local rank's target, and wait for exactly the ranks targeting it, which
    are the ranks reading from it if any does."""
    sizes = remote_tp_size, remote_dcp_size
    workers = [
        _worker(tp_rank, tp_size, _MLA, dcp_size=dcp_size) for tp_rank in range(tp_size)
    ]
    targets = [set(w.transfer_topo.handshake_target_ranks(*sizes)) for w in workers]
    sources = [set(_map(w, *sizes)) for w in workers]
    assert set().union(*targets) == set(range(remote_tp_size))
    for worker, ranks, read in zip(workers, targets, sources):
        consumers = worker.transfer_topo.dcp_consumer_count(*sizes)
        assert read <= ranks
        assert all(consumers == sum(r in other for other in targets) for r in ranks)
        assert all(consumers == sum(r in other for other in sources) for r in read)


@pytest.mark.parametrize("tp_size,dcp_size", _TP_AND_DCP_SIZES)
@pytest.mark.parametrize("remote_tp_size,remote_dcp_size", _TP_AND_DCP_SIZES)
@pytest.mark.parametrize("num_blocks,num_computed_blocks", [(1, 0), (7, 0), (12, 2)])
def test_pull_reads_each_block_once_from_a_rank_holding_it(
    tp_size, dcp_size, remote_tp_size, remote_dcp_size, num_blocks, num_computed_blocks
):
    """Whatever the TP and DCP sizes of the two sides, every rank reads each
    block it lacks of a request exactly once, from a remote rank that holds
    the block at the same position in the request."""
    # Block ids are a rank's index into its own blocks of the request: a DCP
    # rank holds every `dcp_size`-th block from its `dcp_rank`-th.
    num_remote_ids = cdiv(num_blocks, remote_dcp_size)
    for tp_rank in range(tp_size):
        worker = _worker(tp_rank, tp_size, _MLA, dcp_size=dcp_size)
        worker.dcp_size, worker.dcp_rank = dcp_size, tp_rank % dcp_size
        worker._remote_physical_blocks_per_group = {_ENGINE: [1]}
        worker.transfer_topo.register_remote_engine(
            _ENGINE,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_len=worker.block_len_per_layer[0],
                remote_block_size=4,
                remote_physical_blocks_per_logical=1,
                remote_dcp_size=remote_dcp_size,
            ),
        )
        local_ids = range(num_computed_blocks, cdiv(num_blocks, dcp_size))
        dcp_active = dcp_size > 1 or remote_dcp_size > 1

        read: Counter[int] = Counter()
        for remote_rank in _map(worker, remote_tp_size, remote_dcp_size):
            local, remote, _ = worker._pair_blocks(
                _ENGINE,
                remote_rank,
                0,
                list(local_ids),
                0,
                list(range(100, 100 + num_remote_ids)),
                # Without DCP the pull counts back from the last token.
                num_computed_blocks=num_computed_blocks if dcp_active else None,
            )
            positions = [worker.dcp_rank + i * dcp_size for i in local]
            assert positions == [
                remote_rank % remote_dcp_size + (i - 100) * remote_dcp_size
                for i in remote
            ]
            read.update(positions)

        lacked = [worker.dcp_rank + i * dcp_size for i in local_ids]
        assert all(count == 1 for count in read.values())
        # Blocks allocated past the last token may be left behind.
        assert {p for p in lacked if p < num_blocks} <= set(read) <= set(lacked)


# ======================================================================
# Descriptor list tests
# ======================================================================


class _DescLists:
    """Records the desc lists a worker prepares."""

    def __init__(self):
        self.lists: list[list[list[int]]] = []

    def prep_xfer_dlist(self, agent, descs, mem_type=None):
        self.lists.append(np.asarray(descs).tolist())
        return len(self.lists) - 1


def _prep(worker: NixlConnectorWorker, remote_tp_size: int) -> _DescLists:
    """Prepare the worker's desc lists for every rank of the remote."""
    num_layers = len(worker.region_members)
    worker.engine_id, worker.tp_rank, worker.device_id = "local-engine", 0, 0
    worker.nixl_wrapper = _DescLists()
    worker.kv_caches_base_addr = {
        worker.engine_id: {0: [0x1000 * (i + 1) for i in range(num_layers)]}
    }
    worker._mixed_mem_types = False
    worker.region_mem_types = ["VRAM"] * num_layers
    worker._local_desc_lists = {}
    worker.nixl_memory_type = "VRAM"
    stage = remote_stage(worker, _ENGINE)
    stage.region_num_blocks = [8] * num_layers
    meta = _remote_meta(worker, remote_tp_size)
    for rank in range(remote_tp_size):
        worker._prep_descs(meta, rank, "remote-agent", 1, stage)
    return worker.nixl_wrapper


def test_head_slices_are_strided_descs_of_one_local_list():
    """D at TP1 reads half its heads from each of two P ranks: one local list
    holds a strided desc per (rank, range), each moving every block."""
    worker = _worker(0, 1, _GQA)
    assert _map(worker, 2) == (0, 1)
    block = worker.block_len_per_layer[0]
    half = block // 2

    lists = _prep(worker, 2).lists
    local, remote_0, remote_1 = lists
    assert local == [[0x1000, half, 0, block, 8], [0x1000 + half, half, 0, block, 8]]
    assert remote_0 == remote_1 == [[0x10_000, half, 0, half, 8]]
    local_ids, remote_ids = worker._compute_desc_ids(
        remote_stage(worker, _ENGINE), 1, [[3, 5]], [[2, 4]]
    )
    assert local_ids.tolist() == [8 + 3, 8 + 5]
    assert remote_ids.tolist() == [2, 4]


def test_mamba_state_is_one_more_region_of_chunked_pages():
    """MLA's replicated latent and a Mamba state sharded across P's ranks move
    through the same desc list: the latent whole from one rank, the state as
    one range per rank holding part of it."""
    mamba = _sharded_state_spec()
    worker = _worker(0, 1, {"attn": None, "mamba": mamba})
    assert _map(worker, 2) == (0, 1)
    assert worker.source_ranks[_ENGINE] == [(0,), (0, 1)]

    latent = worker.block_len_per_layer[0]
    local, *_ = _prep(worker, 2).lists
    assert local == [
        [0x1000, latent, 0, latent, 8],
        [0x2000, 200, 0, 400, 8],
        [0x2000 + 200, 200, 0, 400, 8],
    ]
    local_ids, remote_ids = worker._compute_desc_ids(
        remote_stage(worker, _ENGINE), 1, [[], [6]], [[], [3]]
    )
    assert local_ids.tolist() == [16 + 6]
    assert remote_ids.tolist() == [3]


def test_push_writes_replicated_latent_to_every_rank_holding_it():
    """Pushing from P at TP1 to D at TP2, the specs' sending-side slices write
    MLA's replicated latent whole to both D ranks and each its Mamba shard."""
    mamba = _sharded_state_spec()
    worker = _worker(0, 1, {"attn": None, "mamba": mamba})
    worker._TRANSFER_MODE = "push"

    assert _map(worker, 2) == (0, 1)
    assert worker.source_ranks[_ENGINE] == [(0, 1), (0, 1)]
    latent_ranges, state_ranges = remote_stage(worker, _ENGINE).byte_ranges
    latent = worker.block_len_per_layer[0]
    assert latent_ranges == {0: [(0, 0, latent)], 1: [(0, 0, latent)]}
    assert state_ranges == {0: [(0, 0, 200)], 1: [(200, 0, 200)]}


@pytest.mark.parametrize(
    ("layers", "p_tp", "d_tp"),
    [
        (_MLA, 4, 2),
        (_MLA, 2, 4),
        (_GQA, 4, 2),
        ({"model.layers.0.self_attn": 2}, 4, 2),
    ],
)
def test_push_decode_waits_for_the_ranks_writing_it(layers, p_tp, d_tp):
    """D counts a push done after one notif per P rank writing to it: the
    specs' receiving side must count exactly the P ranks whose sending side
    writes it, one per replicated head."""
    writes: Counter[int] = Counter()
    for p_rank in range(p_tp):
        worker = _worker(p_rank, p_tp, layers)
        worker._TRANSFER_MODE = "push"
        writes.update(_map(worker, d_tp))
    for d_rank in range(d_tp):
        worker = _worker(d_rank, d_tp, layers)
        worker.nixl_wrapper = MagicMock()
        worker._push_senders = defaultdict(dict)
        worker._add_notif_only_remote_agent(_remote_meta(worker, p_tp), 0, p_tp)
        assert worker._push_senders[_ENGINE] == {0: writes[d_rank]}


def test_shards_one_remote_rank_holds_together_move_as_one_range():
    """12 KV heads at TP 2 and TP 3 are cut 6 ways. A TP 3 rank holds two of
    our three shards, next to each other on both sides: one desc, not two."""
    worker = _worker(0, 2, {"model.layers.0.self_attn": 12})
    assert _map(worker, 3) == (0, 1)
    block = worker.block_len_per_layer[0]
    shard = block // 3
    assert remote_stage(worker, _ENGINE).byte_ranges == [
        {0: [(0, 0, shard), (shard, shard, shard)], 1: [(2 * shard, 0, shard)]}
    ]
    local, *_ = _prep(worker, 3).lists
    assert local == [
        [0x1000, 2 * shard, 0, block, 8],
        [0x1000 + 2 * shard, shard, 0, block, 8],
    ]


def test_smaller_remote_blocks_are_read_by_head():
    """Each remote rank fills its heads of every smaller block of ours."""
    worker = _worker(0, 1, _GQA)
    assert _map(worker, 2, block_size_ratio=2) == (0, 1)
    shard = worker.block_len_per_layer[0] // 4
    assert remote_stage(worker, _ENGINE).byte_ranges == [
        {0: [(0, 0, shard)], 1: [(shard, 0, shard)]}
    ]


def test_smaller_remote_blocks_fill_a_rank_with_fewer_heads():
    """A rank holding half the remote's heads reads them into whole
    sub-blocks of its own, as on main."""
    worker = _worker(1, 2, _GQA)
    meta = _remote_meta(worker, 1, block_size=2)
    ranges = worker._tp_byte_ranges(meta, [(0, 0)], 1, 1, receiving=True)
    sub_block = worker.block_len_per_layer[0] // 2
    assert ranges == [{0: [(0, sub_block, sub_block)]}]


def test_padded_state_page_moves_without_its_padding():
    """As on main, a recurrent state page moves its states, not the page
    padding after them."""
    mamba = replace(_sharded_state_spec(), page_size_padded=1024)
    worker = _worker(0, 1, {"mamba": mamba})
    assert _map(worker, 1) == (0,)
    assert remote_stage(worker, _ENGINE).byte_ranges == [{0: [(0, 0, 400)]}]


@pytest.mark.parametrize(
    ("total_kv_heads", "local_tp", "remote_tp", "compatible"),
    [
        # 2 heads: TP1 packs both heads into a page, TP >= 2 packs one.
        (2, 1, 2, False),
        (2, 2, 1, False),
        (2, 1, 1, True),
        (2, 2, 4, True),
        (2, 4, 2, True),
        # 4 heads: the boundary moves with the head count.
        (4, 2, 4, False),
        (4, 2, 2, True),
        (4, 4, 8, True),
    ],
)
def test_csa_linear_tp_layout_boundary(total_kv_heads, local_tp, remote_tp, compatible):
    if compatible:
        validate_csa_linear_tp_layout(local_tp, remote_tp, total_kv_heads)
    else:
        with pytest.raises(ValueError, match="KV-head sharding boundary"):
            validate_csa_linear_tp_layout(local_tp, remote_tp, total_kv_heads)
