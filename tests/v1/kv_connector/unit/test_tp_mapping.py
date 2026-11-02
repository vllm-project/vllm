# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NIXL source-rank selection and descriptor lists.

These tests verify that the NIXL worker derives source ranks, byte ranges and
descriptor lists from the KV cache specs. No GPU or NIXL required.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.kv_connector.unit.utils import region_pairs_by_index
from vllm.distributed.kv_transfer.kv_connector.utils import (
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

# ======================================================================
# Test fixtures / helpers
# ======================================================================

_ENGINE = "remote-engine"
_HEAD_SIZE = 8


@dataclass(frozen=True)
class _ShardedStateSpec(MambaSpec):
    """One state sharded across TP along its leading dim."""

    def _tp_segments(self) -> tuple[tuple[int, ...], ...]:
        return ((self.shapes[0][0],),)


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
    worker._regions_by_layer = {name: [i] for i, name in enumerate(layers)}
    worker.block_len_per_layer = [
        spec.page_size_bytes for spec in worker._layer_specs.values()
    ]
    worker.block_stride_per_layer = list(worker.block_len_per_layer)
    worker.region_num_blocks = [8] * len(layers)
    worker._physical_blocks_per_group = [1] * len(layers)
    worker._group_spec_types = tuple(
        type(spec) for spec in worker._layer_specs.values()
    )
    worker._dst_pairs = {}
    worker._byte_ranges = {}
    worker.source_ranks = {}
    worker._remote_block_splits = {}
    worker._test_layers = layers
    return worker


def _remote_meta(worker: NixlConnectorWorker, remote_tp_size: int) -> NixlAgentMetadata:
    """Metadata of a remote at `remote_tp_size` holding the worker's layers."""
    remote_specs = [
        replace(
            layer,
            shapes=tuple(
                (shape[0] * worker.transfer_tp_size // remote_tp_size, *shape[1:])
                for shape in layer.shapes
            ),
            page_size_padded=None,
        )
        if isinstance(layer, MambaSpec)
        else _spec(layer, remote_tp_size)
        for layer in worker._test_layers.values()
    ]
    return NixlAgentMetadata(
        engine_id=_ENGINE,
        agent_metadata=b"",
        kv_caches_base_addr=[0x10_000 * (i + 1) for i in range(len(remote_specs))],
        device_id=0,
        num_blocks=8,
        block_lens=[spec.page_size_bytes for spec in remote_specs],
        block_strides=[spec.page_size_bytes for spec in remote_specs],
        kv_cache_layout="LBHNC",
        block_size=4,
        attn_backend_name="",
        physical_blocks_per_group=[1] * len(remote_specs),
        region_members=worker.region_members,
        layer_num_kv_heads={
            name: spec.num_kv_heads
            for name, spec in zip(worker._layer_specs, remote_specs, strict=True)
            if isinstance(spec, AttentionSpec)
        },
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
    pairs = region_pairs_by_index(num_layers, range(num_layers))
    worker._dst_pairs[_ENGINE] = pairs
    ranges = worker._tp_byte_ranges(
        _remote_meta(worker, remote_tp_size),
        list(zip(pairs.local, pairs.group, pairs.remote)),
        remote_tp_size,
        remote_dcp_size,
        block_size_ratio,
        receiving=worker._TRANSFER_MODE == "pull",
    )
    worker._byte_ranges[_ENGINE] = ranges
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
        ranges[source] for ranges in worker._byte_ranges[_ENGINE]
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


@pytest.mark.parametrize(
    "tp_size,remote_tp_size,dcp_size,remote_dcp_size",
    [
        (4, 2, 4, 2),
        (4, 4, 4, 4),
        (4, 1, 4, 1),
        (4, 4, 1, 4),
        (2, 4, 2, 4),
    ],
)
def test_dcp_consumer_count_matches_readers(
    tp_size, remote_tp_size, dcp_size, remote_dcp_size
):
    """Each producer waits for exactly the local ranks that read from it."""
    workers = [
        _worker(tp_rank, tp_size, _MLA, dcp_size=dcp_size) for tp_rank in range(tp_size)
    ]
    sources = [_map(w, remote_tp_size, remote_dcp_size) for w in workers]

    for remote_rank in range(remote_tp_size):
        readers = [w for w, ranks in zip(workers, sources) if remote_rank in ranks]
        for worker in readers:
            consumers = worker.transfer_topo.dcp_consumer_count(
                remote_tp_size, remote_dcp_size
            )
            assert consumers == len(readers)


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
    worker._local_desc_lists, worker._local_descs = {}, {}
    worker._desc_starts = defaultdict(dict)
    worker.dst_xfer_side_handles = defaultdict(dict)
    worker.dst_region_num_blocks = {_ENGINE: [8] * num_layers}
    worker.dst_region_mem_types = {_ENGINE: ["VRAM"] * num_layers}
    meta = _remote_meta(worker, remote_tp_size)
    for rank in range(remote_tp_size):
        worker._prep_descs(meta, rank, "remote-agent", 1)
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
    local_ids, remote_ids = worker._compute_desc_ids(_ENGINE, 1, [[3, 5]], [[2, 4]])
    assert local_ids.tolist() == [8 + 3, 8 + 5]
    assert remote_ids.tolist() == [2, 4]


def test_mamba_state_is_one_more_region_of_chunked_pages():
    """MLA's replicated latent and a Mamba state sharded across P's ranks move
    through the same desc list: the latent whole from one rank, the state as
    one range per rank holding part of it."""
    mamba = _ShardedStateSpec(block_size=4, shapes=((400,),), dtypes=(torch.int8,))
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
    local_ids, remote_ids = worker._compute_desc_ids(_ENGINE, 1, [[], [6]], [[], [3]])
    assert local_ids.tolist() == [16 + 6]
    assert remote_ids.tolist() == [3]


def test_push_writes_replicated_latent_to_every_rank_holding_it():
    """Pushing from P at TP1 to D at TP2, the specs' sending-side slices write
    MLA's replicated latent whole to both D ranks and each its Mamba shard."""
    mamba = _ShardedStateSpec(block_size=4, shapes=((400,),), dtypes=(torch.int8,))
    worker = _worker(0, 1, {"attn": None, "mamba": mamba})
    worker._TRANSFER_MODE = "push"

    assert _map(worker, 2) == (0, 1)
    assert worker.source_ranks[_ENGINE] == [(0, 1), (0, 1)]
    latent_ranges, state_ranges = worker._byte_ranges[_ENGINE]
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


def test_smaller_remote_blocks_cannot_be_read_by_head():
    """Remote blocks smaller than ours move as whole sub-blocks, so blocks
    split across TP ranks cannot also split by block size."""
    worker = _worker(0, 1, _GQA)
    with pytest.raises(NotImplementedError, match="remote blocks are smaller"):
        _map(worker, 2, block_size_ratio=2)


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
    worker = object.__new__(NixlConnectorWorker)
    worker.world_size = local_tp
    worker._is_csa_linear = True
    worker.transfer_topo = SimpleNamespace(total_num_kv_heads=total_kv_heads)

    if compatible:
        worker._validate_csa_linear_tp_layout(remote_tp)
    else:
        with pytest.raises(ValueError, match="KV-head sharding boundary"):
            worker._validate_csa_linear_tp_layout(remote_tp)
