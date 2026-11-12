# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential tests: the Rust Mooncake store hot paths (`vllm._rust_core`)
must return exactly what the Python coordinator and worker lookup return."""

import random
import threading
from math import lcm
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (
    BlobBlockHashes,
    ChunkedTokenDatabase,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (
    KeyMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

rust_core = pytest.importorskip("vllm._rust_core")

pytestmark = pytest.mark.cpu_test

NUM_SEEDS = 100
# Hits that only a missing earlier Store chunk of a page shortens are rare.
NUM_LOOKUP_SEEDS = 300


def _full(block_size):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=torch.float16
    )


def _swa(block_size, sliding_window, extra_retained_tokens=0):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
        sliding_window=sliding_window,
        extra_retained_tokens=extra_retained_tokens,
    )


def _mamba(block_size):
    return MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _dsv41_groups():
    """DeepSeek-V4.1: five bounded-replay SWA MLA groups, the compressed MLA
    full-attention group and the compressor's circular buffer."""
    swa = SlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
    )
    mla = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    return [KVCacheGroupSpec([f"swa{i}"], swa) for i in range(5)] + [
        KVCacheGroupSpec(["mla"], mla),
        KVCacheGroupSpec(["ring"], ring),
    ]


# (groups, hash_block_size, coordinator kwargs)
LAYOUTS = {
    "full": ([KVCacheGroupSpec(["f"], _full(16))], 16, {}),
    "full_partial": (
        [KVCacheGroupSpec(["f"], _full(64))],
        16,
        {"enable_partial_hash_hits": True},
    ),
    "full_eagle": (
        [KVCacheGroupSpec(["f"], _full(16), is_eagle_group=True)],
        16,
        {"use_eagle": True},
    ),
    "two_full": (
        [KVCacheGroupSpec(["a"], _full(16)), KVCacheGroupSpec(["b"], _full(16))],
        16,
        {},
    ),
    "full_swa": (
        [KVCacheGroupSpec(["f"], _full(16)), KVCacheGroupSpec(["s"], _swa(16, 32))],
        16,
        {},
    ),
    "full_swa_eagle_extra": (
        [
            KVCacheGroupSpec(["f"], _full(32)),
            KVCacheGroupSpec(["s"], _swa(16, 48, extra_retained_tokens=16)),
            KVCacheGroupSpec(["mtp"], _swa(16, 32), is_eagle_group=True),
        ],
        16,
        {"use_eagle": True},
    ),
    "full_mamba": (
        [KVCacheGroupSpec(["f"], _full(16)), KVCacheGroupSpec(["m"], _mamba(32))],
        16,
        {},
    ),
    "full_mamba_partial_eagle": (
        [
            KVCacheGroupSpec(["f"], _full(32), is_eagle_group=True),
            KVCacheGroupSpec(["m"], _mamba(64)),
        ],
        16,
        {"use_eagle": True, "enable_partial_hash_hits": True},
    ),
    "full_chunked_local": (
        [
            KVCacheGroupSpec(["f"], _full(16)),
            KVCacheGroupSpec(
                ["c"],
                ChunkedLocalAttentionSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=64,
                    dtype=torch.float16,
                    attention_chunk_size=64,
                ),
            ),
        ],
        16,
        {},
    ),
    "dsv4_five_group_eagle": (
        [
            KVCacheGroupSpec(["full_mla"], _full(256)),
            KVCacheGroupSpec(["swa"], _swa(64, 128)),
            KVCacheGroupSpec(["mtp"], _swa(64, 128), is_eagle_group=True),
            KVCacheGroupSpec(["c4"], _swa(4, 8)),
            KVCacheGroupSpec(["c128"], _swa(8, 128)),
        ],
        4,
        {"use_eagle": True},
    ),
    "dsv41_all_groups": (_dsv41_groups(), 128, {}),
    "dsv41": (
        [g for g in _dsv41_groups() if g.kv_cache_spec.prefix_cacheable],
        128,
        {},
    ),
}


def _make_coord(groups, hash_block_size, **kwargs) -> MooncakeStoreCoordinator:
    block_sizes = [
        g.kv_cache_spec.block_size for g in groups if g.kv_cache_spec.prefix_cacheable
    ]
    return MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=lcm(hash_block_size, *block_sizes),
        hash_block_size=hash_block_size,
        **kwargs,
    )


def _random_hashes(rng: random.Random, n: int) -> list[BlockHash]:
    hash_len = rng.choice([16, 32])
    return [BlockHash(rng.randbytes(hash_len)) for _ in range(n)]


def _random_exists(
    rng: random.Random, num_groups: int, hashes: list[BlockHash]
) -> set[tuple[int, bytes]]:
    """Mostly-present prefixes with random holes and a random cut-off, so hits
    land at varied lengths."""
    exists = set()
    for g in range(num_groups):
        cutoff = rng.randint(0, len(hashes))
        p = rng.choice([0.5, 0.9, 1.0])
        for i, h in enumerate(hashes):
            if i < cutoff and rng.random() < p:
                exists.add((g, bytes(h)))
    return exists


@pytest.mark.parametrize("layout", LAYOUTS)
def test_coordinator_matches_python(layout):
    groups, hash_block_size, kwargs = LAYOUTS[layout]
    coord = _make_coord(groups, hash_block_size, **kwargs)
    rust = rust_core.MooncakeStoreCoordinator(coord)
    alignment = coord.lcm_block_size
    if coord.enable_partial_hash_hits:
        alignment = hash_block_size
    for seed in range(NUM_SEEDS):
        rng = random.Random(seed)
        hashes = _random_hashes(rng, rng.randint(0, 3 * alignment // hash_block_size))
        exists = _random_exists(rng, len(groups), hashes)
        max_length = rng.randint(0, len(hashes) * hash_block_size)
        apply_eagle = rng.random() < 0.5
        expected = coord.find_longest_cache_hit(
            hashes,
            max_length,
            ExternalCachedBlockPool(hash_block_size, exists),
            apply_eagle=apply_eagle,
        )
        actual = rust.find_longest_cache_hit(
            hashes, max_length, exists, apply_eagle=apply_eagle
        )
        assert actual == expected, (seed, max_length)

        aligned = coord.align_lookup_length(max_length)
        assert rust.align_lookup_length(max_length) == aligned
        assert rust.load_mask(hashes, aligned) == coord.load_mask(hashes, aligned)
        assert tuple(rust.lookup_mask(aligned)) == coord.lookup_mask(aligned)


def _make_worker(layout: str, tp_size: int, chunk_divisor: int):
    """A lookup-only worker; `chunk_divisor` splits each block into that many
    Store chunks (as a Store chunk size below the block size does)."""
    groups, hash_block_size, kwargs = LAYOUTS[layout]
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker._capacity_only = False
    worker._selective_lease = None
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MooncakeStoreConnectorStats()
    worker.coord = _make_coord(groups, hash_block_size, **kwargs)
    worker._kv_cache_groups = worker.coord.kv_cache_groups
    worker.hash_block_size = hash_block_size
    worker.tp_size = tp_size
    # One KV head per rank: non-MLA groups need every rank's namespace.
    worker.num_kv_head = tp_size
    worker.dcp_size = worker.pcp_size = worker.pp_size = 1
    worker.pp_rank = 0
    worker._group_tp_replication_factors = (
        worker._compute_group_tp_replication_factors()
    )
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("model", 0, 0, 0, 0, group_id=g_idx),
            max(
                hash_block_size,
                group.kv_cache_spec.block_size
                // chunk_divisor
                // hash_block_size
                * hash_block_size,
            ),
            hash_block_size=hash_block_size,
        )
        for g_idx, group in enumerate(worker._kv_cache_groups)
    ]
    worker._init_lookup_key_prefixes()
    worker.store = MagicMock()
    return worker


def _prefix_store(rng: random.Random, hashes: list[BlockHash], num_groups: int):
    """`batch_is_exist` of a store holding each group's keys up to a random
    cut-off, with random holes; a missing key reads as absent or as an error
    code."""
    hash_index = {h.hex(): i for i, h in enumerate(hashes)}
    cutoffs = [
        rng.choice([len(hashes), rng.randint(0, len(hashes))])
        for _ in range(num_groups)
    ]
    p_hole = rng.choice([0.0, 0.05, 0.3])
    results: dict[str, int] = {}

    def batch_is_exist(keys: list[str]) -> list[int]:
        for key in keys:
            if key not in results:
                prefix, hash_hex = key.rsplit("@", 1)
                group = int(prefix.rsplit("@group:", 1)[1])
                if hash_index[hash_hex] < cutoffs[group] and rng.random() >= p_hole:
                    results[key] = 1
                else:
                    results[key] = rng.choice([0, -1])
        return [results[key] for key in keys]

    return batch_is_exist


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("tp_size,chunk_divisor", [(1, 1), (2, 1), (1, 2)])
def test_worker_lookup_matches_python(layout, tp_size, chunk_divisor):
    worker = _make_worker(layout, tp_size, chunk_divisor)
    rust_lookup = rust_core.MooncakeStoreLookup(worker)
    alignment = worker.coord.lcm_block_size
    for seed in range(NUM_LOOKUP_SEEDS):
        rng = random.Random(seed)
        hashes = _random_hashes(rng, rng.randint(0, 3 * alignment // 4))
        worker.store.batch_is_exist.side_effect = _prefix_store(
            rng, hashes, len(worker.token_dbs)
        )
        num_tokens = rng.randint(0, len(hashes) * worker.hash_block_size + 8)
        # As a list and as the lookup server's view of the request frame.
        blob = BlobBlockHashes(
            memoryview(b"".join(hashes)), len(hashes[0]) if hashes else 0
        )
        for block_hashes in (hashes, blob):
            outcomes = []
            for impl in (None, rust_lookup):
                worker._rust_lookup = impl
                try:
                    outcomes.append(worker.lookup(num_tokens, block_hashes))
                except AssertionError as e:
                    outcomes.append(type(e))
            assert outcomes[0] == outcomes[1], (seed, num_tokens)
