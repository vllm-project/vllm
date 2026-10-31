# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for testing PooledTierManager subclasses (fs, valkey)."""

import mmap
import time
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.pooled import PooledTierManager

CTX = ReqContext(req_id="test")


def make_offloading_spec(
    enable_kv_cache_events: bool = False,
    *,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int | None = None,
    replicated_layout: bool = False,
    is_parallelism_agnostic: bool = False,
) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    if world_size is None:
        world_size = tp_size
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    chunk_ids: list[int] | None = None,
    is_promotion: bool = False,
) -> TransferJob:
    if chunk_ids is None:
        chunk_ids = list(range(len(keys)))
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array(chunk_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=CTX,
    )


def drain(tier: PooledTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: PooledTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def assert_store_then_load_roundtrip(
    tier: PooledTierManager, tensor: torch.Tensor
) -> None:
    """Store every chunk of ``tensor``, zero it, and load it back intact."""
    tensor[:] = torch.rand_like(tensor)
    expected = tensor.clone()
    keys = [key(i) for i in range(len(tensor))]
    tier.submit_store(make_job(1, keys))
    assert [r.success for r in drain(tier)] == [True]
    assert lookup_and_wait(tier, keys) == [LookupResult.HIT] * len(keys)

    tensor.zero_()
    tier.submit_load(make_job(2, keys, is_promotion=True))
    assert [r.success for r in drain(tier)] == [True]
    assert torch.equal(tensor, expected)


def page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)
