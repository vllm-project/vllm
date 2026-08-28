# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ECCPUWorker.

The byte-level tests exercise real accelerator (CUDA/XPU/...) stream/event
coordination against a real ``ECSharedRegion`` mmap and are skipped on hosts
without an accelerator. The lifecycle tests don't fire any
``current_platform.*`` device primitives and run anywhere.

Mocking policy
--------------
- ``create_ec_shared_region`` is patched to inject a real ``ECSharedRegion``
  (mmap backed) with deterministic small dimensions, so byte assertions
  hit the actual mmap. The real responsibility of the worker — block
  index → mmap byte mapping, stream coordination, dtype/shape — is
  exercised end to end against a real GPU.
- ``is_pin_memory_available`` is patched per-test so the rank-gating
  path can be probed deterministically without depending on host caps.
- The shutdown test bypasses ``__init__`` via ``object.__new__`` and
  injects a ``MagicMock`` region.
"""

import contextlib
import logging
import time
import uuid
from collections import deque
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker
from vllm.platforms import current_platform

# ── shape constants ──────────────────────────────────────────────────────────

# hidden_dim=8, dtype=fp16 (2 bytes) → block_size_bytes = 16, so one row of
# encoder output fits in exactly one block.
_HIDDEN_DIM = 8
_DTYPE = torch.float16
_BLOCK_SIZE_BYTES = _HIDDEN_DIM * _DTYPE.itemsize
_NUM_BLOCKS = 8

DEVICE_TYPE = current_platform.device_type

_requires_accelerator = pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="exercises real accelerator stream/event coordination in ECCPUWorker",
)

# The memory-lifetime tests hold a copy back with `torch.cuda._sleep`, which the
# CUDA-like platforms provide but XPU does not.
_requires_cuda_alike = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="stalling the compute stream requires a CUDA-like platform",
)

# Cycles to stall the compute stream for. Must comfortably outlast the host-side
# work the lifetime tests do while a copy is deliberately held back — ~0.5 s on
# a current datacenter GPU.
_STALL_CYCLES = 1_000_000_000


def _make_region() -> ECSharedRegion:
    """Fresh region backed by a real per-test mmap file."""
    return ECSharedRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=_NUM_BLOCKS,
        block_size_bytes=_BLOCK_SIZE_BYTES,
    )


def _vllm_config(rank: int = 0) -> Mock:
    return create_ec_vllm_config(rank=rank, dtype=_DTYPE)


def _meta(
    *, saves: dict | None = None, loads: dict | None = None
) -> ECCPUConnectorMetadata:
    """Build step metadata from plain `{mm_hash: block_ids}` dicts.

    Load transfer ids are synthesized here so tests can keep naming loads by
    mm_hash; `_load_id` recovers the id the worker will report.
    """
    return ECCPUConnectorMetadata(
        saves=saves or {},
        loads={
            mm_hash: (idx, block_ids)
            for idx, (mm_hash, block_ids) in enumerate((loads or {}).items())
        },
    )


def _load_id(meta: ECCPUConnectorMetadata, mm_hash: str) -> int:
    return meta.loads[mm_hash][0]


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def make_worker():
    """Factory that builds an ``ECCPUWorker`` over a real ``ECSharedRegion``.

    Tracks the workers and regions it creates and tears them down at the
    end of the test, so each test starts from a fresh /dev/shm file.
    """
    workers: list[ECCPUWorker] = []
    regions: list[ECSharedRegion] = []

    def factory(
        *,
        rank: int = 0,
        tp_rank: int = 0,
        pcp_rank: int = 0,
        pin_memory_available: bool = False,
    ) -> ECCPUWorker:
        region = _make_region()
        regions.append(region)
        pcp_group = Mock()
        pcp_group.rank_in_group = pcp_rank
        with (
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.worker.create_ec_shared_region",
                return_value=region,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.worker.is_pin_memory_available",
                return_value=pin_memory_available,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.worker.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.worker.get_pcp_group",
                return_value=pcp_group,
            ),
        ):
            worker = ECCPUWorker(_vllm_config(rank=rank))
        workers.append(worker)
        return worker

    yield factory

    for worker in workers:
        with contextlib.suppress(Exception):
            worker.shutdown()
    for region in regions:
        with contextlib.suppress(Exception):
            region.cleanup()


def _wait_for_completion(
    worker: ECCPUWorker, expected, direction: str, timeout_s: float = 30.0
) -> None:
    """Poll ``build_connector_worker_meta`` until ``expected`` is reported done.

    Saves are reported by mm_hash, loads by transfer id.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        meta = worker.build_connector_worker_meta()
        if meta is not None and expected in getattr(meta, f"completed_{direction}"):
            return
        time.sleep(0.001)
    raise AssertionError(f"{direction} for {expected!r} never reported complete")


def _warm_up_stream_pools(worker: ECCPUWorker) -> None:
    """Run one throwaway save so the stream and event pools are populated.

    The first CUDA stream creation in a process blocks the host for ~0.5 s on
    driver initialization, which would swallow the stall the lifetime tests
    below install. One flush puts a stream and its events in the worker's pools,
    matching the steady state every step after the first sees.
    """
    scratch = torch.zeros(_HIDDEN_DIM, dtype=_DTYPE, device=DEVICE_TYPE)
    worker.save_caches({"w": scratch}, "w", _meta(saves={"w": [0]}))
    worker.flush_saves()
    _wait_for_completion(worker, "w", "saves")
    assert worker._stream_pool, "warm-up did not recycle a stream"


# ── save_caches ──────────────────────────────────────────────────────────────


@_requires_accelerator
@pytest.mark.parametrize(
    "n_elements,block_ids",
    [
        (3 * _HIDDEN_DIM, [7, 2, 5]),  # exact fit: 3 full blocks, no remainder
        (3 * _HIDDEN_DIM + 4, [7, 2, 5, 0]),  # partial last block: 4 fp16 trail
        (3 * _HIDDEN_DIM, [2, 3, 4]),  # adjacent: one coalesced copy
        (3 * _HIDDEN_DIM + 4, [2, 3, 4, 6]),  # coalesced run, then a gap
    ],
    ids=["exact-fit", "partial-last-block", "adjacent-blocks", "run-then-gap"],
)
def test_save_caches_writes_to_assigned_blocks(make_worker, n_elements, block_ids):
    """``save_caches`` + ``flush_saves`` copies the source GPU tensor's bytes
    into the block IDs named by ``meta.saves``, in the order they appear,
    via a single batched GPU→CPU copy.

    When ``total_bytes`` is not a multiple of ``block_size_bytes`` the last
    block is partially written; the unwritten tail must remain whatever
    was there before. Blocks not named in ``meta.saves`` must be untouched.
    Blocks adjacent in the region share one descriptor, which must land the
    same bytes in the same blocks as one descriptor per block does.
    """
    worker = make_worker()
    sentinel = 0x5A
    worker._region.blocks.fill_(sentinel)

    src = torch.arange(n_elements, dtype=_DTYPE, device=DEVICE_TYPE)
    expected_bytes = src.cpu().reshape(-1).view(torch.uint8)
    total_bytes = n_elements * _DTYPE.itemsize

    worker.save_caches({"h": src}, "h", _meta(saves={"h": block_ids}))
    worker.flush_saves()
    _wait_for_completion(worker, "h", "saves")

    for slot, block_idx in enumerate(block_ids):
        block_byte_start = slot * _BLOCK_SIZE_BYTES
        block_byte_end = min(block_byte_start + _BLOCK_SIZE_BYTES, total_bytes)
        n_written = block_byte_end - block_byte_start
        actual_bytes = worker._region.blocks[block_idx, :n_written].view(torch.uint8)
        assert torch.equal(
            actual_bytes, expected_bytes[block_byte_start:block_byte_end]
        ), f"block {block_idx} (slot {slot}) bytes mismatch"
        if n_written < _BLOCK_SIZE_BYTES:
            tail = worker._region.blocks[block_idx, n_written:]
            assert torch.all(tail == sentinel), (
                f"block {block_idx} tail was overwritten"
            )

    for idx in set(range(_NUM_BLOCKS)) - set(block_ids):
        assert torch.all(worker._region.blocks[idx] == sentinel), (
            f"block {idx} (unassigned) was overwritten"
        )


@_requires_accelerator
@pytest.mark.parametrize(
    "n_elements,block_ids,expected_sizes",
    [
        (3 * _HIDDEN_DIM, [2, 3, 4], [3 * _BLOCK_SIZE_BYTES]),
        (3 * _HIDDEN_DIM, [7, 2, 5], [_BLOCK_SIZE_BYTES] * 3),
        (3 * _HIDDEN_DIM, [2, 3, 6], [2 * _BLOCK_SIZE_BYTES, _BLOCK_SIZE_BYTES]),
        (2 * _HIDDEN_DIM + 4, [2, 3, 4], [(2 * _HIDDEN_DIM + 4) * _DTYPE.itemsize]),
        (_HIDDEN_DIM, [2, 3, 4], [_BLOCK_SIZE_BYTES]),
    ],
    ids=[
        "one-run",
        "no-run",
        "run-then-gap",
        "run-stops-at-output",
        "blocks-past-output-skipped",
    ],
)
def test_save_caches_coalesces_only_adjacent_blocks(
    make_worker, n_elements, block_ids, expected_sizes
):
    """One descriptor per run of region-adjacent blocks, not one per block.

    Per-descriptor overhead dominates these copies, so the byte tests above
    would pass either way; this pins the descriptor layout that makes them
    fast. A coalesced copy also stops at the encoder output, since entries are
    sized by placeholder count, which can exceed the number of embeddings.
    """
    worker = make_worker()
    src = torch.arange(n_elements, dtype=_DTYPE, device=DEVICE_TYPE)

    worker.save_caches({"h": src}, "h", _meta(saves={"h": block_ids}))

    assert worker._save_bufs.sizes[: worker._save_count].tolist() == expected_sizes


def test_save_caches_noop_when_mm_hash_not_in_saves(make_worker):
    """When the scheduler hasn't pre-allocated blocks for ``mm_hash``,
    ``save_caches`` + ``flush_saves`` is a pure no-op."""
    worker = make_worker()
    sentinel = 0x42
    worker._region.blocks.fill_(sentinel)

    worker.save_caches({}, "h", _meta(saves={}))
    worker.flush_saves()

    assert torch.all(worker._region.blocks == sentinel)


@pytest.mark.parametrize(
    "tp_rank,pcp_rank",
    [(1, 0), (0, 1), (1, 1)],
    ids=["tp1-pcp0", "tp0-pcp1", "tp1-pcp1"],
)
def test_save_caches_noop_for_non_save_rank(make_worker, tp_rank, pcp_rank):
    """Only TP rank 0 + PCP rank 0 writes to mmap. All other combinations
    must skip the save entirely."""
    worker = make_worker(tp_rank=tp_rank, pcp_rank=pcp_rank)
    sentinel = 0x42
    worker._region.blocks.fill_(sentinel)

    worker.save_caches(
        {"h": torch.zeros(8, dtype=_DTYPE)}, "h", _meta(saves={"h": [0]})
    )
    worker.flush_saves()

    assert torch.all(worker._region.blocks == sentinel)


def test_save_caches_raises_when_allocated_blocks_too_small(make_worker):
    """``save_caches`` must raise ``AssertionError`` when the encoder output
    is larger than the allocated block space."""
    worker = make_worker()
    src = torch.zeros(3 * _HIDDEN_DIM, dtype=_DTYPE)
    with pytest.raises(AssertionError, match="exceeds allocated blocks"):
        worker.save_caches({"h": src}, "h", _meta(saves={"h": [0, 1]}))


@_requires_accelerator
def test_save_caches_batches_multiple_hashes(make_worker):
    """Multiple save_caches calls are batched into a single flush."""
    worker = make_worker()
    sentinel = 0x5A
    worker._region.blocks.fill_(sentinel)

    src_a = torch.arange(_HIDDEN_DIM, dtype=_DTYPE, device=DEVICE_TYPE)
    src_b = torch.arange(_HIDDEN_DIM, 2 * _HIDDEN_DIM, dtype=_DTYPE, device=DEVICE_TYPE)

    cache = {"a": src_a, "b": src_b}
    worker.save_caches(cache, "a", _meta(saves={"a": [1], "b": [3]}))
    worker.save_caches(cache, "b", _meta(saves={"a": [1], "b": [3]}))
    worker.flush_saves()
    _wait_for_completion(worker, "a", "saves")

    expected_a = src_a.cpu().view(torch.uint8)
    expected_b = src_b.cpu().view(torch.uint8)
    actual_a = worker._region.blocks[1].view(torch.uint8)
    actual_b = worker._region.blocks[3].view(torch.uint8)
    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_b, expected_b)

    # Unassigned blocks untouched.
    for idx in (0, 2, 4, 5, 6, 7):
        assert torch.all(worker._region.blocks[idx] == sentinel)


class _HighAddress:
    """A device tensor reporting an address with the high bit set.

    XPU USM allocations live up there and no host allocator will hand out such
    an address, so naming one is the only way to reach the arithmetic. Every
    attribute other than the address comes from the real tensor.
    """

    def __init__(self, real: torch.Tensor, address: int):
        self._real = real
        self._address = address

    def __getattr__(self, name):
        return getattr(self._real, name)

    def view(self, *args):
        return self

    def data_ptr(self) -> int:
        return self._address


@_requires_accelerator
def test_save_descriptors_hold_high_bit_source_addresses(make_worker):
    """Source addresses >= 2**63 survive into the descriptor buffers."""
    worker = make_worker()
    real = torch.zeros(2 * _HIDDEN_DIM, dtype=_DTYPE, device=DEVICE_TYPE)
    address = 2**64 - 8 * _BLOCK_SIZE_BYTES

    # Two runs, so the second descriptor's address is an offset off the base.
    worker.save_caches(
        {"a": _HighAddress(real, address)}, "a", _meta(saves={"a": [1, 5]})
    )

    assert worker._save_count == 2
    stored = worker._save_bufs.src_np[:2].astype(np.uint64)
    assert list(stored) == [address, address + _BLOCK_SIZE_BYTES]

    # The batch describes an address nothing may read from; drop it unflushed.
    worker._save_bufs = None
    worker._save_count = 0
    worker._save_mm_hashes = []


# ── start_load_caches ────────────────────────────────────────────────────────


@_requires_accelerator
def test_start_load_caches_copies_with_correct_shape_dtype_and_bytes(make_worker):
    """Single batched load across all hashes with correct byte→dtype→shape."""
    worker = make_worker()
    n_blocks = 3
    src_orig = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE).reshape(
        n_blocks, _HIDDEN_DIM
    )
    src_int8 = src_orig.view(torch.int8).reshape(n_blocks, _BLOCK_SIZE_BYTES)

    block_ids = [3, 1, 6]
    for i, idx in enumerate(block_ids):
        worker._region.blocks[idx].copy_(src_int8[i])

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": block_ids}))

    out = encoder_cache["h"]
    assert out.device.type == DEVICE_TYPE, (
        "consumer worker must place the tensor on the accelerator"
    )
    assert out.shape == (n_blocks, _HIDDEN_DIM)
    assert out.dtype == _DTYPE
    assert torch.equal(out.cpu(), src_orig)


@_requires_accelerator
@pytest.mark.parametrize(
    "loads,expected_sizes",
    [
        ({"a": [2, 3], "b": [4, 5]}, [4 * _BLOCK_SIZE_BYTES]),
        ({"b": [4, 5], "a": [2, 3]}, [4 * _BLOCK_SIZE_BYTES]),
        ({"a": [2, 3], "b": [6, 7]}, [2 * _BLOCK_SIZE_BYTES] * 2),
    ],
    ids=["adjacent-entries", "adjacent-entries-reversed", "gap-between-entries"],
)
def test_start_load_caches_coalesces_across_entries(make_worker, loads, expected_sizes):
    """Blocks adjacent in the region share a descriptor across entries too.

    Entries are ordered by first block, so whether they coalesce follows from
    the region layout rather than the order the scheduler dispatched them.
    """
    worker = make_worker()
    captured: list[list[int]] = []

    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu.worker.swap_blocks_batch",
        side_effect=lambda src, dst, sizes, **kw: captured.append(sizes.tolist()),
    ):
        worker.start_load_caches({}, _meta(loads=loads))

    assert captured == [expected_sizes]


@_requires_accelerator
def test_start_load_caches_noop_when_loads_is_empty(make_worker):
    """When ``meta.loads`` is empty the early-return must fire."""
    worker = make_worker()
    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={}))

    assert encoder_cache == {}


@_requires_accelerator
def test_start_load_caches_skips_cached_and_loads_new_in_same_step(make_worker):
    """Every hash in ``meta.loads`` is copied from mmap, including one whose
    key is already resident in ``encoder_cache`` — there is no worker-side
    skip."""
    worker = make_worker()
    n_blocks = 3
    src_orig = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE).reshape(
        n_blocks, _HIDDEN_DIM
    )
    src_int8 = src_orig.view(torch.int8).reshape(n_blocks, _BLOCK_SIZE_BYTES)
    block_ids = [0, 4, 2]
    for i, idx in enumerate(block_ids):
        worker._region.blocks[idx].copy_(src_int8[i])

    cached_tensor = torch.full((1, _HIDDEN_DIM), 99.0, dtype=_DTYPE, device=DEVICE_TYPE)
    encoder_cache: dict[str, torch.Tensor] = {"cached_h": cached_tensor}
    worker.start_load_caches(
        encoder_cache,
        _meta(loads={"resident_h": [block_ids[0]], "new_h": block_ids[1:]}),
    )

    resident = encoder_cache["resident_h"]
    assert resident is not cached_tensor, "resident entry must be reloaded, not skipped"
    assert torch.equal(resident.cpu(), src_orig[:1])

    new = encoder_cache["new_h"]
    assert new.shape == (2, _HIDDEN_DIM)
    assert new.dtype == _DTYPE
    assert torch.equal(new.cpu(), src_orig[1:])


@_requires_accelerator
@pytest.mark.parametrize(
    "tp_rank,pcp_rank",
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    ids=["tp0-pcp0", "tp1-pcp0", "tp0-pcp1", "tp1-pcp1"],
)
def test_start_load_caches_works_on_all_ranks(make_worker, tp_rank, pcp_rank):
    """All TP/PCP ranks must load from mmap — loads are NOT gated like saves."""
    worker = make_worker(tp_rank=tp_rank, pcp_rank=pcp_rank)
    n_blocks = 2
    src_orig = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE).reshape(
        n_blocks, _HIDDEN_DIM
    )
    src_int8 = src_orig.view(torch.int8).reshape(n_blocks, _BLOCK_SIZE_BYTES)
    block_ids = [1, 3]
    for i, idx in enumerate(block_ids):
        worker._region.blocks[idx].copy_(src_int8[i])

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": block_ids}))

    out = encoder_cache["h"]
    assert out.device.type == DEVICE_TYPE
    assert torch.equal(out.cpu(), src_orig)


# ── round-trip ───────────────────────────────────────────────────────────────


@_requires_accelerator
def test_save_then_load_round_trips_bytes(make_worker):
    """Full producer→mmap→consumer byte path in one shot."""
    worker = make_worker()
    n_blocks = 3
    block_ids = [5, 1, 6]

    src = torch.arange(
        n_blocks * _HIDDEN_DIM, dtype=_DTYPE, device=DEVICE_TYPE
    ).reshape(n_blocks, _HIDDEN_DIM)
    worker.save_caches({"h": src}, "h", _meta(saves={"h": block_ids}))
    worker.flush_saves()
    _wait_for_completion(worker, "h", "saves")

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": block_ids}))

    out = encoder_cache["h"]
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert torch.equal(out.cpu(), src.cpu())


# ── memory lifetime across in-flight copies ─────────────────────────────────


@_requires_cuda_alike
def test_save_survives_encoder_cache_free_before_copy_runs(make_worker):
    """The bytes handed to ``save_caches`` must reach the mmap intact even when
    the caller drops its ``encoder_cache`` reference before the copy has run.

    The model runner pops encoder cache entries on whatever step the scheduler
    asks it to, with no dependency on the save copy having landed. Dropping the
    last reference hands the GPU memory back to the caching allocator, which is
    free to reuse it for the next same-sized allocation on the compute stream —
    the copy reads that memory on a different stream, so the allocator has no
    reason to hold it back.

    The stall makes the hazard deterministic rather than timing-dependent: the
    save copy is gated behind the compute stream, so it provably has not started
    while the overwrite (issued on an ungated stream and waited on) completes.

    The mmap must be pinned, as it is in production: a copy into pageable host
    memory blocks the host until it lands, which would hide the hazard.
    """
    worker = make_worker(pin_memory_available=True)
    _warm_up_stream_pools(worker)
    worker._region.blocks.fill_(0x5A)

    block_ids = [0, 1, 2]
    saved = torch.full(
        (len(block_ids), _HIDDEN_DIM), 1.0, dtype=_DTYPE, device=DEVICE_TYPE
    )
    expected = saved.cpu().reshape(-1).view(torch.uint8)

    encoder_cache = {"h": saved}
    worker.save_caches(encoder_cache, "h", _meta(saves={"h": block_ids}))

    torch.cuda._sleep(_STALL_CYCLES)
    worker.flush_saves()

    saved_ptr = saved.data_ptr()
    del saved
    encoder_cache.clear()

    # Same size on the same stream, so the allocator hands back the block just
    # freed. Overwriting it from an ungated stream — and waiting for that write
    # — orders the overwrite strictly before the still-stalled save copy.
    overwrite = torch.empty(
        (len(block_ids), _HIDDEN_DIM), dtype=_DTYPE, device=DEVICE_TYPE
    )
    reused = overwrite.data_ptr() == saved_ptr
    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        overwrite.fill_(-1.0)
    other_stream.synchronize()

    _wait_for_completion(worker, "h", "saves")

    actual = worker._region.blocks[block_ids].reshape(-1).view(torch.uint8)
    assert torch.equal(actual, expected), (
        "mmap holds bytes written after save_caches was called; the copy read "
        f"freed source memory (allocator reused the freed block: {reused})"
    )


@_requires_cuda_alike
def test_load_buffer_survives_eviction_while_consumer_read_is_queued(make_worker):
    """A loaded encoder cache entry must keep its bytes for a consumer already
    queued on the compute stream, even after the entry is dropped and another
    load is dispatched.

    ``start_load_caches`` allocates its destination inside the load stream's
    context, which ties that memory to the load stream in the caching allocator.
    Dropping the encoder cache entry returns it to the load stream's pool, where
    the next load can claim it — while the model's read of it is still queued on
    the compute stream.
    """
    worker = make_worker(pin_memory_available=True)
    _warm_up_stream_pools(worker)
    a_blocks, b_blocks = [4, 5], [6, 7]
    for block_idx, fill in zip(a_blocks + b_blocks, (0x11, 0x12, 0x21, 0x22)):
        worker._region.blocks[block_idx].fill_(fill)
    expected_a = worker._region.blocks[a_blocks].reshape(-1).clone()

    encoder_cache: dict[str, torch.Tensor] = {}
    meta_a = _meta(loads={"a": a_blocks})
    worker.start_load_caches(encoder_cache, meta_a)
    loaded_a = encoder_cache["a"]

    # Queue a consumer of the loaded entry on the compute stream behind a stall,
    # the way the model reads the encoder cache later in the step.
    torch.cuda._sleep(_STALL_CYCLES)
    consumed = loaded_a.clone()

    # Draining the completion report recycles the load stream into the pool, so
    # the next load allocates from the same allocator pool the entry was freed to.
    _wait_for_completion(worker, _load_id(meta_a, "a"), "loads")

    loaded_a_ptr = loaded_a.data_ptr()
    del loaded_a
    encoder_cache.clear()

    worker.start_load_caches(encoder_cache, _meta(loads={"b": b_blocks}))
    reused = encoder_cache["b"].data_ptr() == loaded_a_ptr
    torch.accelerator.synchronize()

    assert torch.equal(consumed.cpu().reshape(-1).view(torch.int8), expected_a), (
        "queued consumer read bytes from a later load; the load destination was "
        f"recycled while still in use (allocator reused the buffer: {reused})"
    )


# ── buffer recycling ────────────────────────────────────────────────────────


@_requires_accelerator
def test_buffer_pool_is_reused_across_save_steps(make_worker):
    """Once a save copy completes its descriptor buffers return to the pool and
    are reused by the next flush — no reallocation.

    The buffers hold the addresses the copy reads, so they stay out of the pool
    until the transfer's end event fires.
    """
    worker = make_worker()
    src = torch.arange(_HIDDEN_DIM, dtype=_DTYPE, device=DEVICE_TYPE)

    worker.save_caches({"h": src}, "h", _meta(saves={"h": [0]}))
    worker.flush_saves()
    _wait_for_completion(worker, "h", "saves")

    assert len(worker._buf_pool._pool) == 1
    buf_id = id(worker._buf_pool._pool[0].src_ptrs)

    # Second step reuses the same buffer.
    worker.save_caches({"h": src}, "h", _meta(saves={"h": [1]}))
    worker.flush_saves()
    _wait_for_completion(worker, "h", "saves")

    assert len(worker._buf_pool._pool) == 1
    assert id(worker._buf_pool._pool[0].src_ptrs) == buf_id


@_requires_accelerator
def test_buffer_pool_is_reused_across_load_steps(make_worker):
    """Once a load copy completes its descriptor buffers return to the pool and
    are reused by the next call — no reallocation."""
    worker = make_worker()
    worker._region.blocks[0].fill_(0x01)
    worker._region.blocks[1].fill_(0x02)

    encoder_cache: dict[str, torch.Tensor] = {}
    meta = _meta(loads={"a": [0]})
    worker.start_load_caches(encoder_cache, meta)
    _wait_for_completion(worker, _load_id(meta, "a"), "loads")

    assert len(worker._buf_pool._pool) == 1
    buf_id = id(worker._buf_pool._pool[0].src_ptrs)

    encoder_cache2: dict[str, torch.Tensor] = {}
    meta2 = _meta(loads={"b": [1]})
    worker.start_load_caches(encoder_cache2, meta2)
    _wait_for_completion(worker, _load_id(meta2, "b"), "loads")

    assert len(worker._buf_pool._pool) == 1
    assert id(worker._buf_pool._pool[0].src_ptrs) == buf_id


# ── lifecycle ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pin_available,expected_pinned",
    [
        (True, True),
        (False, False),
    ],
    ids=["available", "unavailable"],
)
def test_init_pins_memory_when_available(pin_available, expected_pinned):
    """Every worker must call ``pin_memory()`` when the platform allows it."""
    region = _make_region()
    region.pin_memory = MagicMock()  # type: ignore[method-assign]

    pcp_group = Mock()
    pcp_group.rank_in_group = 0
    with (
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.create_ec_shared_region",
            return_value=region,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.is_pin_memory_available",
            return_value=pin_available,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.get_pcp_group",
            return_value=pcp_group,
        ),
    ):
        ECCPUWorker(_vllm_config(rank=0))

    try:
        if expected_pinned:
            region.pin_memory.assert_called_once()
        else:
            region.pin_memory.assert_not_called()
    finally:
        region.cleanup()


def test_shutdown_calls_region_cleanup_and_swallows_errors(caplog_vllm):
    """``shutdown`` must always call ``region.cleanup`` — and must never
    raise."""
    worker = object.__new__(ECCPUWorker)
    mock_region = Mock(spec=ECSharedRegion)
    worker._region = mock_region
    worker._save_bufs = None
    worker._save_count = 0
    worker._save_mm_hashes = []
    worker._inflight_saves = deque()
    worker._inflight_loads = deque()
    worker._stream_pool = []
    worker._event_pool = []

    worker.shutdown()
    mock_region.cleanup.assert_called_once()

    mock_region.cleanup.side_effect = RuntimeError("boom")
    with caplog_vllm.at_level(logging.DEBUG, logger="vllm"):
        worker.shutdown()  # exception must be swallowed

    assert mock_region.cleanup.call_count == 2
    assert any(
        "worker region cleanup failed" in r.message
        for r in caplog_vllm.records
        if r.levelno == logging.DEBUG
    )


# ── e2e: scheduler + worker pipeline ────────────────────────────────────────


@_requires_accelerator
def test_e2e_scheduler_worker_save_then_load(make_worker, monkeypatch):
    """Full pipeline: scheduler allocates blocks, worker saves a GPU tensor to
    mmap via flush_saves, the worker's completion report marks the entry ready,
    the worker loads from mmap back to GPU, and the result matches the original.

    Exercises the real scheduler + worker cooperation through a shared
    ECSharedRegion, with real accelerator transfers and stream coordination,
    and the event-driven mark_ready path.
    """
    import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
        ECCPUScheduler,
    )

    # Build worker first (gets us a real region).
    worker = make_worker()
    region = worker._region

    # Build scheduler sharing the same region.
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: region)

    scheduler = ECCPUScheduler(_vllm_config())

    # -- Step 1: scheduler allocates, worker saves --
    n_blocks = 3
    src = torch.arange(
        n_blocks * _HIDDEN_DIM, dtype=_DTYPE, device=DEVICE_TYPE
    ).reshape(n_blocks, _HIDDEN_DIM)

    class _Pos:
        offset = 0
        length = n_blocks

    class _Feature:
        identifier = "img_001"
        mm_position = _Pos()

    class _Request:
        request_id = "req_e2e"
        mm_features = [_Feature()]

    scheduler.update_state_after_alloc(_Request(), 0)
    meta_save = scheduler.build_connector_meta(scheduler_output=None)
    assert "img_001" in meta_save.saves

    encoder_cache = {"img_001": src}
    worker.save_caches(encoder_cache, "img_001", meta_save)
    worker.flush_saves()

    # -- Step 2: worker reports the save memcpy complete → scheduler marks ready --
    torch.accelerator.synchronize()
    worker_meta = worker.build_connector_worker_meta()
    assert worker_meta is not None
    assert "img_001" in worker_meta.completed_saves

    class _Output:
        ec_connector_worker_meta = worker_meta

    scheduler.update_connector_output(_Output())
    assert scheduler.has_cache_item("img_001") is True

    # -- Step 3: scheduler emits load, worker loads --
    scheduler.update_state_after_alloc(_Request(), 0)
    meta_load = scheduler.build_connector_meta(scheduler_output=None)
    assert "img_001" in meta_load.loads
    _, load_blocks = meta_load.loads["img_001"]
    assert load_blocks == meta_save.saves["img_001"]

    load_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(load_cache, meta_load)

    out = load_cache["img_001"]
    assert out.device.type == DEVICE_TYPE
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert torch.equal(out.cpu(), src.cpu())

    scheduler.shutdown()
