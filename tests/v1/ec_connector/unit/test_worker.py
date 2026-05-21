# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ECCPUWorker.

The byte-level tests exercise real CUDA stream/event coordination
against a real ``ECSharedRegion`` mmap and are skipped on hosts without
CUDA. The lifecycle tests don't fire any ``torch.cuda.*`` primitives and
run anywhere.

Mocking policy
--------------
- ``setup_ec_region`` is patched to inject a real ``ECSharedRegion``
  (mmap backed) with deterministic small dimensions, so byte assertions
  hit the actual mmap. The real responsibility of the worker — block
  index → mmap byte mapping, stream coordination, dtype/shape — is
  exercised end to end against a real GPU.
- ``is_pin_memory_available`` is patched per-test so the rank-gating
  path can be probed deterministically without depending on host caps.
- The shutdown test bypasses ``__init__`` via ``object.__new__`` and
  injects a ``MagicMock`` region, mirroring the unit-test pattern from
  ``tests/v1/kv_connector/unit/test_nixl_connector_hma.py``.
"""

import contextlib
import logging
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import ECRegionLayout
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    ECSharedRegion,
)

# ── shape constants ──────────────────────────────────────────────────────────

# hidden_dim=8, dtype=fp16 (2 bytes) → block_size_bytes = 16, so one row of
# encoder output fits in exactly one block.
_HIDDEN_DIM = 8
_DTYPE = torch.float16
_BLOCK_SIZE_BYTES = _HIDDEN_DIM * _DTYPE.itemsize
_NUM_BLOCKS = 8

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="exercises real CUDA stream/event coordination in ECCPUWorker",
)


def _make_layout() -> ECRegionLayout:
    """Fresh layout backed by a real per-test mmap file."""
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()),
        num_blocks=_NUM_BLOCKS,
        block_size_bytes=_BLOCK_SIZE_BYTES,
    )
    return ECRegionLayout(
        region=region,
        dtype=_DTYPE,
        hidden_dim=_HIDDEN_DIM,
        element_size=_DTYPE.itemsize,
        block_size_bytes=_BLOCK_SIZE_BYTES,
        num_blocks=_NUM_BLOCKS,
    )


def _vllm_config(rank: int = 0) -> Mock:
    cfg = Mock(spec=VllmConfig)
    cfg.parallel_config = Mock(spec=ParallelConfig)
    cfg.parallel_config.rank = rank
    return cfg


def _meta(
    *, saves: dict | None = None, loads: dict | None = None
) -> ECCPUConnectorMetadata:
    return ECCPUConnectorMetadata(saves=saves or {}, loads=loads or {})


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def make_worker():
    """Factory that builds an ``ECCPUWorker`` over a real ``ECSharedRegion``.

    Tracks the workers and regions it creates and tears them down at the
    end of the test, so each test starts from a fresh /dev/shm file.
    """
    workers: list[ECCPUWorker] = []
    layouts: list[ECRegionLayout] = []

    def factory(
        *,
        rank: int = 0,
        pin_memory_available: bool = False,
    ) -> ECCPUWorker:
        layout = _make_layout()
        layouts.append(layout)
        with (
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.worker.setup_ec_region",
                return_value=layout,
            ),
            patch(
                "vllm.distributed.ec_transfer.ec_connector.cpu.worker.is_pin_memory_available",
                return_value=pin_memory_available,
            ),
        ):
            worker = ECCPUWorker(_vllm_config(rank=rank))
        workers.append(worker)
        return worker

    yield factory

    for worker in workers:
        with contextlib.suppress(Exception):
            worker.shutdown()
    for layout in layouts:
        with contextlib.suppress(Exception):
            layout.region.cleanup()


# ── save_caches ──────────────────────────────────────────────────────────────


@_requires_cuda
@pytest.mark.parametrize(
    "n_elements,n_blocks",
    [
        (3 * _HIDDEN_DIM, 3),  # exact fit: 3 full blocks, no remainder
        (3 * _HIDDEN_DIM + 4, 4),  # partial last block: 4 fp16 trail in slot 3
    ],
    ids=["exact-fit", "partial-last-block"],
)
def test_save_caches_writes_to_assigned_blocks(make_worker, n_elements, n_blocks):
    """``save_caches`` copies the source GPU tensor's bytes into the block
    indices named by ``meta.saves``, in the order they appear, via a real
    GPU→CPU copy on the dedicated stream + synchronize.

    When ``total_bytes`` is not a multiple of ``block_size_bytes`` the last
    block is partially written; the unwritten tail must remain whatever
    was there before. Blocks not named in ``meta.saves`` must be untouched.
    """
    worker = make_worker()
    sentinel = 0x5A  # int8-safe; lets us detect any stray write.
    worker._cpu_blocks.fill_(sentinel)

    # Source on GPU is the encoder output shape and forces
    # save_caches's GPU→CPU copy + stream synchronize to be the only
    # thing that makes these bytes visible to the assertions below.
    src = torch.arange(n_elements, dtype=_DTYPE, device="cuda")
    expected_bytes = src.cpu().reshape(-1).view(torch.uint8)
    total_bytes = n_elements * _DTYPE.itemsize

    # Non-monotonic block indices catch any "slot" / "block_idx" mixup.
    block_indices = [7, 2, 5, 0][:n_blocks]
    worker.save_caches({"h": src}, "h", _meta(saves={"h": block_indices}))

    for slot, block_idx in enumerate(block_indices):
        block_byte_start = slot * _BLOCK_SIZE_BYTES
        block_byte_end = min(block_byte_start + _BLOCK_SIZE_BYTES, total_bytes)
        n_written = block_byte_end - block_byte_start
        actual_bytes = worker._cpu_blocks[block_idx, :n_written].view(torch.uint8)
        assert torch.equal(
            actual_bytes, expected_bytes[block_byte_start:block_byte_end]
        ), f"block {block_idx} (slot {slot}) bytes mismatch"
        # Untouched tail of a partial last block must keep the sentinel.
        if n_written < _BLOCK_SIZE_BYTES:
            tail = worker._cpu_blocks[block_idx, n_written:]
            assert torch.all(tail == sentinel), (
                f"block {block_idx} tail was overwritten"
            )

    # Blocks the scheduler didn't assign must be entirely untouched.
    for idx in set(range(_NUM_BLOCKS)) - set(block_indices):
        assert torch.all(worker._cpu_blocks[idx] == sentinel), (
            f"block {idx} (unassigned) was overwritten"
        )


def test_save_caches_noop_when_mm_hash_not_in_saves(make_worker):
    """When the scheduler hasn't pre-allocated blocks for ``mm_hash`` (the
    request opted out, or this is a consumer-only node), ``save_caches`` is
    a pure no-op — the early return must trigger before
    ``_ensure_copy_stream`` is even called.

    No @_requires_cuda: on hosts without a GPU, any accidental CUDA call
    will raise immediately, turning a silent correctness regression
    into an obvious test failure
    """
    worker = make_worker()
    sentinel = 0x42
    worker._cpu_blocks.fill_(sentinel)

    # ``encoder_cache[mm_hash]`` is intentionally absent — the early
    # return must trigger before any dict lookup.
    worker.save_caches({}, "h", _meta(saves={}))

    assert torch.all(worker._cpu_blocks == sentinel)


# ── start_load_caches ────────────────────────────────────────────────────────


@_requires_cuda
def test_start_load_caches_copies_with_correct_shape_dtype_and_bytes(make_worker):
    """``start_load_caches`` materializes ``encoder_cache[mm_hash]`` from
    the bytes at ``meta.loads[mm_hash]`` in the mmap, on the GPU, with
    block ordering preserved end to end.

    Verifies the full byte → dtype → shape pipeline through real CUDA
    stream coordination:
    - the i-th row corresponds to mmap block ``block_indices[i]``;
    - the result is shape ``(n_blocks, hidden_dim)`` with the layout dtype;
    - bytes are visible on the compute stream by the time the test reads
      them
    """
    worker = make_worker()
    n_blocks = 3
    src_orig = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE).reshape(
        n_blocks, _HIDDEN_DIM
    )
    # Same bytes, viewed as int8 to write directly into ``_cpu_blocks``.
    src_int8 = src_orig.view(torch.int8).reshape(n_blocks, _BLOCK_SIZE_BYTES)

    # Scrambled mmap indices catch any reliance on natural block ordering.
    block_indices = [3, 1, 6]
    for i, idx in enumerate(block_indices):
        worker._cpu_blocks[idx].copy_(src_int8[i])

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": block_indices}))

    out = encoder_cache["h"]
    assert out.is_cuda, "consumer worker must place the tensor on the GPU"
    assert out.shape == (n_blocks, _HIDDEN_DIM)
    assert out.dtype == _DTYPE
    # ``out.cpu()`` synchronizes the current stream; the worker's
    # ``current_stream().wait_event`` is the only reason the bytes are
    # well-defined here.
    assert torch.equal(out.cpu(), src_orig)


@_requires_cuda
def test_start_load_caches_preserves_existing_encoder_cache_entry(make_worker):
    """If ``encoder_cache`` already holds the ``mm_hash`` (a fresher local
    encode landed first, or this is a retried step), the worker must not
    overwrite it — silently replacing a live tensor would be a correctness
    bug visible only at attention time.
    """
    worker = make_worker()
    # Plant garbage in the named block so an accidental overwrite would
    # produce values that differ from the sentinel.
    worker._cpu_blocks[0].fill_(0x42)

    sentinel = torch.full((_HIDDEN_DIM,), 7.0, dtype=_DTYPE, device="cuda")
    encoder_cache = {"h": sentinel}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": [0]}))

    assert encoder_cache["h"] is sentinel, (
        "existing encoder_cache entry must not be replaced"
    )


def test_start_load_caches_noop_when_loads_is_empty(make_worker):
    """When ``meta.loads`` is empty the early-return must fire before
    ``_ensure_copy_stream`` is called — mirroring
    ``test_save_caches_noop_when_mm_hash_not_in_saves``.

    No @_requires_cuda: on hosts without a GPU any accidental
    ``torch.cuda.Stream()`` call will raise immediately.
    """
    worker = make_worker()
    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={}))

    assert worker._copy_stream is None
    assert encoder_cache == {}


@_requires_cuda
def test_start_load_caches_skips_cached_and_loads_new_in_same_step(make_worker):
    """When a step carries both already-cached and new mm_hashes in
    ``meta.loads``, the cached entries are preserved and the new ones are
    loaded from the mmap — the ``if mm_hash in encoder_cache: continue``
    branch and the load branch must both fire correctly in one call.
    """
    worker = make_worker()
    n_blocks = 2
    src_orig = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE).reshape(
        n_blocks, _HIDDEN_DIM
    )
    src_int8 = src_orig.view(torch.int8).reshape(n_blocks, _BLOCK_SIZE_BYTES)
    new_block_indices = [4, 2]
    for i, idx in enumerate(new_block_indices):
        worker._cpu_blocks[idx].copy_(src_int8[i])

    cached_tensor = torch.full((1, _HIDDEN_DIM), 99.0, dtype=_DTYPE, device="cuda")
    encoder_cache: dict[str, torch.Tensor] = {"cached_h": cached_tensor}
    worker.start_load_caches(
        encoder_cache,
        _meta(loads={"cached_h": [0], "new_h": new_block_indices}),
    )

    assert encoder_cache["cached_h"] is cached_tensor
    assert "new_h" in encoder_cache
    out = encoder_cache["new_h"]
    assert out.shape == (n_blocks, _HIDDEN_DIM)
    assert out.dtype == _DTYPE
    assert torch.equal(out.cpu(), src_orig)


# ── round-trip ───────────────────────────────────────────────────────────────


@_requires_cuda
def test_save_then_load_round_trips_bytes(make_worker):
    """``save_caches`` followed by ``start_load_caches`` on the same worker
    must reproduce the original tensor exactly.

    This is the only test that exercises the full producer→mmap→consumer
    byte path in one shot: GPU tensor → save_caches → mmap → start_load_caches
    → GPU tensor. Non-monotonic block indices catch any slot/block_idx
    mapping inconsistency between the two directions.
    """
    worker = make_worker()
    n_blocks = 3
    block_indices = [5, 1, 6]

    src = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE, device="cuda").reshape(
        n_blocks, _HIDDEN_DIM
    )
    worker.save_caches({"h": src}, "h", _meta(saves={"h": block_indices}))

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": block_indices}))

    out = encoder_cache["h"]
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert torch.equal(out.cpu(), src.cpu())


# ── stream management ────────────────────────────────────────────────────────


@_requires_cuda
def test_ensure_copy_stream_is_idempotent(make_worker):
    """``_ensure_copy_stream`` must create the stream and event exactly once;
    subsequent calls must be no-ops that leave the existing objects in place.
    """
    worker = make_worker()
    assert worker._copy_stream is None
    assert worker._copy_event is None

    worker._ensure_copy_stream()
    stream = worker._copy_stream
    event = worker._copy_event
    assert stream is not None
    assert event is not None

    worker._ensure_copy_stream()
    assert worker._copy_stream is stream, "_copy_stream was replaced on second call"
    assert worker._copy_event is event, "_copy_event was replaced on second call"


# ── lifecycle ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rank,pin_available,expected_pinned",
    [
        (0, True, True),  # only path that should pin
        (0, False, False),  # rank 0 but platform forbids pinning
        (1, True, False),  # other rank: must NOT re-register the shared range
        (3, True, False),
    ],
    ids=["rank0-available", "rank0-unavailable", "rank1", "rank3"],
)
def test_init_pins_memory_only_on_rank_zero_when_available(
    rank, pin_available, expected_pinned
):
    """``cudaHostRegister`` must run at most once per shared address range.
    The worker enforces this by gating ``pin_memory()`` on rank == 0 AND
    ``is_pin_memory_available()``.
    """
    layout = _make_layout()
    layout.region.pin_memory = MagicMock()  # type: ignore[method-assign]

    with (
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.setup_ec_region",
            return_value=layout,
        ),
        patch(
            "vllm.distributed.ec_transfer.ec_connector.cpu.worker.is_pin_memory_available",
            return_value=pin_available,
        ),
    ):
        ECCPUWorker(_vllm_config(rank=rank))

    try:
        if expected_pinned:
            layout.region.pin_memory.assert_called_once()
        else:
            layout.region.pin_memory.assert_not_called()
    finally:
        layout.region.cleanup()


def test_shutdown_calls_region_cleanup_and_swallows_errors(caplog_vllm):
    """``shutdown`` must always call ``region.cleanup`` — and must never
    raise. Engine teardown should not fail because of a stale mmap or a
    flaky ``cudaHostUnregister``.

    Bypasses ``__init__`` via ``object.__new__`` (same pattern as
    ``test_nixl_connector_hma.py::test_logical_to_kernel_block_ids_with_hma``).
    Only the two fields ``shutdown`` reads need to be set, so this test
    runs anywhere — no real mmap, no CUDA, no pin-memory machinery.
    """
    worker = object.__new__(ECCPUWorker)
    worker._region = Mock(spec=ECSharedRegion)
    worker._cpu_blocks = MagicMock()  # cleared to None by shutdown

    worker.shutdown()
    worker._region.cleanup.assert_called_once()
    assert worker._cpu_blocks is None

    worker._region.cleanup.side_effect = RuntimeError("boom")
    with caplog_vllm.at_level(logging.DEBUG, logger="vllm"):
        worker.shutdown()  # exception must be swallowed

    assert worker._region.cleanup.call_count == 2
    assert any(
        "worker region cleanup failed" in r.message
        for r in caplog_vllm.records
        if r.levelno == logging.DEBUG
    )
