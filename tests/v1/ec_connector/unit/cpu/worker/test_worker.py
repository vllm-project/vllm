# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ECCPUWorker.

The byte-level tests exercise real CUDA stream/event coordination
against a real ``ECSharedRegion`` mmap and are skipped on hosts without
CUDA. The lifecycle tests don't fire any ``torch.cuda.*`` primitives and
run anywhere.

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
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker

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


def _make_region() -> ECSharedRegion:
    """Fresh region backed by a real per-test mmap file."""
    return ECSharedRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=_NUM_BLOCKS,
        block_size_bytes=_BLOCK_SIZE_BYTES,
    )


def _vllm_config(rank: int = 0) -> Mock:
    cfg = Mock(spec=VllmConfig)
    cfg.parallel_config = Mock(spec=ParallelConfig)
    cfg.parallel_config.rank = rank
    cfg.model_config = Mock()
    cfg.model_config.dtype = _DTYPE
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
    """``save_caches`` + ``flush_saves`` copies the source GPU tensor's bytes
    into the block IDs named by ``meta.saves``, in the order they appear,
    via a single batched GPU→CPU copy.

    When ``total_bytes`` is not a multiple of ``block_size_bytes`` the last
    block is partially written; the unwritten tail must remain whatever
    was there before. Blocks not named in ``meta.saves`` must be untouched.
    """
    worker = make_worker()
    sentinel = 0x5A
    worker._region.blocks.fill_(sentinel)

    src = torch.arange(n_elements, dtype=_DTYPE, device="cuda")
    expected_bytes = src.cpu().reshape(-1).view(torch.uint8)
    total_bytes = n_elements * _DTYPE.itemsize

    block_ids = [7, 2, 5, 0][:n_blocks]
    worker.save_caches({"h": src}, "h", _meta(saves={"h": block_ids}))
    worker.flush_saves()

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


@_requires_cuda
def test_save_caches_batches_multiple_hashes(make_worker):
    """Multiple save_caches calls are batched into a single flush."""
    worker = make_worker()
    sentinel = 0x5A
    worker._region.blocks.fill_(sentinel)

    src_a = torch.arange(_HIDDEN_DIM, dtype=_DTYPE, device="cuda")
    src_b = torch.arange(_HIDDEN_DIM, 2 * _HIDDEN_DIM, dtype=_DTYPE, device="cuda")

    cache = {"a": src_a, "b": src_b}
    worker.save_caches(cache, "a", _meta(saves={"a": [1], "b": [3]}))
    worker.save_caches(cache, "b", _meta(saves={"a": [1], "b": [3]}))
    worker.flush_saves()

    expected_a = src_a.cpu().view(torch.uint8)
    expected_b = src_b.cpu().view(torch.uint8)
    actual_a = worker._region.blocks[1].view(torch.uint8)
    actual_b = worker._region.blocks[3].view(torch.uint8)
    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_b, expected_b)

    # Unassigned blocks untouched.
    for idx in (0, 2, 4, 5, 6, 7):
        assert torch.all(worker._region.blocks[idx] == sentinel)


# ── start_load_caches ────────────────────────────────────────────────────────


@_requires_cuda
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
    assert out.is_cuda, "consumer worker must place the tensor on the GPU"
    assert out.shape == (n_blocks, _HIDDEN_DIM)
    assert out.dtype == _DTYPE
    assert torch.equal(out.cpu(), src_orig)


@_requires_cuda
def test_start_load_caches_preserves_existing_encoder_cache_entry(make_worker):
    """If ``encoder_cache`` already holds the ``mm_hash``, the worker must
    not overwrite it."""
    worker = make_worker()
    worker._region.blocks[0].fill_(0x42)

    sentinel = torch.full((_HIDDEN_DIM,), 7.0, dtype=_DTYPE, device="cuda")
    encoder_cache = {"h": sentinel}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": [0]}))

    assert encoder_cache["h"] is sentinel, (
        "existing encoder_cache entry must not be replaced"
    )


@_requires_cuda
def test_start_load_caches_noop_when_loads_is_empty(make_worker):
    """When ``meta.loads`` is empty the early-return must fire."""
    worker = make_worker()
    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={}))

    assert encoder_cache == {}


@_requires_cuda
def test_start_load_caches_skips_cached_and_loads_new_in_same_step(make_worker):
    """Cached entries are preserved while new ones are loaded."""
    worker = make_worker()
    n_blocks = 2
    src_orig = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE).reshape(
        n_blocks, _HIDDEN_DIM
    )
    src_int8 = src_orig.view(torch.int8).reshape(n_blocks, _BLOCK_SIZE_BYTES)
    new_block_ids = [4, 2]
    for i, idx in enumerate(new_block_ids):
        worker._region.blocks[idx].copy_(src_int8[i])

    cached_tensor = torch.full((1, _HIDDEN_DIM), 99.0, dtype=_DTYPE, device="cuda")
    encoder_cache: dict[str, torch.Tensor] = {"cached_h": cached_tensor}
    worker.start_load_caches(
        encoder_cache,
        _meta(loads={"cached_h": [0], "new_h": new_block_ids}),
    )

    assert encoder_cache["cached_h"] is cached_tensor
    assert "new_h" in encoder_cache
    out = encoder_cache["new_h"]
    assert out.shape == (n_blocks, _HIDDEN_DIM)
    assert out.dtype == _DTYPE
    assert torch.equal(out.cpu(), src_orig)


@_requires_cuda
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
    assert out.is_cuda
    assert torch.equal(out.cpu(), src_orig)


# ── round-trip ───────────────────────────────────────────────────────────────


@_requires_cuda
def test_save_then_load_round_trips_bytes(make_worker):
    """Full producer→mmap→consumer byte path in one shot."""
    worker = make_worker()
    n_blocks = 3
    block_ids = [5, 1, 6]

    src = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE, device="cuda").reshape(
        n_blocks, _HIDDEN_DIM
    )
    worker.save_caches({"h": src}, "h", _meta(saves={"h": block_ids}))
    worker.flush_saves()

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={"h": block_ids}))

    out = encoder_cache["h"]
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert torch.equal(out.cpu(), src.cpu())


# ── buffer recycling ────────────────────────────────────────────────────────


@_requires_cuda
def test_buffer_pool_is_reused_across_save_steps(make_worker):
    """After flush_saves, descriptor buffers are returned to the pool and
    reused on the next flush — no reallocation."""
    worker = make_worker()
    src = torch.arange(_HIDDEN_DIM, dtype=_DTYPE, device="cuda")

    worker.save_caches({"h": src}, "h", _meta(saves={"h": [0]}))
    worker.flush_saves()

    assert len(worker._buf_pool._pool) == 1
    buf_id = id(worker._buf_pool._pool[0].src_ptrs)

    # Second step reuses the same buffer.
    worker.save_caches({"h": src}, "h", _meta(saves={"h": [1]}))
    worker.flush_saves()

    assert len(worker._buf_pool._pool) == 1
    assert id(worker._buf_pool._pool[0].src_ptrs) == buf_id


@_requires_cuda
def test_buffer_pool_is_reused_across_load_steps(make_worker):
    """After start_load_caches, descriptor buffers are returned to the pool
    and reused on the next call."""
    worker = make_worker()
    worker._region.blocks[0].fill_(0x01)
    worker._region.blocks[1].fill_(0x02)

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, _meta(loads={"a": [0]}))

    assert len(worker._buf_pool._pool) == 1
    buf_id = id(worker._buf_pool._pool[0].src_ptrs)

    encoder_cache2: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache2, _meta(loads={"b": [1]}))

    assert len(worker._buf_pool._pool) == 1
    assert id(worker._buf_pool._pool[0].src_ptrs) == buf_id


# ── stream management ────────────────────────────────────────────────────────


@_requires_cuda
def test_stream_initialized_at_construction(make_worker):
    """``_load_stream`` must be a fully initialized CUDA stream."""
    worker = make_worker()
    assert isinstance(worker._load_stream, torch.cuda.Stream)


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
    worker._load_stream = MagicMock()
    worker._save_bufs = None
    worker._save_count = 0

    worker.shutdown()
    worker._load_stream.synchronize.assert_called_once()
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


@_requires_cuda
def test_e2e_scheduler_worker_save_then_load(make_worker, monkeypatch):
    """Full pipeline: scheduler allocates blocks, worker saves GPU tensor to
    mmap via flush_saves, scheduler marks ready after step delay, worker
    loads from mmap back to GPU, and the result matches the original.

    Exercises the real scheduler + worker cooperation through a shared
    ECSharedRegion, with real CUDA transfers and stream coordination.
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

    class _EC:
        is_ec_producer = True
        is_ec_consumer = True

    class _Cfg:
        ec_transfer_config = _EC()
        max_concurrent_batches = 1

    scheduler = ECCPUScheduler(_Cfg())

    # -- Step 1: scheduler allocates, worker saves --
    n_blocks = 3
    src = torch.arange(n_blocks * _HIDDEN_DIM, dtype=_DTYPE, device="cuda").reshape(
        n_blocks, _HIDDEN_DIM
    )

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

    # -- Step 2: readiness delay (max_concurrent_batches=1) --
    meta_step2 = scheduler.build_connector_meta(scheduler_output=None)
    assert scheduler.has_cache_item("img_001") is True
    assert meta_step2.loads == {}  # no load requested yet

    # -- Step 3: scheduler emits load, worker loads --
    scheduler.update_state_after_alloc(_Request(), 0)
    meta_load = scheduler.build_connector_meta(scheduler_output=None)
    assert "img_001" in meta_load.loads
    assert meta_load.loads["img_001"] == meta_save.saves["img_001"]

    load_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(load_cache, meta_load)

    out = load_cache["img_001"]
    assert out.is_cuda
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert torch.equal(out.cpu(), src.cpu())

    scheduler.shutdown()
