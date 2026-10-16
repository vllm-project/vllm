# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the native KV-offloading scheduler's partial-tail
store path (vLLM issue #61970).

``OffloadingConnectorScheduler._build_partial_tail_store_jobs`` used to crash
the whole EngineCore with ``assert boundary <= max_boundary`` when a hand-off
(CoW) Mamba boundary exceeded the request's current offloadable window, while
the functionally equivalent aligned-boundary path
(``_build_aligned_boundary_store_jobs``) simply skips such out-of-window
entries.  The fix mirrors the aligned path: skip the store job with a
warning instead of asserting.

How these tests work
--------------------
They import the *real* ``offloading/scheduler.py`` module source with its
heavy dependencies (torch, ``vllm.config``, engine machinery, ...) stubbed
out via ``sys.modules``, then drive
``_build_partial_tail_store_jobs`` / ``_build_aligned_boundary_store_jobs``
directly against hand-constructed scheduler state.  No GPU, no engine and no
installed vllm are required.

The pre-fix behavior is verified by loading a second copy of the module whose
source is reverse-patched back to the original assert (``OLD_BLOCK`` below):
on the same input it must raise ``AssertionError`` (red), while the fixed
module skips the job and leaves all scheduler state untouched (green).
Aligned-boundary behavior is asserted to be identical in both variants.
"""

from __future__ import annotations

import logging
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

SCHEDULER_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "offloading"
    / "scheduler.py"
)

# --------------------------------------------------------------------------
# The fix under test (must match scheduler.py verbatim) and its pre-fix form.
# --------------------------------------------------------------------------

NEW_BLOCK = """\
            assert boundary % self.config.tokens_per_hash == 0

            # The hand-off boundary can exceed the request's current
            # offloadable window, e.g. a Mamba boundary-state snapshot
            # retained before the request was resumed with a smaller
            # max_offload_tokens / num_prompt_tokens. The aligned-boundary
            # path above skips such out-of-window entries; mirror it here
            # instead of crashing the EngineCore (issue #61970).
            if boundary > max_boundary:
                logger.warning(
                    "Skipping partial-tail store job for request %s: "
                    "boundary %d exceeds the max offloadable boundary %d "
                    "(num_prompt_tokens=%d, max_offload_tokens=%s).",
                    req_id,
                    boundary,
                    max_boundary,
                    req.num_prompt_tokens,
                    req_status.max_offload_tokens,
                )
                continue

            cow_blocks\
"""

OLD_BLOCK = """\
            assert boundary % self.config.tokens_per_hash == 0
            assert boundary <= max_boundary

            cow_blocks\
"""


# --------------------------------------------------------------------------
# Dependency stubs: everything scheduler.py imports at module scope.
# --------------------------------------------------------------------------


def _install_stubs() -> None:
    def put(name: str, **attrs) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    # Package parents (import machinery consults sys.modules first).
    for pkg in (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading",
        "vllm.utils",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.metrics",
    ):
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        sys.modules[pkg] = mod

    class _Stub:
        pass

    put("vllm.config", VllmConfig=_Stub)
    put("vllm.distributed.kv_events", KVCacheEvent=_Stub)
    put(
        "vllm.distributed.kv_transfer.kv_connector.utils",
        yield_req_data=lambda *a, **k: iter(()),
    )
    put(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorMetadata=_Stub,
    )

    common = put(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common",
        ReqId=str,
        OffloadingConnectorMetadata=_Stub,
        OffloadingWorkerMetadata=_Stub,
    )

    class TransferJob:
        def __init__(self, req_id, src_spec, dst_spec):
            self.req_id = req_id
            self.src_spec = src_spec
            self.dst_spec = dst_spec

    common.TransferJob = TransferJob

    put(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.events",
        OffloadingEventGroupSpec=_Stub,
        OffloadingEventsTracker=_Stub,
        get_offloading_event_group_spec=lambda *a, **k: None,
    )

    class _ConnectorMetricName(Enum):
        ALLOCATION_FAILURE = "allocation_failure"
        LOOKUP_ASYNC_DELAY = "lookup_async_delay"
        LOOKUP_SYNC_DELAY = "lookup_sync_delay"

    put(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics",
        OffloadingConnectorStats=_Stub,
        _ConnectorMetricName=_ConnectorMetricName,
        _TransferMetricName=_ConnectorMetricName,
    )
    put(
        "vllm.logger",
        init_logger=lambda name: logging.getLogger(f"stubbed.{name}"),
    )
    put(
        "vllm.utils.math_utils",
        cdiv=lambda a, b: -(-a // b),
        round_down=lambda x, n: x - x % n,
    )
    put("vllm.v1.core.kv_cache_manager", KVCacheBlocks=_Stub)
    put("vllm.v1.core.sched.output", SchedulerOutput=_Stub)
    put(
        "vllm.v1.core.single_type_kv_cache_manager",
        SingleTypeKVCacheManager=_Stub,
    )

    class FullAttentionSpec(_Stub):
        pass

    class SlidingWindowSpec(_Stub):
        pass

    class MambaSpec(_Stub):
        pass

    class ChunkedLocalAttentionSpec(_Stub):
        pass

    put(
        "vllm.v1.kv_cache_interface",
        ChunkedLocalAttentionSpec=ChunkedLocalAttentionSpec,
        FullAttentionSpec=FullAttentionSpec,
        KVCacheConfig=_Stub,
        KVCacheSpec=_Stub,
        MambaSpec=MambaSpec,
        SlidingWindowSpec=SlidingWindowSpec,
    )
    put("vllm.v1.kv_cache_spec_registry", KVCacheSpecRegistry=_Stub)

    class LookupResult(Enum):
        HIT = "hit"
        HIT_PENDING = "hit_pending"
        RETRY = "retry"
        MISS = "miss"

    class GPULoadStoreSpec:
        # Mirrors vllm.v1.kv_offload.base.GPULoadStoreSpec's constructor.
        def __init__(self, block_ids, group_sizes=None, block_indices=None):
            self.block_ids = list(block_ids)
            self.group_sizes = group_sizes
            self.block_indices = block_indices

    def make_offload_key(block_hash, group_idx):
        # Mirrors vllm.v1.kv_offload.base.make_offload_key.
        return bytes(block_hash) + group_idx.to_bytes(4, "big")

    put(
        "vllm.v1.kv_offload.base",
        GPULoadStoreSpec=GPULoadStoreSpec,
        Locality=_Stub,
        LookupResult=LookupResult,
        Medium=_Stub,
        OffloadingManager=_Stub,
        OffloadingSpec=_Stub,
        OffloadKey=bytes,
        OffloadPolicy=_Stub,
        ReqContext=_Stub,
        RequestOffloadingContext=_Stub,
        ScheduleEndContext=_Stub,
        TierFilter=_Stub,
        TierMatcher=_Stub,
        make_offload_key=make_offload_key,
    )
    put("vllm.v1.metrics.cache_hit_source", CacheHitSource=_Stub)
    put("vllm.v1.outputs", KVConnectorOutput=_Stub)

    class RequestStatus(Enum):
        WAITING = "waiting"
        RUNNING = "running"
        FINISHED_STOPPED = "finished_stopped"
        FINISHED_LENGTH_CAPPED = "finished_length_capped"
        FINISHED_ABORTED = "finished_aborted"
        FINISHED_IGNORED = "finished_ignored"

    put("vllm.v1.request", Request=_Stub, RequestStatus=RequestStatus)


def _load_module(name: str, source: str):
    _install_stubs()
    mod = types.ModuleType(name)
    mod.__file__ = str(SCHEDULER_PATH)
    sys.modules[name] = mod
    exec(compile(source, str(SCHEDULER_PATH), "exec"), mod.__dict__)
    return mod


_FILE_SOURCE = SCHEDULER_PATH.read_text()
IS_FIXED = NEW_BLOCK in _FILE_SOURCE
_OLD_SOURCE = _FILE_SOURCE.replace(NEW_BLOCK, OLD_BLOCK) if IS_FIXED else _FILE_SOURCE

NEW_MOD = _load_module("ut_offloading_scheduler_fixed", _FILE_SOURCE)
OLD_MOD = _load_module("ut_offloading_scheduler_original", _OLD_SOURCE)
STUB_IFACE = sys.modules["vllm.v1.kv_cache_interface"]
STUB_REQUEST = sys.modules["vllm.v1.request"]


# --------------------------------------------------------------------------
# Fakes for the collaborators the two store-job builders touch.
# --------------------------------------------------------------------------


class FakeRequest:
    """Just enough of vllm.v1.request.Request for the store-job builders."""

    def __init__(self, request_id: str, num_prompt_tokens: int, num_tokens: int):
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_tokens = num_tokens
        self.block_hashes = [i.to_bytes(8, "big") for i in range(64)]
        self.kv_transfer_params = {}
        self.status = STUB_REQUEST.RequestStatus.RUNNING

    def is_finished(self) -> bool:
        return False


class FakeReqContext:
    def __init__(self):
        self.key_positions = {}

    def set_offload_key_position(self, key, position):
        self.key_positions[key] = position


class FakeManager:
    def __init__(self):
        self.prepare_store_calls = []

    def prepare_store(self, keys, req_context):
        self.prepare_store_calls.append(list(keys))
        return SimpleNamespace(
            keys_to_store=list(keys), store_spec=SimpleNamespace(tag="store")
        )


class FakeStats:
    def __init__(self):
        self.counters = {}

    def increase_counter(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1

    def observe_histogram(self, name, value):
        pass


class FakeTracker:
    def __init__(self):
        self.records = []

    def record_store(self, req, group_config, chunk_idx, key):
        self.records.append(("store", group_config.group_idx, chunk_idx, key))

    def record_partial_store(self, req, group_config, boundary, key):
        self.records.append(("partial_store", group_config.group_idx, boundary, key))


class FakeBlockState:
    def __init__(self, handoffs):
        self.boundary_state_offloads = handoffs
        self.req_ids = set(handoffs)
        self.resolve_block_ids = lambda req_id: ()


class FakeSchedulerOutput:
    def __init__(self, handoffs):
        self.kv_connector_block_state = FakeBlockState(handoffs)


class Fixture:
    """A scheduler instance plus the per-request state it operates on."""

    def __init__(self, mod, max_offload_tokens, handoffs, attn_block_ids):
        self.mod = mod
        self.manager = FakeManager()
        self.stats = FakeStats()
        self.tracker = FakeTracker()

        group_full = mod.GroupOffloadConfig(
            group_idx=0,
            tokens_per_block=16,
            tokens_per_chunk=16,
            hashes_per_chunk=2,
            kv_event_group_spec=None,
            sliding_window_size_in_chunks=None,
            kv_cache_spec=STUB_IFACE.FullAttentionSpec(),
            manager_cls=object,
            requires_cow_source=False,
            is_eagle_group=False,
        )
        group_mamba = mod.GroupOffloadConfig(
            group_idx=1,
            tokens_per_block=16,
            tokens_per_chunk=16,
            hashes_per_chunk=2,
            kv_event_group_spec=None,
            sliding_window_size_in_chunks=None,
            kv_cache_spec=STUB_IFACE.MambaSpec(),
            manager_cls=object,
            requires_cow_source=True,
            is_eagle_group=False,
        )
        self.config = mod.SchedulerOffloadConfig(
            kv_group_configs=(group_full, group_mamba),
            blocks_per_chunk=1,
            tokens_per_hash=8,
            num_workers=1,
            offload_prompt_only=False,
            supports_partial_tail=True,
        )

        self.req = FakeRequest("req-1", num_prompt_tokens=128, num_tokens=128)
        self.req_context = FakeReqContext()
        self.req_status = mod.RequestOffloadState(
            config=self.config,
            req=self.req,
            req_context=self.req_context,
            offloading_context=SimpleNamespace(),
        )
        self.req_status.max_offload_tokens = max_offload_tokens
        # Non-CoW groups read the block holding the partial tail from the
        # request's own block table (group 0 = full attention here).
        self.req_status.group_states[0].block_ids = list(attn_block_ids)

        sched = mod.OffloadingConnectorScheduler.__new__(
            mod.OffloadingConnectorScheduler
        )
        sched.config = self.config
        sched._req_status = {"req-1": self.req_status}
        sched._partial_tail_block_size = group_full.tokens_per_block
        sched._cow_source_groups = frozenset({group_mamba.group_idx})
        sched.manager = self.manager
        sched._connector_stats = self.stats
        sched._events_tracker = self.tracker
        sched._job_counter = 0
        sched._jobs = {}
        sched._block_id_to_pending_jobs = {}
        self.scheduler = sched

        self.output = FakeSchedulerOutput(dict(handoffs))

    def call(self):
        return self.scheduler._build_partial_tail_store_jobs(self.output)


def _bug_fixture(mod) -> Fixture:
    """The issue #61970 scenario.

    A retained Mamba boundary-state hand-off at boundary=40 (hash-aligned:
    40 % tokens_per_hash == 0; partial: 40 % 16 != 0) while the request's
    current offloadable window caps out at 32 (max_offload_tokens=32,
    num_prompt_tokens=128) -- i.e. boundary > max_boundary.
    """
    return Fixture(
        mod,
        max_offload_tokens=32,
        handoffs={"req-1": [(0, 101, 40), (1, 201, 40)]},
        attn_block_ids=[11, 12, 13, 14],
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_fix_hunk_present():
    """Canary: the file under test actually contains the fix."""
    assert IS_FIXED, (
        "scheduler.py does not contain the issue #61970 fix hunk; "
        "the 'green' tests below would silently test the buggy code"
    )


def test_issue_61970_old_code_asserts():
    """Red: the pre-fix module crashes the EngineCore on the same input."""
    fixture = _bug_fixture(OLD_MOD)
    with pytest.raises(AssertionError):
        fixture.call()


def test_issue_61970_new_code_skips_job(caplog):
    """Green: the fixed module skips the store job and stays consistent."""
    fixture = _bug_fixture(NEW_MOD)
    with caplog.at_level(logging.WARNING):
        jobs = fixture.call()

    # No store job was built ...
    assert jobs == {}
    assert fixture.scheduler._jobs == {}
    assert fixture.scheduler._block_id_to_pending_jobs == {}
    assert fixture.req_status.transfer_jobs == set()
    # ... neither by the aligned pre-pass (boundary is unaligned to the
    # 16-token chunk) nor by the skipped partial-tail path.
    assert fixture.manager.prepare_store_calls == []
    assert fixture.stats.counters == {}
    assert fixture.tracker.records == []
    # No job IDs were burned either.
    assert fixture.scheduler._job_counter == 0
    # The skip is observable: a warning naming the request and boundary.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "Skipping partial-tail store job" in r.getMessage()
        and "req-1" in r.getMessage()
        and "boundary 40" in r.getMessage()
        for r in warnings
    ), [r.getMessage() for r in warnings]


def test_partial_tail_within_max_boundary_still_stores():
    """Positive control: the fix does not skip legitimate partial tails."""
    fixture = Fixture(
        NEW_MOD,
        max_offload_tokens=None,  # max_boundary = num_prompt_tokens = 128
        handoffs={"req-1": [(0, 101, 40), (1, 201, 40)]},
        attn_block_ids=[11, 12, 13, 14],
    )
    jobs = fixture.call()

    assert len(jobs) == 1
    (job_id, job) = next(iter(jobs.items()))

    # CoW group (1) stores its handed-off block; the attention group (0)
    # stores the block holding token boundary 40 -> block index 40 // 16 = 2.
    assert job.req_id == "req-1"
    assert job.src_spec.block_ids == [13, 201]
    assert job.src_spec.group_sizes == [1, 1]
    assert job.src_spec.block_indices == [2, 2]

    hash_idx = 40 // 8 - 1  # tokens_per_hash == 8
    key0 = fixture.req.block_hashes[hash_idx] + (0).to_bytes(4, "big")
    key1 = fixture.req.block_hashes[hash_idx] + (1).to_bytes(4, "big")

    status = fixture.scheduler._jobs[job_id]
    assert status.req_id == "req-1"
    assert status.is_store
    assert status.pending_count == 1  # num_workers
    assert status.keys == {key0, key1}
    assert status.fenced_block_ids == [13, 201]

    assert fixture.req_status.transfer_jobs == {job_id}
    assert fixture.scheduler._block_id_to_pending_jobs == {
        13: {job_id},
        201: {job_id},
    }
    assert fixture.tracker.records == [
        ("partial_store", 0, 40, key0),
        ("partial_store", 1, 40, key1),
    ]
    assert fixture.req_context.key_positions == {key0: 40, key1: 40}
    assert fixture.manager.prepare_store_calls == [[key0, key1]]


def test_hash_alignment_invariant_still_enforced():
    """The fix must not swallow genuine invariants: a boundary that is not
    hash-aligned still fails loudly instead of being skipped."""
    fixture = Fixture(
        NEW_MOD,
        max_offload_tokens=None,
        handoffs={"req-1": [(0, 101, 44), (1, 201, 44)]},  # 44 % 8 != 0
        attn_block_ids=[11, 12, 13, 14],
    )
    with pytest.raises(AssertionError):
        fixture.call()


@pytest.mark.parametrize("mod_name", ["old", "new"])
def test_aligned_boundary_over_max_is_skipped_unchanged(mod_name):
    """Aligned-path regression: an out-of-window aligned boundary (48 > 32)
    is skipped silently -- identically before and after the fix."""
    mod = OLD_MOD if mod_name == "old" else NEW_MOD
    fixture = Fixture(
        mod,
        max_offload_tokens=32,
        handoffs={"req-1": [(0, 101, 48), (1, 201, 48)]},
        attn_block_ids=[11, 12, 13, 14],
    )
    jobs = fixture.call()

    assert jobs == {}
    assert fixture.scheduler._jobs == {}
    assert fixture.req_status.transfer_jobs == set()
    assert fixture.manager.prepare_store_calls == []
    assert fixture.stats.counters == {}


@pytest.mark.parametrize("mod_name", ["old", "new"])
def test_aligned_boundary_within_max_stores_unchanged(mod_name):
    """Aligned-path regression: an in-window aligned boundary (32 <= 32)
    produces one store job per group -- identically before and after."""
    mod = OLD_MOD if mod_name == "old" else NEW_MOD
    fixture = Fixture(
        mod,
        max_offload_tokens=32,
        handoffs={"req-1": [(0, 101, 32), (1, 201, 32)]},
        attn_block_ids=[11, 12, 13, 14],
    )
    jobs = fixture.call()

    assert len(jobs) == 2  # one job per group entry
    hash_idx = 32 // 8 - 1

    by_src_block = {}
    for job_id, job in jobs.items():
        assert job.req_id == "req-1"
        assert len(job.src_spec.block_ids) == 1
        by_src_block[job.src_spec.block_ids[0]] = (job_id, job)

    for group_idx, block_id in ((0, 101), (1, 201)):
        job_id, job = by_src_block[block_id]
        assert job.src_spec.group_sizes[group_idx] == 1
        assert sum(job.src_spec.group_sizes) == 1
        assert job.src_spec.block_indices[group_idx] == (
            32 // 16 - 1
        )  # boundary // tokens_per_block - 1
        status = fixture.scheduler._jobs[job_id]
        assert status.is_store
        expected_key = fixture.req.block_hashes[hash_idx] + group_idx.to_bytes(4, "big")
        assert status.keys == {expected_key}

    assert fixture.req_status.transfer_jobs == set(jobs)
    assert fixture.scheduler._job_counter == 2
