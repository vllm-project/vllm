# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
from collections.abc import Collection, Iterable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("kvcr")

from kvcr import KVCRBindings
from kvcr.config import G3Options, KVCRBackendConfigs, KVCRConfig, KVCRGuardConfig
from kvcr.policy import FIFOPolicy, G3FIFOPolicy, G3LRUPolicy, LRUPolicy
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    KVCRStartupError,
    MemoryRef,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    QueryStatus,
)

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.backpressure import EMABackpressureDetector
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.kvcr import manager as kvcr_manager
from vllm.v1.kv_offload.tiering.kvcr.manager import KVCRSecondaryTierManager


def _op_entries(
    entries: Mapping[BlockKey, bool],
) -> dict[BlockKey, OpEntryResult]:
    return {
        key: OpEntryResult(OpEntryStatus.SUCCESS if success else OpEntryStatus.FAILED)
        for key, success in entries.items()
    }


class _StubControlChannel:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def send(self, endpoint: str, message: bytes) -> bool:
        return True

    def recv(self) -> list[bytes]:
        return []

    def close(self) -> None:
        pass


class RecordingKVCR:
    def __init__(self) -> None:
        self.config: KVCRConfig | None = None
        self.guard_config: KVCRGuardConfig | None = None
        self.backend_configs = KVCRBackendConfigs()
        self.constructor_bindings: KVCRBindings | None = None
        self.nixl_agent_name = "recording"
        self.framework_control: _StubControlChannel | None = None
        self.inventory_sink = None
        self.query_status = QueryStatus.MISS
        self.query_calls = []
        self.stats: OffloadingConnectorStats | None = None
        self.submit_hint_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.discard_hint_calls: list[str] = []
        self.align_sequence_calls: list[tuple[list[BlockKey], bool]] = []
        self.deliver_calls: list[
            tuple[OpHandle, dict[BlockKey, list[MemoryRef]], str | None]
        ] = []
        self.deposit_calls: list[tuple[OpHandle, dict[BlockKey, list[MemoryRef]]]] = []
        self.completed: list[tuple[OpHandle, dict[BlockKey, OpEntryResult]]] = []
        self._next_op_handle = 1

    def submit_hint(self, *args: Any, **kwargs: Any) -> None:
        self.submit_hint_calls.append((args, kwargs))

    def discard_hint(self, request_id: str) -> None:
        self.discard_hint_calls.append(request_id)

    def align_sequence(
        self, keys: list[BlockKey], use_current_time: bool = False
    ) -> None:
        self.align_sequence_calls.append((list(keys), use_current_time))

    def query(
        self,
        keys: Collection[BlockKey],
        request_id: str | None = None,
    ) -> list[tuple[QueryStatus, CacheTier | None]]:
        self.query_calls.append((tuple(keys), request_id))
        tier = {
            QueryStatus.HIT: CacheTier.LOCAL_G2,
            QueryStatus.FETCHING: CacheTier.LOCAL_G2,
            QueryStatus.FETCHABLE: CacheTier.REMOTE_G2,
            QueryStatus.MISS: None,
        }[self.query_status]
        return [(self.query_status, tier) for _ in keys]

    def deliver(
        self,
        blocks: Mapping[BlockKey, list[MemoryRef]],
        request_id: str | None = None,
    ) -> OpHandle:
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        self.deliver_calls.append((op_handle, dict(blocks), request_id))
        self.complete(op_handle, {key: True for key in blocks})
        return op_handle

    def deposit(
        self,
        blocks: Mapping[BlockKey, list[MemoryRef]],
    ) -> OpHandle:
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        self.deposit_calls.append((op_handle, dict(blocks)))
        self.complete(op_handle, {key: True for key in blocks})
        return op_handle

    def complete(
        self,
        op_handle: OpHandle,
        entries: Mapping[BlockKey, bool],
    ) -> None:
        self.completed.append((op_handle, _op_entries(entries)))

    def poll_completed(
        self,
    ) -> Iterable[tuple[OpHandle, dict[BlockKey, OpEntryResult]]]:
        completed = self.completed
        self.completed = []
        return completed

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self.stats
        self.stats = None
        return stats

    def close(self) -> None:
        pass


class _ExternalPolicy(FIFOPolicy):
    pass


def _job(
    job_id: int,
    req_context: ReqContext,
    key: OffloadKey | None = None,
    chunk_id: int = 0,
) -> TransferJob:
    return TransferJob(
        job_id=job_id,
        keys=[key if key is not None else OffloadKey(b"k0")],
        chunk_ids=np.array([chunk_id], dtype=np.int64),
        is_promotion=True,
        req_context=req_context,
    )


def _make_tier(
    monkeypatch,
    kvcr: RecordingKVCR,
    *,
    startup_error: BaseException | None = None,
    primary_kv_view: memoryview | None = None,
    enable_telemetry: bool = False,
    secondary_g2_slots: int = 0,
    kvcr_service_socket_path: str | None = None,
    compatibility_digest: str | None = None,
    enable_kv_cache_events: bool = False,
    self_describing_kv_events: bool = False,
    policy: str | None = None,
    g3: dict[str, object] | None = None,
    control_ports: list[int] | None = None,
    data_parallel_rank_local: int | None = None,
    backpressure: dict[str, Any] | None = None,
    opportunistic_query: bool = True,
) -> KVCRSecondaryTierManager:
    def make_control(_bind_host, bind_port, advertise_host):
        return _StubControlChannel(f"tcp://{advertise_host}:{int(bind_port)}")

    def make_kvcr(config, bindings, backend_configs, guard_config):
        kvcr.config = config
        kvcr.guard_config = guard_config
        kvcr.nixl_agent_name = config.nixl_agent_name
        kvcr.backend_configs = backend_configs
        kvcr.constructor_bindings = bindings
        kvcr.framework_control = bindings.framework_control
        kvcr.inventory_sink = bindings.inventory_sink
        if startup_error is not None:
            raise startup_error
        return kvcr

    monkeypatch.setattr(kvcr_manager, "KVCR", make_kvcr)
    monkeypatch.setattr(kvcr_manager, "ZmqPeerControlChannel", make_control)
    tier_config = {
        "type": "kvcr",
        "opportunistic_query": opportunistic_query,
        "router_capabilities": ["router_hint"],
        "control_host": "127.0.0.1",
        "control_ports": control_ports if control_ports is not None else [7777],
        "control_advertise_host": "127.0.0.1",
        "enable_telemetry": enable_telemetry,
        "secondary_g2_slots": secondary_g2_slots,
        "kvcr_service_socket_path": kvcr_service_socket_path,
        "compatibility_digest": compatibility_digest,
        "policy": policy,
        "g3": g3,
        "local_dram_backend": "UCX",
        "remote_fw_dram_backend": "UCX",
    }
    if backpressure is not None:
        tier_config["backpressure"] = backpressure
    tier = SecondaryTierFactory.create_secondary_tier(
        tier_config=tier_config,
        offloading_spec=SimpleNamespace(
            config=SimpleNamespace(
                parallel=SimpleNamespace(
                    data_parallel_rank_local=data_parallel_rank_local,
                )
            ),
            kv_events_config=SimpleNamespace(
                enable_kv_cache_events=enable_kv_cache_events,
                self_describing_kv_events=self_describing_kv_events,
            ),
        ),
        primary_kv_view=(
            primary_kv_view
            if primary_kv_view is not None
            else memoryview(np.zeros((4, 16), dtype=np.int8))
        ),
    )
    assert isinstance(tier, KVCRSecondaryTierManager)
    return tier


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KVCRStartupError, KeyboardInterrupt]
)
@pytest.mark.parametrize("secondary_g2_slots", [0, 1])
def test_startup_failure_cleanup(monkeypatch, error_type, secondary_g2_slots):
    """Unsafe failures must hold a buffer export so nothing can unmap memory
    native work may still write into. Ordinary failures release everything."""
    retained: list[object] = []
    monkeypatch.setattr(kvcr_manager, "_RETAINED_NATIVE_BUFFERS", retained)
    close = Mock()
    monkeypatch.setattr(_StubControlChannel, "close", close)
    unsafe = error_type is not RuntimeError

    mapping = mmap.mmap(-1, mmap.PAGESIZE)
    primary_kv_view = memoryview(mapping)
    local_mapping = mmap.mmap(-1, mmap.PAGESIZE) if secondary_g2_slots else None
    monkeypatch.setattr(
        kvcr_manager, "mmap", SimpleNamespace(mmap=lambda *_: local_mapping)
    )

    with pytest.raises(error_type, match="startup failed"):
        _make_tier(
            monkeypatch,
            RecordingKVCR(),
            startup_error=error_type("startup failed"),
            primary_kv_view=primary_kv_view,
            secondary_g2_slots=secondary_g2_slots,
        )

    assert close.call_count == (0 if unsafe else 1)
    # Drop the test's own export, leaving only whatever the manager retained:
    # the mapping is unmappable afterwards only if that retention covers it.
    primary_kv_view.release()
    mappings = [mapping] if local_mapping is None else [mapping, local_mapping]
    if local_mapping is not None:
        assert local_mapping.closed is not unsafe
    for region in mappings:
        if unsafe:
            with pytest.raises(BufferError):
                region.close()
    retained.clear()
    for region in mappings:
        region.close()


def test_kvcr_tier_configures_service_for_local_dp_rank(monkeypatch):
    """Keep the control endpoint and guard pool aligned to the local DP rank."""
    kvcr = RecordingKVCR()
    tier = _make_tier(
        monkeypatch,
        kvcr,
        kvcr_service_socket_path="/tmp/kvcr.sock",
        compatibility_digest="Opaque-Digest",
        secondary_g2_slots=1,
        control_ports=[7001, 7002],
        data_parallel_rank_local=1,
    )

    assert kvcr.framework_control is not None
    assert kvcr.framework_control.endpoint == "tcp://127.0.0.1:7002"
    assert kvcr.guard_config == KVCRGuardConfig(
        kvcr_service_socket_path="/tmp/kvcr.sock",
        guard_index=1,
        compatibility_digest="Opaque-Digest",
    )
    assert kvcr.config is not None
    assert kvcr.config.pool_layouts == [("", tier.block_size_bytes)]
    (region,) = kvcr.backend_configs.framework_regions
    assert region.addr == np.asarray(tier._primary_kv_view).ctypes.data
    assert (region.size, region.stride or region.size, region.count) == (16, 16, 4)
    assert (region.label, region.mem_type, region.device_Id) == ("", "DRAM", 0)
    assert kvcr.backend_configs.local_dram is None


def test_kvcr_tier_converts_g3_paths(monkeypatch, tmp_path):
    """Convert user-provided G3 paths to KVCR's typed configuration."""
    kvcr = RecordingKVCR()
    path = tmp_path / "g3.data"

    _make_tier(
        monkeypatch,
        kvcr,
        g3={"paths": [str(path)], "capacity_bytes_per_file": 64},
    )

    assert kvcr.backend_configs.g3 == G3Options(
        paths=(path,),
        capacity_bytes_per_file=64,
    )


@pytest.mark.parametrize(
    "backpressure",
    [None, {"backpressure_cls": "EMABackpressureDetector"}],
    ids=["default", "backpressure"],
)
def test_kvcr_tier_adapts_request_and_load(monkeypatch, backpressure):
    """Factory-created tiers preserve configuration and complete hinted loads."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, backpressure=backpressure)
    if backpressure is None:
        assert tier.bp_detector is None
    else:
        assert isinstance(tier.bp_detector, EMABackpressureDetector)
    kv_hint = KvHintsEnvelope(protocol_version="0.1", message_id="msg", actions=[])
    ctx = ReqContext(
        req_id="req",
        kv_hints=kv_hint,
    )
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    same_hash_other_group = make_offload_key((123).to_bytes(8, "big"), 7)
    other_key = make_offload_key((124).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)

    assert kvcr.submit_hint_calls == [((), {"request_id": "req", "hints": kv_hint})]

    bindings = kvcr.constructor_bindings
    assert bindings is not None
    assert bindings.key_adapter is not None
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    encode_to_router = bindings.key_adapter.encode_to_router
    hash_123 = (123).to_bytes(8, "big")
    assert encode_to_router(BlockKey(bytes(key))) == hash_123
    assert encode_to_router(BlockKey(bytes(same_hash_other_group))) == hash_123
    assert encode_to_router(BlockKey(bytes(other_key))) == (124).to_bytes(8, "big")

    tier.submit_load(_job(7, ctx, key=key, chunk_id=2))

    # Check load destinations and job completion independently of hint parsing.
    assert len(kvcr.submit_hint_calls) == 1
    _, blocks, request_id = kvcr.deliver_calls[0]
    assert request_id == "req"
    assert list(blocks) == [key]
    (descriptor,) = blocks[key]
    assert descriptor == MemoryRef(end_point_name=kvcr.nixl_agent_name, element_index=2)
    assert type(descriptor.element_index) is int
    assert list(tier.get_finished_jobs()) == [JobResult(7, True)]

    # Here we verify that request cleanup discards the request-scoped hint in
    # KVCR.
    tier.on_request_finished(ctx)
    assert kvcr.discard_hint_calls == ["req"]


@pytest.mark.parametrize(
    ("computed", "in_flight", "total_tokens", "expected_keys"),
    [
        (None, None, None, 5),
        (32, 16, 64, 1),
        (16, 16, 64, 0),
        (64, 0, 64, 4),
        (80, 0, 80, 4),
    ],
)
def test_kvcr_tier_refreshes_processed_prefix(
    monkeypatch, computed, in_flight, total_tokens, expected_keys
):
    """Refresh only processed keys on request completion."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)

    ctx = ReqContext(req_id="req")
    if computed is not None:

        class TokenCountContext(ReqContext):
            num_computed_tokens = computed
            num_in_flight_tokens = in_flight
            num_prompt_tokens = 64
            num_tokens = total_tokens

        ctx = TokenCountContext(req_id="req")
    keys = [make_offload_key(bytes([index]), 0) for index in range(5)]
    for position, key in enumerate(keys):
        ctx.set_offload_key_position(key, (position + 1) * 16)
    tier.on_new_request(ctx)
    tier.on_request_finished(ctx)

    assert kvcr.submit_hint_calls == []
    assert kvcr.align_sequence_calls == (
        [(keys[:expected_keys], True)] if expected_keys else []
    )
    assert list(ctx._offload_key_positions) == keys


def test_kvcr_tier_batches_consecutive_completions(monkeypatch):
    """Batch completions without reordering requests that share keys."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    ctx = ReqContext(req_id="req")
    head, middle, tail, future = map(OffloadKey, (b"h", b"m", b"t", b"f"))
    for key, end_token in ((tail, 48), (head, 16), (middle, 32), (future, 64)):
        ctx.set_offload_key_position(key, end_token)
    other = ReqContext(req_id="other")
    other.set_offload_key_position(head, 16)
    tier.submit_store(_job(7, ctx, key=tail))
    tier.submit_load(_job(8, ctx, key=middle))
    tier.submit_store(_job(9, other, key=head))
    tier.submit_load(_job(10, ctx, key=middle))

    assert [job.job_id for job in tier.get_finished_jobs()] == [7, 8, 9, 10]
    assert kvcr.align_sequence_calls == [
        ([head, middle, tail], False),
        ([head], False),
        ([head, middle], False),
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (QueryStatus.MISS, LookupResult.MISS),
        (QueryStatus.HIT, LookupResult.HIT),
        (QueryStatus.FETCHABLE, LookupResult.HIT),
        (QueryStatus.FETCHING, LookupResult.RETRY),
    ],
)
def test_kvcr_tier_maps_query_status(monkeypatch, status, expected):
    """Map KVCR cache states to the scheduler's lookup contract."""
    kvcr = RecordingKVCR()
    kvcr.query_status = status
    tier = _make_tier(monkeypatch, kvcr)

    assert tier.lookup(OffloadKey(b"k0"), ReqContext(req_id="req")) is expected


@pytest.mark.parametrize(
    ("job_keys", "chunk_ids", "success"),
    [
        (None, [1, 3], True),
        ((b"k2", b"k0"), [3, 1], True),
        (None, [1], False),
        ((b"k0", b"other"), [1, 3], False),
    ],
    ids=["original-keys", "reordered-keys", "short-chunks", "unexpected-key"],
)
def test_kvcr_tier_serves_primary_pin_request(
    monkeypatch, job_keys, chunk_ids, success
):
    """Validate pin jobs and hold accepted hits until KVCR releases the pin."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    keys = (BlockKey(b"k0"), BlockKey(b"k1"), BlockKey(b"k2"))
    hit_keys = (keys[0], keys[2])
    lifecycle: list[str] = []

    class Parent:
        def on_new_request(self, req_context):
            lifecycle.append("new")
            return SimpleNamespace()

        def lookup(self, key, req_context):
            return LookupResult.HIT if key in hit_keys else LookupResult.MISS

        def create_store_job(self, requested_keys, req_context):
            return TransferJob(
                job_id=11,
                keys=requested_keys if job_keys is None else job_keys,
                chunk_ids=np.array(chunk_ids, dtype=np.int32),
                is_promotion=True,
                req_context=req_context,
            )

        def on_request_finished(self, req_context):
            lifecycle.append("finished")

    bindings = kvcr.constructor_bindings
    assert bindings is not None
    request = bindings.request_pin(keys)

    tier.serve_external_requests(Parent())

    [(queued_request, result)] = bindings.poll_pin_results()
    assert queued_request == request
    assert lifecycle == ["new", "finished"]
    if not success:
        assert result is None
        assert list(tier.get_finished_jobs()) == [JobResult(11, False)]
        return
    assert result is not None
    pin_handle, descriptors = result
    assert descriptors[keys[1]] is None
    block_descriptors = descriptors[keys[2]]
    assert block_descriptors is not None
    (descriptor,) = block_descriptors
    assert descriptor == MemoryRef(end_point_name=kvcr.nixl_agent_name, element_index=3)
    assert type(descriptor.element_index) is int

    polls = 0

    def release_pin():
        nonlocal polls
        polls += 1
        bindings.release_pin(pin_handle)
        return []

    monkeypatch.setattr(kvcr, "poll_completed", release_pin)
    tier.drain_jobs()

    assert polls == 1
    assert list(tier.get_finished_jobs()) == [JobResult(11, True)]


def test_kvcr_telemetry_is_opt_in_and_namespaced_at_vllm_boundary(monkeypatch):
    """Keep telemetry opt-in and namespace metrics only at the vLLM boundary."""
    assert KVCRSecondaryTierManager.build_metric_definitions({}) == {}
    definitions = KVCRSecondaryTierManager.build_metric_definitions(
        {"enable_telemetry": True}
    )
    assert "vllm:kvcr_duration_seconds" in definitions
    assert "vllm:kvcr_transfer_blocks" in definitions

    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, enable_telemetry=True)
    bindings = kvcr.constructor_bindings
    assert bindings is not None
    assert bindings.stats_factory is not None
    stats = bindings.stats_factory()
    stats.increase_counter(
        "kvcr_transfer_blocks",
        2,
        ("remote_deliver",),
    )
    kvcr.stats = stats

    returned = tier.get_stats()

    assert returned is stats
    assert returned.reduce() == {"vllm:kvcr_transfer_blocks:('remote_deliver',)": 2}


@pytest.mark.parametrize(
    ("policy", "expected_type"),
    [
        ("fifo", FIFOPolicy),
        ("lru", LRUPolicy),
        ("g3_fifo", G3FIFOPolicy),
        ("g3_lru", G3LRUPolicy),
        (f"{__name__}._ExternalPolicy", _ExternalPolicy),
    ],
)
def test_kvcr_tier_passes_selected_policy(monkeypatch, policy, expected_type):
    """Resolve every built-in and fully qualified external policy."""
    kvcr = RecordingKVCR()
    _make_tier(monkeypatch, kvcr, policy=policy)

    bindings = kvcr.constructor_bindings
    assert bindings is not None
    assert type(bindings.policy) is expected_type


@pytest.mark.parametrize(
    ("socket_path", "digest"),
    [("/tmp/kvcr.sock", None), (None, "Opaque-Digest")],
)
def test_kvcr_tier_requires_complete_service_config(monkeypatch, socket_path, digest):
    """Reject partial guard configuration before connecting to the service."""
    with pytest.raises(ValueError, match="configured together"):
        _make_tier(
            monkeypatch,
            RecordingKVCR(),
            kvcr_service_socket_path=socket_path,
            compatibility_digest=digest,
        )


def test_kvcr_tier_stores_and_emits_inventory(monkeypatch):
    """Cover store descriptors and inventory translation at the KVCR boundary."""
    kvcr = RecordingKVCR()
    tier = _make_tier(
        monkeypatch,
        kvcr,
        secondary_g2_slots=2,
        enable_kv_cache_events=True,
        self_describing_kv_events=True,
    )
    local_dram = kvcr.backend_configs.local_dram
    assert local_dram is not None
    assert [(name, size) for name, _, size in local_dram.pools] == [
        ("", 2 * tier.block_size_bytes)
    ]

    key = OffloadKey(b"k0")
    tier.submit_store(_job(11, ReqContext(req_id="req"), key=key, chunk_id=2))

    _, blocks = kvcr.deposit_calls[0]
    assert blocks[key] == [
        MemoryRef(end_point_name=kvcr.nixl_agent_name, element_index=2)
    ]
    assert type(blocks[key][0].element_index) is int
    assert list(tier.get_finished_jobs()) == [JobResult(11, True)]

    assert kvcr.inventory_sink is not None
    kvcr.inventory_sink(
        InventoryEvent((BlockKey(bytes(key)),), CacheTier.LOCAL_G2, False)
    )
    kvcr.inventory_sink(InventoryEvent((BlockKey(bytes(key)),), CacheTier.G3, False))
    events = list(tier.take_events())
    assert [
        (event.keys, event.medium, event.ownership, event.removed) for event in events
    ] == [
        ([key], Medium.CPU, "kvcr", False),
        ([key], Medium.STORAGE, "kvcr", False),
    ]
    assert all(event.removal_expected for event in events)
    tier.shutdown()


def test_kvcr_tier_requires_self_describing_inventory_events(monkeypatch):
    """Require tier-aware events whenever KVCR owns local cache inventory."""
    with pytest.raises(ValueError, match="self_describing_kv_events"):
        _make_tier(
            monkeypatch,
            RecordingKVCR(),
            secondary_g2_slots=1,
            enable_kv_cache_events=True,
        )


@pytest.mark.parametrize(
    ("successes", "expected"),
    [
        ((True, False), JobResult(13, False, {OffloadKey(b"k0")})),
        ((False, True), JobResult(13, False, {OffloadKey(b"k1")})),
        ((True, True), JobResult(13, True)),
    ],
    ids=["success-then-failure", "failure-then-success", "all-success"],
)
def test_kvcr_tier_accumulates_block_results(monkeypatch, successes, expected):
    """Wait for every block result and preserve partial success while draining."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    keys = [OffloadKey(b"k0"), OffloadKey(b"k1")]
    tier.submit_load(
        TransferJob(
            job_id=13,
            keys=keys,
            chunk_ids=np.array([0, 1], dtype=np.int64),
            is_promotion=True,
            req_context=ReqContext(req_id="req"),
        )
    )

    op_handle, _, _ = kvcr.deliver_calls[0]
    kvcr.completed.clear()
    kvcr.complete(op_handle, {keys[0]: successes[0]})
    assert list(tier.get_finished_jobs()) == []

    kvcr.complete(op_handle, {keys[1]: successes[1]})
    tier.drain_jobs()

    assert list(tier.get_finished_jobs()) == [expected]


def test_strict_rejection_keeps_local_lookup_and_dispatch(monkeypatch):
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = ReqContext(req_id="req")
    key = make_offload_key(bytes(32), 0)
    tier.on_new_request(ctx)
    kvcr.query_status = QueryStatus.HIT
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert kvcr.query_calls == [((key,), None)]
    # A local eviction between query and dispatch must not enable remote work.
    kvcr.query_status = QueryStatus.MISS
    tier.submit_load(_job(7, ctx, key))
    assert kvcr.deliver_calls[0][2] is None
    assert list(tier.get_finished_jobs()) == [JobResult(7, True)]


def _hint_context(anchors=(4,), *, source="tcp://source:7000", **payload):
    return ReqContext(
        req_id="req",
        kv_hints=KvHintsEnvelope(
            "0.1",
            "msg",
            [
                KvHintAction(
                    "fetch",
                    "kv.fetch",
                    "1.0",
                    {
                        "source_control_endpoint": source,
                        "block_hashes": list(anchors),
                        **payload,
                    },
                )
            ],
        ),
    )


def _set_chain(ctx, hashes=(1, 2, 3, 4), groups=((0, 16), (1, 32)), **kwargs):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestHashChain,
    )

    ctx.set_state(
        RequestHashChain(
            request_id=ctx.req_id,
            block_hashes=tuple(h.to_bytes(32, "big") for h in hashes),
            tokens_per_hash=16,
            group_chunks=groups,
            anchor_tokens=64,
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    "anchors,payload,authorized",
    [
        ((4,), {}, True),
        ((99,), {}, False),
        ((3,), {}, False),
        ((True,), {}, False),
        ((4,), {"mode": "move"}, False),
        ((4,), {"no_retain": False}, False),
    ],
)
def test_strict_hybrid_prefix_and_mixed_dispatch(
    monkeypatch, anchors, payload, authorized
):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = _hint_context(anchors, **payload)
    _set_chain(ctx)
    tier.on_new_request(ctx)
    fine = make_offload_key((1).to_bytes(32, "big"), 0)
    wrong_group = make_offload_key((1).to_bytes(32, "big"), 1)
    kvcr.query_status = QueryStatus.HIT
    for key in (fine, wrong_group):
        assert tier.lookup(key, ctx) is LookupResult.HIT
    assert [call[1] for call in kvcr.query_calls] == [
        "req" if authorized else None,
        None,
    ]
    if authorized:
        hints = kvcr.submit_hint_calls[-1][1]["hints"]
        assert set(hints.actions[0].payload["block_hashes"]) == {1, 2, 3, 4}
    tier.submit_load(
        TransferJob(
            job_id=7,
            keys=[fine, wrong_group],
            chunk_ids=np.array([0, 1]),
            is_promotion=True,
            req_context=ctx,
        )
    )
    dispatched = {
        key: request for _, blocks, request in kvcr.deliver_calls for key in blocks
    }
    assert dispatched == {fine: "req" if authorized else None, wrong_group: None}
    assert list(tier.get_finished_jobs()) == [JobResult(7, True)]


@pytest.mark.parametrize("conflicting_source", [False, True])
def test_strict_preserves_all_actions_and_last_fetch(monkeypatch, conflicting_source):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    import msgspec

    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = _hint_context()
    unknown = KvHintAction("other", "other", "1.0", {})
    first = ctx.kv_hints.actions[0]
    second = _hint_context(
        (8,), source="tcp://other:7000" if conflicting_source else "tcp://source:7000"
    ).kv_hints.actions[0]
    ctx.kv_hints = msgspec.structs.replace(
        ctx.kv_hints, actions=[unknown, first, second]
    )
    _set_chain(ctx, hashes=tuple(range(1, 9)))
    tier.on_new_request(ctx)
    submitted = kvcr.submit_hint_calls[-1][1]["hints"]
    assert [a.action_type for a in submitted.actions] == [
        "other",
        "kv.fetch",
        "kv.fetch",
    ]
    assert set(submitted.actions[-1].payload["block_hashes"]) == set(range(1, 9))
    key = make_offload_key((7).to_bytes(32, "big"), 0)
    kvcr.query_status = QueryStatus.HIT
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert kvcr.query_calls[-1][1] == (None if conflicting_source else "req")


@pytest.mark.parametrize("hashes", [(1, 2, 3, 4, 4), (1, 2, 3, 4, (1 << 64) + 4)])
def test_strict_ambiguous_anchor_stays_local(monkeypatch, hashes):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = _hint_context()
    _set_chain(ctx, hashes=hashes)
    tier.on_new_request(ctx)
    key = make_offload_key((1).to_bytes(32, "big"), 0)
    kvcr.query_status = QueryStatus.HIT
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert kvcr.query_calls[-1][1] is None
    assert kvcr.submit_hint_calls == []


@pytest.mark.parametrize(
    "change",
    [
        "append",
        "prefix",
        "geometry",
        "ambiguous",
        "hint",
        "shrink",
        "scope",
        "step",
        "anchor",
    ],
)
def test_strict_cached_validation_and_invalidation(monkeypatch, change):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = _hint_context()
    _set_chain(ctx)
    tier.on_new_request(ctx)
    key = make_offload_key((1).to_bytes(32, "big"), 0)
    kvcr.query_status = QueryStatus.HIT
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert len(kvcr.submit_hint_calls) == 1
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert len(kvcr.submit_hint_calls) == 1
    if change == "append":
        _set_chain(ctx, hashes=(1, 2, 3, 4, 5))
    elif change == "prefix":
        _set_chain(ctx, hashes=(9, 2, 3, 4))
    elif change == "geometry":
        _set_chain(ctx, groups=((0, 32), (1, 32)))
    elif change == "ambiguous":
        _set_chain(ctx, hashes=(1, 2, 3, 4, (1 << 64) + 4))
    elif change == "hint":
        ctx.kv_hints.actions[0].payload["block_hashes"] = [99]
    else:
        from dataclasses import replace

        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
            RequestHashChain,
        )

        chain = ctx.get_state(RequestHashChain)
        changes = {
            "shrink": {"block_hashes": chain.block_hashes[:3]},
            "scope": {"request_id": "other"},
            "step": {"tokens_per_hash": 8},
            "anchor": {"anchor_tokens": 32},
        }
        ctx.set_state(replace(chain, **changes[change]))
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert kvcr.query_calls[-1][1] == ("req" if change == "append" else None)
    assert len(kvcr.submit_hint_calls) == 1


def test_strict_cached_validation_avoids_hash_encoding(monkeypatch):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = _hint_context()
    _set_chain(ctx)
    tier.on_new_request(ctx)
    # Changing the codec cannot affect a previously validated stable prefix.
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    kvcr.query_status = QueryStatus.HIT
    key = make_offload_key((1).to_bytes(32, "big"), 0)
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert kvcr.query_calls[-1][1] == "req"
    assert len(kvcr.submit_hint_calls) == 1


def test_scheduler_reuses_only_unchanged_trusted_snapshot():
    from vllm import SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestHashChain,
        RequestOffloadState,
        SchedulerOffloadConfig,
    )
    from vllm.v1.kv_offload.base import RequestOffloadingContext
    from vllm.v1.request import Request

    ctx = _hint_context()
    request = Request("req", [1], SamplingParams(max_tokens=1), None)
    request.block_hashes.extend([bytes(32), (1).to_bytes(32, "big")])
    state = RequestOffloadState(
        SchedulerOffloadConfig((), 1, 16, 1, False, False, alignment_tokens=64),
        request,
        ctx,
        RequestOffloadingContext(),
    )
    state.update_offload_keys()
    first = ctx.get_state(RequestHashChain)
    state.update_offload_keys()
    assert ctx.get_state(RequestHashChain) is first
    request.block_hashes[0] = (2).to_bytes(32, "big")
    state.update_offload_keys()
    changed = ctx.get_state(RequestHashChain)
    assert changed is not first
    assert changed.block_hashes[0] == (2).to_bytes(32, "big")
    assert first.block_hashes[0] == bytes(32)


@pytest.mark.parametrize("successes", [(True, False), (False, True), (True, True)])
def test_strict_partition_completions_preserve_partial_success(monkeypatch, successes):
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = _hint_context()
    # Request registration may precede the first scheduler hash update.
    tier.on_new_request(ctx)
    assert kvcr.submit_hint_calls == []
    _set_chain(ctx)
    fine = make_offload_key((1).to_bytes(32, "big"), 0)
    local = make_offload_key((1).to_bytes(32, "big"), 1)
    tier.submit_load(
        TransferJob(
            job_id=7,
            keys=[fine, local],
            chunk_ids=np.array([0, 1]),
            is_promotion=True,
            req_context=ctx,
        )
    )
    first, second = kvcr.deliver_calls
    kvcr.completed.clear()
    kvcr.complete(first[0], {fine: successes[0]})
    assert list(tier.get_finished_jobs()) == []
    kvcr.complete(second[0], {local: successes[1]})
    expected = {key for key, success in zip((fine, local), successes) if success}
    assert list(tier.get_finished_jobs()) == [
        JobResult(7, all(successes), None if all(successes) else expected)
    ]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("malformed", [None, 7, "bad", b"short"])
def test_strict_malformed_snapshot_stays_local(monkeypatch, cached, malformed):
    from dataclasses import replace

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestHashChain,
    )

    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "1")
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr, opportunistic_query=False)
    ctx = _hint_context()
    _set_chain(ctx)
    if cached:
        tier.on_new_request(ctx)
    chain = ctx.get_state(RequestHashChain)
    ctx.set_state(replace(chain, block_hashes=(*chain.block_hashes, malformed)))
    kvcr.query_status = QueryStatus.HIT
    key = make_offload_key((1).to_bytes(32, "big"), 0)
    assert tier.lookup(key, ctx) is LookupResult.HIT
    assert kvcr.query_calls[-1][1] is None
    tier.submit_load(_job(7, ctx, key))
    assert kvcr.deliver_calls[-1][2] is None
    assert list(tier.get_finished_jobs()) == [JobResult(7, True)]
