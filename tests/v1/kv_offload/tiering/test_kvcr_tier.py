# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection, Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("kvcr")

from kvcr import KVCRBindings, ROUTER_HINT_KEY
from kvcr.config import G3Options, KVCRBackendConfigs, KVCRConfig, KVCRGuardConfig
from kvcr.policy import FIFOPolicy, G3FIFOPolicy, G3LRUPolicy, LRUPolicy
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    MemDescriptor,
    OpEntryResult,
    OpEntryStatus,
    OpHandle,
    QueryStatus,
)

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
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
        self.stats: OffloadingConnectorStats | None = None
        self.submit_hint_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.discard_hint_calls: list[str] = []
        self.deliver_calls: list[
            tuple[OpHandle, dict[BlockKey, MemDescriptor], str | None]
        ] = []
        self.deposit_calls: list[tuple[OpHandle, dict[BlockKey, MemDescriptor]]] = []
        self.completed: list[tuple[OpHandle, dict[BlockKey, OpEntryResult]]] = []
        self._next_op_handle = 1

    def submit_hint(self, *args: Any, **kwargs: Any) -> None:
        self.submit_hint_calls.append((args, kwargs))

    def discard_hint(self, request_id: str) -> None:
        self.discard_hint_calls.append(request_id)

    def query(
        self,
        keys: Collection[BlockKey],
        request_id: str | None = None,
    ) -> list[tuple[QueryStatus, CacheTier | None]]:
        tier = {
            QueryStatus.HIT: CacheTier.LOCAL_G2,
            QueryStatus.FETCHING: CacheTier.LOCAL_G2,
            QueryStatus.FETCHABLE: CacheTier.REMOTE_G2,
            QueryStatus.MISS: None,
        }[self.query_status]
        return [(self.query_status, tier) for _ in keys]

    def deliver(
        self,
        blocks: Mapping[BlockKey, MemDescriptor],
        request_id: str | None = None,
    ) -> OpHandle:
        op_handle = self._next_op_handle
        self._next_op_handle += 1
        self.deliver_calls.append((op_handle, dict(blocks), request_id))
        self.complete(op_handle, {key: True for key in blocks})
        return op_handle

    def deposit(
        self,
        blocks: Mapping[BlockKey, MemDescriptor],
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
    block_id: int = 0,
) -> TransferJob:
    return TransferJob(
        job_id=job_id,
        keys=[key if key is not None else OffloadKey(b"k0")],
        block_ids=np.array([block_id], dtype=np.int64),
        is_promotion=True,
        req_context=req_context,
    )


def _make_tier(
    monkeypatch,
    kvcr: RecordingKVCR,
    *,
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
        return kvcr

    monkeypatch.setattr(kvcr_manager, "KVCR", make_kvcr)
    monkeypatch.setattr(kvcr_manager, "ZmqPeerControlChannel", make_control)
    return KVCRSecondaryTierManager(
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
        primary_kv_view=memoryview(np.zeros((4, 16), dtype=np.int8)),
        tier_type="kvcr",
        router_capabilities=["router_hint"],
        control_host="127.0.0.1",
        control_ports=control_ports if control_ports is not None else [7777],
        control_advertise_host="127.0.0.1",
        enable_telemetry=enable_telemetry,
        secondary_g2_slots=secondary_g2_slots,
        kvcr_service_socket_path=kvcr_service_socket_path,
        compatibility_digest=compatibility_digest,
        policy=policy,
        g3=g3,
        local_dram_backend="UCX",
        remote_fw_dram_backend="UCX",
    )


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
        row_stride=tier._primary_row_stride,
        compatibility_digest="Opaque-Digest",
    )
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


def test_kvcr_tier_maps_router_hint_to_load(monkeypatch):
    """Exercise the complete vLLM router-hint-to-KVCR load translation."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    router_hint = {
        "source_control_endpoint": "tcp://source:1234",
        "block_hashes": [123],
        "framework_hint": {"opaque": True},
    }
    ctx = ReqContext(
        req_id="req",
        kv_transfer_params={ROUTER_HINT_KEY: router_hint, "unrelated": object()},
    )
    key = make_offload_key((123).to_bytes(8, "big"), 0)
    same_hash_other_group = make_offload_key((123).to_bytes(8, "big"), 7)
    other_key = make_offload_key((124).to_bytes(8, "big"), 0)

    tier.on_new_request(ctx)

    assert kvcr.submit_hint_calls == [((), {"request_id": "req", "hints": router_hint})]

    bindings = kvcr.constructor_bindings
    assert bindings is not None
    assert bindings.key_adapter is not None
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    decode = bindings.key_adapter.decode
    hash_123 = (123).to_bytes(8, "big")
    assert decode(BlockKey(bytes(key))) == hash_123
    assert decode(BlockKey(bytes(same_hash_other_group))) == hash_123
    assert decode(BlockKey(bytes(other_key))) == (124).to_bytes(8, "big")

    tier.submit_load(_job(7, ctx, key=key, block_id=2))

    # Here we verify that submitting a load for a hinted key asks KVCR to
    # deliver that key into the expected primary memory slot and completes the
    # vLLM transfer job successfully.
    assert len(kvcr.submit_hint_calls) == 1
    _, blocks, request_id = kvcr.deliver_calls[0]
    assert request_id == "req"
    assert list(blocks) == [key]
    assert blocks[key].end_point_name == kvcr.nixl_agent_name
    assert blocks[key].addr == tier._primary_base_addr + 2 * 16
    assert blocks[key].size == 16
    assert list(tier.get_finished_jobs()) == [JobResult(7, True)]

    # Here we verify that request cleanup discards the request-scoped hint in
    # KVCR.
    tier.on_request_finished(ctx)
    assert kvcr.discard_hint_calls == ["req"]


def test_kvcr_tier_allows_request_without_router_hint(monkeypatch):
    """Keep router hints optional for requests from non-hint-aware routers."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)

    tier.on_new_request(ReqContext(req_id="req"))

    assert kvcr.submit_hint_calls == []


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


def test_kvcr_tier_serves_primary_pin_request(monkeypatch):
    """Hold primary-tier hits until KVCR releases the corresponding pin."""
    kvcr = RecordingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    keys = (BlockKey(b"k0"), BlockKey(b"k1"), BlockKey(b"k2"))
    hit_keys = (keys[0], keys[2])
    block_ids = {keys[0]: 1, keys[2]: 5}
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
                keys=requested_keys,
                block_ids=np.array([block_ids[key] for key in requested_keys]),
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
    assert result is not None
    pin_handle, descriptors = result
    assert descriptors[keys[1]] is None
    descriptor = descriptors[keys[2]]
    assert descriptor is not None
    assert descriptor.addr == (tier._primary_base_addr + 5 * tier._primary_row_stride)
    assert lifecycle == ["new", "finished"]

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
    assert local_dram.length == 2 * tier._primary_row_stride
    assert local_dram.slot_count == 2

    key = OffloadKey(b"k0")
    tier.submit_store(_job(11, ReqContext(req_id="req"), key=key, block_id=2))

    _, blocks = kvcr.deposit_calls[0]
    assert blocks[key].addr == tier._primary_base_addr + 2 * 16
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


def test_kvcr_tier_waits_for_all_completions_and_drains(monkeypatch):
    """Wait for every block result and preserve partial success while draining."""

    class DrainingKVCR(RecordingKVCR):
        def __init__(self):
            super().__init__()
            self.polls: list[list[tuple[OpHandle, dict[BlockKey, OpEntryResult]]]] = []

        def deliver(
            self,
            blocks: Mapping[BlockKey, MemDescriptor],
            request_id: str | None = None,
        ) -> OpHandle:
            op_handle = self._next_op_handle
            self._next_op_handle += 1
            self.deliver_calls.append((op_handle, dict(blocks), request_id))
            keys = list(blocks)
            self.polls = [
                [(op_handle, _op_entries({keys[0]: True}))],
                [(op_handle, _op_entries({keys[1]: False}))],
            ]
            return op_handle

        def poll_completed(
            self,
        ) -> Iterable[tuple[OpHandle, dict[BlockKey, OpEntryResult]]]:
            return self.polls.pop(0) if self.polls else []

    kvcr = DrainingKVCR()
    tier = _make_tier(monkeypatch, kvcr)
    keys = [OffloadKey(b"k0"), OffloadKey(b"k1")]
    tier.submit_load(
        TransferJob(
            job_id=13,
            keys=keys,
            block_ids=np.array([0, 1], dtype=np.int64),
            is_promotion=True,
            req_context=ReqContext(req_id="req"),
        )
    )

    assert list(tier.get_finished_jobs()) == []
    tier.drain_jobs()

    [result] = tier.get_finished_jobs()
    assert result.job_id == 13
    assert not result.success
    assert result.successful_keys == {keys[0]}
