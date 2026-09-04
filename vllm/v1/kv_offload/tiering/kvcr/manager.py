# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM secondary-tier adapter for KVCR."""

import ctypes
import mmap
import socket
import time
import uuid
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kvcr import (
    DURATION_METRIC,
    KVCR,
    ROUTER_HINT_CAPABILITIES,
    ROUTER_HINT_KEY,
    STATE_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
    KVCRBindings,
)
from kvcr.config import (
    FrameworkDramInput,
    G3Options,
    KeyAdapter,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    LocalDramOptions,
    RemoteFWDramOptions,
)
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.policy import (
    FIFOPolicy,
    G3FIFOPolicy,
    G3LRUPolicy,
    KVCachePolicy,
    LRUPolicy,
)
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    MemDescriptor,
    OpHandle,
    PinRequestId,
    PinResult,
    QueryStatus,
)
from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    ParentManager,
    SecondaryTierManager,
    TransferJob,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


_REQUIRED_ROUTER_CAPABILITIES = ROUTER_HINT_CAPABILITIES

logger = init_logger(__name__)

_BUILTIN_POLICIES: dict[str, type[KVCachePolicy]] = {
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "g3_fifo": G3FIFOPolicy,
    "g3_lru": G3LRUPolicy,
}


def _resolve_policy(name: str | None) -> KVCachePolicy | None:
    if name is None:
        return None
    policy_type = _BUILTIN_POLICIES.get(name)
    if policy_type is None:
        if "." not in name:
            raise ValueError(
                f"Unknown KVCR policy: {name!r}. "
                f"Supported: {list(_BUILTIN_POLICIES)}; external policies "
                "must use a fully qualified module.Class path."
            )
        policy_type = resolve_obj_by_qualname(name)
        if not isinstance(policy_type, type) or not issubclass(
            policy_type, KVCachePolicy
        ):
            raise TypeError(f"{name} is not a KVCachePolicy")
    return policy_type()


def _vllm_metric_name(name: str) -> str:
    return f"vllm:{name}"


def _kvcr_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        _vllm_metric_name(DURATION_METRIC): OffloadingHistogramMetadata(
            documentation="KVCR operation and stage duration in seconds.",
            labelnames=("scope", "result"),
        ),
        _vllm_metric_name(TRANSFER_BYTES_METRIC): OffloadingCounterMetadata(
            documentation="Bytes transferred by KVCR operations.",
            labelnames=("operation",),
        ),
        _vllm_metric_name(TRANSFER_BLOCKS_METRIC): OffloadingCounterMetadata(
            documentation="KV blocks transferred by successful KVCR operations.",
            labelnames=("operation",),
        ),
        _vllm_metric_name(STATE_METRIC): OffloadingGaugeMetadata(
            documentation="Current KVCR metadata and operation counts.",
            labelnames=("resource",),
        ),
    }


class _FrameworkPinAdapter:
    """Adapt KVCR framework pin callbacks to primary-tier store jobs."""

    def __init__(self, tier: "KVCRSecondaryTierManager") -> None:
        self._tier = tier
        self._next_request_id = 0
        self._next_pin_request_id = 0
        self._pending_pins: dict[PinRequestId, tuple[BlockKey, ...]] = {}
        self._completed_pins: list[tuple[PinRequestId, PinResult]] = []
        self._pin_jobs: dict[str, int] = {}
        self._pin_job_results: list[JobResult] = []

    def request_pin(self, keys: Collection[BlockKey]) -> PinRequestId:
        # The parent callback is valid only from serve_external_requests().
        request = PinRequestId(self._next_pin_request_id)
        self._next_pin_request_id += 1
        self._pending_pins[request] = tuple(keys)
        return request

    def process_pending(self, parent: ParentManager) -> None:
        pending = self._pending_pins
        self._pending_pins = {}
        self._completed_pins.extend(
            (request, self._resolve_pin(parent, keys))
            for request, keys in pending.items()
        )

    def poll_pin_results(self) -> list[tuple[PinRequestId, PinResult]]:
        completed = self._completed_pins
        self._completed_pins = []
        return completed

    def cancel_pin_request(self, request: PinRequestId) -> None:
        self._pending_pins.pop(request, None)

    def _resolve_pin(
        self, parent: ParentManager, keys: Collection[BlockKey]
    ) -> tuple[str, dict[BlockKey, MemDescriptor | None]] | None:
        offload_keys = tuple(OffloadKey(bytes(key)) for key in keys)
        request_id = f"kvcr-source:{self._next_request_id}"
        self._next_request_id += 1
        req_context = ReqContext(req_id=request_id)
        started = False
        job: TransferJob | None = None
        pin_handle: str | None = None
        try:
            parent.on_new_request(req_context)
            started = True
            hit_keys = tuple(
                key
                for key in offload_keys
                if parent.lookup(key, req_context) is LookupResult.HIT
            )
            if not hit_keys:
                return None

            job = parent.create_store_job(hit_keys, req_context)
            job_keys = tuple(job.keys)
            block_ids = tuple(int(block_id) for block_id in job.block_ids)
            if len(job_keys) != len(block_ids) or set(job_keys) != set(hit_keys):
                return None

            descriptors: dict[BlockKey, MemDescriptor | None] = {
                BlockKey(bytes(key)): None for key in offload_keys
            }
            descriptors.update(
                (
                    BlockKey(bytes(key)),
                    self._tier._make_descriptor(block_id),
                )
                for key, block_id in zip(job_keys, block_ids)
            )
            pin_handle = f"kvcr-job-{job.job_id}"
            self._pin_jobs[pin_handle] = job.job_id
            return pin_handle, descriptors
        except Exception:
            logger.warning("KVCR source acquisition failed", exc_info=True)
            return None
        finally:
            if started:
                try:
                    parent.on_request_finished(req_context)
                except Exception:
                    logger.warning("KVCR source request cleanup failed", exc_info=True)
            if job is not None and pin_handle is None:
                self._pin_job_results.append(
                    JobResult(job_id=job.job_id, success=False)
                )

    def release_pin(self, pin_handle: str) -> bool:
        job_id = self._pin_jobs.pop(pin_handle, None)
        if job_id is not None:
            self._pin_job_results.append(JobResult(job_id=job_id, success=True))
        return True

    def has_active_pins(self) -> bool:
        return bool(self._pin_jobs)

    def take_pin_job_results(self) -> list[JobResult]:
        results = self._pin_job_results
        self._pin_job_results = []
        return results


class _VllmKeyAdapter:
    """Adapt vLLM's local key format to KVCR."""

    def encode(self, framework_key: object) -> BlockKey:
        if not isinstance(framework_key, bytes):
            raise TypeError("vLLM offload keys must be bytes")
        return BlockKey(framework_key)

    def decode(self, key: BlockKey) -> int | bytes:
        block_hash = BlockHash(get_offload_block_hash(OffloadKey(key)))
        return maybe_convert_block_hash(block_hash)


# Job ID, remaining blocks, aggregate success, and successful load keys.
_JobState = tuple[int, int, bool, set[OffloadKey] | None]


class KVCRSecondaryTierManager(SecondaryTierManager):
    """Secondary tier wrapper around the KVCR KV P2P API."""

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        if not extra_config.get("enable_telemetry", False):
            return {}
        return _kvcr_metric_definitions()

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        router_capabilities: Iterable[str] | None = None,
        control_host: str = "0.0.0.0",
        control_ports: list[int] | None = None,
        control_advertise_host: str | None = None,
        eager_ctrl_connect: bool = True,
        opportunistic_query: bool = True,
        enable_telemetry: bool = False,
        operation_timeout_ms: int = 1000,
        metadata_retry_interval_ms: int = 100,
        secondary_g2_slots: int = 0,
        kvcr_service_socket_path: str | None = None,
        compatibility_digest: str | None = None,
        policy: str | None = None,
        g3: dict[str, Any] | None = None,
        local_dram_backend: str = "UCX",
        remote_fw_dram_backend: str = "UCX",
    ) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        selected_policy = _resolve_policy(policy)
        if (kvcr_service_socket_path is None) != (compatibility_digest is None):
            raise ValueError(
                "kvcr_service_socket_path and compatibility_digest must be "
                "configured together"
            )
        missing = _REQUIRED_ROUTER_CAPABILITIES - set(router_capabilities or ())
        if missing:
            raise ValueError(
                f"KVCR secondary tier requires router capabilities: {sorted(missing)}"
            )
        events_enabled = offloading_spec.kv_events_config.enable_kv_cache_events
        self_describing_events = (
            offloading_spec.kv_events_config.self_describing_kv_events
        )
        if (
            events_enabled
            and (
                secondary_g2_slots > 0
                or kvcr_service_socket_path is not None
                or g3 is not None
            )
            and not self_describing_events
        ):
            raise ValueError(
                "KVCR local G2/G3 inventory requires self_describing_kv_events"
            )
        self._key_adapter: KeyAdapter = _VllmKeyAdapter()
        advertise_host = control_advertise_host or socket.gethostname()
        dp_local_rank = offloading_spec.config.parallel.data_parallel_rank_local
        if dp_local_rank is None:
            if control_ports is None or len(control_ports) != 1:
                raise ValueError(
                    "control_ports must contain exactly one port when "
                    "data_parallel_rank_local is unset"
                )
            dp_local_rank = 0
        if control_ports is None or not 0 <= dp_local_rank < len(control_ports):
            raise ValueError(
                "control_ports must contain a port for "
                f"local data-parallel rank {dp_local_rank}"
            )
        if any(not 1 <= port <= 65535 for port in control_ports):
            raise ValueError("control_ports must be between 1 and 65535")
        control = ZmqPeerControlChannel(
            control_host,
            control_ports[dp_local_rank],
            advertise_host,
        )
        # Give colocated workers distinct transport listen ports.
        with socket.socket() as _s:
            _s.bind(("", 0))
            _nixl_listen_port = _s.getsockname()[1]
        self._primary_base_addr = ctypes.addressof(
            ctypes.c_char.from_buffer(primary_kv_view)
        )
        if primary_kv_view.strides is None:
            raise ValueError("primary KV memoryview must expose strides")
        self._primary_row_stride = int(primary_kv_view.strides[0])
        if secondary_g2_slots < 0:
            raise ValueError("secondary_g2_slots must be non-negative")
        if (
            secondary_g2_slots or kvcr_service_socket_path is not None
        ) and not events_enabled:
            logger.warning(
                "KVCR local DRAM is enabled, but KV cache events are disabled; "
                "local DRAM inventory will not be published"
            )
        local_mapping: mmap.mmap | None = None
        local_dram: LocalDramOptions | None = None
        if secondary_g2_slots:
            if kvcr_service_socket_path is not None:
                logger.warning(
                    "secondary_g2_slots is ignored when "
                    "kvcr_service_socket_path is configured"
                )
            else:
                local_mapping = mmap.mmap(
                    -1, secondary_g2_slots * self._primary_row_stride
                )
                local_dram = LocalDramOptions(
                    address=ctypes.addressof(ctypes.c_char.from_buffer(local_mapping)),
                    length=len(local_mapping),
                    slot_count=secondary_g2_slots,
                    backend=local_dram_backend,
                )
        guard_config = (
            KVCRGuardConfig(
                kvcr_service_socket_path=kvcr_service_socket_path,
                guard_index=dp_local_rank,
                row_stride=self._primary_row_stride,
                compatibility_digest=compatibility_digest,
            )
            if kvcr_service_socket_path is not None and compatibility_digest is not None
            else None
        )
        g3_config = None
        if g3 is not None:
            g3_values = dict(g3)
            g3_values["paths"] = tuple(Path(path) for path in g3_values["paths"])
            g3_config = G3Options(**g3_values)
        nixl_agent_name = f"KVCR-{uuid.uuid4()}"
        self._framework_pin_adapter = _FrameworkPinAdapter(self)
        self._inventory_events: list[OffloadingEvent] = []

        try:
            self._kvcr = KVCR(
                KVCRConfig(
                    enable_telemetry=enable_telemetry,
                    operation_timeout_ms=operation_timeout_ms,
                    nixl_agent_name=nixl_agent_name,
                    nixl_listen_port=_nixl_listen_port,
                ),
                KVCRBindings(
                    request_pin=self._framework_pin_adapter.request_pin,
                    poll_pin_results=self._framework_pin_adapter.poll_pin_results,
                    release_pin=self._framework_pin_adapter.release_pin,
                    cancel_pin_request=(self._framework_pin_adapter.cancel_pin_request),
                    framework_control=control,
                    key_adapter=self._key_adapter,
                    inventory_sink=(self._record_inventory if events_enabled else None),
                    stats_factory=(
                        OffloadingConnectorStats if enable_telemetry else None
                    ),
                    policy=selected_policy,
                ),
                KVCRBackendConfigs(
                    framework_dram=FrameworkDramInput(
                        self._primary_base_addr, primary_kv_view.nbytes
                    ),
                    local_dram=local_dram,
                    g3=g3_config,
                    remote_fw_dram=RemoteFWDramOptions(
                        eager_ctrl_connect=eager_ctrl_connect,
                        opportunistic_query=opportunistic_query,
                        metadata_retry_interval_ms=metadata_retry_interval_ms,
                        backend=remote_fw_dram_backend,
                    ),
                ),
                guard_config,
            )
        except BaseException:
            control.close()
            if local_mapping is not None:
                local_mapping.close()
            raise
        self._local_dram_mmap = local_mapping
        self._finished_jobs: list[JobResult] = []
        self._jobs_by_op: dict[OpHandle, _JobState] = {}

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        block_key = self._key_adapter.encode(key)
        status, _ = self._kvcr.query((block_key,), req_context.req_id)[0]
        if status is QueryStatus.FETCHING:
            return LookupResult.RETRY
        if status in (QueryStatus.HIT, QueryStatus.FETCHABLE):
            return LookupResult.HIT
        return LookupResult.MISS

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        blocks = {
            self._key_adapter.encode(key): self._make_descriptor(int(block_id))
            for key, block_id in zip(
                job_metadata.keys, job_metadata.block_ids, strict=True
            )
        }
        if not blocks:
            self._finished_jobs.append(
                JobResult(job_id=job_metadata.job_id, success=True)
            )
            return

        op_handle = self._kvcr.deliver(
            blocks, request_id=job_metadata.req_context.req_id
        )
        self._jobs_by_op[op_handle] = (
            job_metadata.job_id,
            len(blocks),
            True,
            set(),
        )

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        blocks = {
            self._key_adapter.encode(key): self._make_descriptor(int(block_id))
            for key, block_id in zip(
                job_metadata.keys, job_metadata.block_ids, strict=True
            )
        }
        if not blocks:
            self._finished_jobs.append(
                JobResult(job_id=job_metadata.job_id, success=True)
            )
            return
        op_handle = self._kvcr.deposit(blocks)
        self._jobs_by_op[op_handle] = (
            job_metadata.job_id,
            len(blocks),
            True,
            None,
        )

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        results = self._finished_jobs
        self._finished_jobs = []
        results.extend(self._poll_finished_jobs())
        return results

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = cast(OffloadingConnectorStats | None, self._kvcr.get_stats())
        if stats is not None:
            # Add the framework namespace only at the vLLM boundary.
            for metrics in stats.data.values():
                for name in tuple(metrics):
                    metrics[_vllm_metric_name(name)] = metrics.pop(name)
        return stats

    @override
    def serve_external_requests(self, parent: ParentManager) -> None:
        self._framework_pin_adapter.process_pending(parent)

    def _poll_finished_jobs(self) -> list[JobResult]:
        results: list[JobResult] = []
        for op_handle, entries in self._kvcr.poll_completed():
            job_state = self._jobs_by_op.get(op_handle)
            if job_state is None:
                continue
            job_id, remaining, success, successful_keys = job_state
            remaining -= len(entries)
            success = success and all(entry.success for entry in entries.values())
            if successful_keys is not None:
                successful_keys.update(
                    OffloadKey(bytes(key))
                    for key, entry in entries.items()
                    if entry.success
                )
            if remaining > 0:
                self._jobs_by_op[op_handle] = (
                    job_id,
                    remaining,
                    success,
                    successful_keys,
                )
                continue
            self._jobs_by_op.pop(op_handle, None)
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    successful_keys=successful_keys if not success else None,
                )
            )
        results.extend(self._framework_pin_adapter.take_pin_job_results())
        return results

    @override
    def has_pending_work(self) -> bool:
        # Keep scheduler polling: control messages may arrive without a load job.
        return True

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        events = self._inventory_events
        self._inventory_events = []
        return events

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        params = getattr(req_context, "kv_transfer_params", None)
        if isinstance(params, Mapping):
            hint = params.get(ROUTER_HINT_KEY)
            if hint is not None:
                self._kvcr.submit_hint(
                    request_id=req_context.req_id,
                    hints=hint,
                )
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._kvcr.discard_hint(req_context.req_id)

    @override
    def drain_jobs(self) -> None:
        while self._jobs_by_op or self._framework_pin_adapter.has_active_pins():
            drained = self._poll_finished_jobs()
            if drained:
                self._finished_jobs.extend(drained)
            else:
                time.sleep(0.001)

    @override
    def shutdown(self) -> None:
        self.drain_jobs()
        # TODO: Keep registered buffers alive if KVCR.close() fails.
        self._kvcr.close()
        if self._local_dram_mmap is not None:
            self._local_dram_mmap.close()
            self._local_dram_mmap = None

    def _record_inventory(self, event: InventoryEvent) -> None:
        if event.tier is CacheTier.LOCAL_G2:
            medium = Medium.CPU
        elif event.tier is CacheTier.G3:
            medium = Medium.STORAGE
        else:
            raise ValueError(f"unsupported KVCR inventory tier: {event.tier}")
        self._inventory_events.append(
            OffloadingEvent(
                keys=[OffloadKey(bytes(key)) for key in event.keys],
                medium=medium,
                removed=event.removed,
                ownership="kvcr",
                removal_expected=True,
            )
        )

    def _make_descriptor(self, block_id: int) -> MemDescriptor:
        return MemDescriptor(
            end_point_name=self._kvcr.config.nixl_agent_name,
            mem_type="DRAM",
            addr=self._primary_base_addr + block_id * self._primary_row_stride,
            size=self._primary_row_stride,
            device_Id=0,
            info="",
        )
