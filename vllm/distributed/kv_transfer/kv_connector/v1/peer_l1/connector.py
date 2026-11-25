# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort prompt-prefix sharing between data-parallel GPU caches."""

from collections.abc import Callable
from typing import Any

import torch

from vllm.config import KVTransferConfig, VllmConfig
from vllm.config.kv_transfer import KVCacheAccess
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import PeerHandshake, PeerMetadata
from .registry import Domain
from .scheduler import PeerScheduler
from .stats import PeerL1PromMetrics, PeerL1Stats
from .worker import PeerWorker


def supported_groups(cache: KVCacheConfig) -> tuple[int, ...]:
    """Validate device attention groups; omit non-prefix-cacheable state."""
    groups = []
    for index, group in enumerate(cache.kv_cache_groups):
        if not group.prefix_cacheable:
            continue
        if (
            group.manager_cls not in (FullAttentionManager, SlidingWindowManager)
            or not group.enable_kv_transfer
            or group.host_resident
        ):
            raise ValueError(
                f"Peer L1 cannot restore prefix-cache group {index} "
                f"({group.manager_cls.__name__}); "
                "stable device attention blocks are required"
            )
        groups.append(index)
    if not groups:
        raise ValueError("Peer L1 requires a prefix-cacheable attention group")
    return tuple(groups)


class PeerL1Connector(KVConnectorBase_V1, SupportsHMA):
    """Dispatch peer-cache hooks to the rank's scheduler or GPU worker process.

    Scheduler hooks manage source leases; worker hooks copy into local KV storage.
    """

    _cache_hit_source = CacheHitSource.P2P

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> PeerL1Stats:
        return PeerL1Stats(data) if data is not None else PeerL1Stats()

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> PeerL1PromMetrics:
        return PeerL1PromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        stats = (
            self._scheduler.get_stats()
            if self._scheduler
            else self._worker.get_stats()
            if self._worker
            else None
        )
        return stats if stats is not None and not stats.is_empty() else None

    @classmethod
    def kv_cache_access(cls, kv_transfer_config: KVTransferConfig) -> KVCacheAccess:
        return KVCacheAccess.LEGACY_IPC | KVCacheAccess.REGISTERS

    @classmethod
    def only_restores_canonical_kv(cls, kv_transfer_config: KVTransferConfig) -> bool:
        return True

    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config: KVTransferConfig) -> bool:
        return False

    @classmethod
    def supports_lookahead_block_hashes(
        cls, kv_transfer_config: KVTransferConfig
    ) -> bool:
        return True

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if not vllm_config.model_config.is_moe:
            raise ValueError(
                "Peer L1 requires a MoE model: vLLM removes the worker DP "
                "group for independent dense-model ranks"
            )
        domain = Domain.from_options(self._kv_transfer_config.kv_connector_extra_config)
        parallel = vllm_config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "Peer L1 requires TP=PP=DCP=PCP=1; use DP for cache sharing"
            )
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Peer L1 requires prefix caching")
        if self._kv_transfer_config.kv_role != "kv_both":
            raise ValueError("Peer L1 requires kv_role=kv_both")
        if (
            domain.instance_id
            and self._kv_transfer_config.kv_load_failure_policy != "recompute"
        ):
            raise ValueError(
                "Cross-instance peer L1 requires kv_load_failure_policy=recompute"
            )
        supported = supported_groups(kv_cache_config)
        self._supported = supported
        self._scheduler: PeerScheduler | None = None
        self._worker: PeerWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = PeerScheduler(
                vllm_config, kv_cache_config, supported, domain=domain
            )
        else:
            self._worker = PeerWorker(vllm_config, kv_cache_config, domain=domain)

    def bind_kv_cache_manager(self, kv_cache_manager: KVCacheManager) -> None:
        super().bind_kv_cache_manager(kv_cache_manager)
        assert self._scheduler is not None
        self._scheduler.bind(kv_cache_manager)

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        assert self._scheduler is not None
        handshake = metadata[0]
        assert isinstance(handshake, PeerHandshake)
        self._scheduler.handshake(handshake)

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        assert self._worker is not None
        return self._worker.handshake()

    def set_engine_wakeup(self, callback: Callable[[], None]) -> None:
        assert self._scheduler is not None
        self._scheduler.set_engine_wakeup(callback)

    def has_pending_push_work(self) -> bool:
        """Report whether owner or publication work needs an idle engine step."""
        return bool(self._scheduler and self._scheduler.pending())

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self._scheduler is not None
        return self._scheduler.match(request, num_computed_tokens)

    def get_loaded_kv_cache_group_ids(self, request: Request) -> tuple[int, ...]:
        return self._supported

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        assert self._scheduler is not None
        self._scheduler.allocated(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build()

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self._scheduler is not None
        self._scheduler.update(connector_output)

    def request_finished(self, request: Request, block_ids: list[int]):
        assert self._scheduler is not None
        self._scheduler.finished(request.request_id)
        return False, None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ):
        return self.request_finished(request, [])

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self._worker is not None
        self._worker.register(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        assert self._worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, PeerMetadata)
        self._worker.load(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        """Record completion after all step writers on the worker's current stream.

        The caller must order speculative draft writes before this hook.
        """
        assert self._worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, PeerMetadata)
        self._worker.save(meta)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self._worker is not None
        return self._worker.report()

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self._worker is not None
        return self._worker.transfer_results(finished_req_ids)

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
        if self._worker is not None:
            self._worker.close()
