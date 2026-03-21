# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffloadConnector: minimal CPU KV cache offloading."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import (
    SimpleCPUOffloadScheduler,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
)
from vllm.v1.simple_kv_offload.worker import (
    SimpleCPUOffloadWorker,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Default CPU capacity: 8 GB
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)


class SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """CPU KV cache offloading with custom kernel transfers and BlockPool LRU."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        cpu_capacity_bytes = int(
            extra_config.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES)
        )
        lazy_offload = bool(extra_config.get("lazy_offload", False))

        self.scheduler_manager: SimpleCPUOffloadScheduler | None = None
        self.worker_handler: SimpleCPUOffloadWorker | None = None

        if not enable_prefix_caching:
            logger.warning(
                "Detected prefix caching disabled, disabling CPU offload "
                "since it requires prefix caching."
            )
            return

        logger.info(
            "CPUOffloadConnector: Initializing with role=%s, cpu_capacity=%.2f GB, "
            "mode=%s",
            role.name,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = SimpleCPUOffloadScheduler(
                vllm_config,
                kv_cache_config,
                cpu_capacity_bytes,
                lazy_offload=lazy_offload,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SimpleCPUOffloadWorker(
                vllm_config, kv_cache_config, cpu_capacity_bytes
            )

    # --- Worker-side methods ---

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # When prefix caching is disabled, the connector is inactive.
        if self.worker_handler is None and self.scheduler_manager is None:
            return False
        if self._kv_cache_config is None:
            return False
        from vllm.v1.kv_cache_interface import MambaSpec

        if any(
            isinstance(group.kv_cache_spec, MambaSpec)
            for group in self._kv_cache_config.kv_cache_groups
        ):
            return False
        return len(self._kv_cache_config.kv_cache_groups) == 1

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: "type[Any]",
    ) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches({"ALL_LAYERS": kv_cache})

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.handle_preemptions(preempted_req_ids)

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # Always load asynchronously and deferred to get_finished()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass  # Always save asynchronously and deferred to get_finished()

    def wait_for_save(self) -> None:
        pass  # All stores are driven by get_finished() and no wait needed

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker_handler is not None:
            return self.worker_handler.get_finished(finished_req_ids)
        return None, None

    # --- Scheduler-side methods ---

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.build_connector_meta(scheduler_output)
        return SimpleCPUOffloadMetadata()

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(
                request, block_ids
            )
        return False, None

    def has_pending_transfers(self) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_pending_stores()
        return False

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []
