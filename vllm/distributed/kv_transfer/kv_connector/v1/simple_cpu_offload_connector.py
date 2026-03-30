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
        # cpu_bytes_to_use is server-wide for compatibility;
        # cpu_bytes_to_use_per_rank overrides for per-rank capacity.
        world_size = vllm_config.parallel_config.world_size
        cpu_capacity_per_rank = cpu_capacity_bytes // world_size
        if "cpu_bytes_to_use_per_rank" in extra_config:
            explicit = int(extra_config["cpu_bytes_to_use_per_rank"])
            if explicit != cpu_capacity_per_rank:
                logger.warning(
                    "cpu_bytes_to_use_per_rank (%.2f GB) != "
                    "cpu_bytes_to_use/world_size (%.2f GB). Using per-rank value.",
                    explicit / (1024**3),
                    cpu_capacity_per_rank / (1024**3),
                )
            cpu_capacity_per_rank = explicit

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
            "SimpleCPUOffloadConnector: role=%s, "
            "per_rank=%.2f GB, world_size=%d, mode=%s",
            role.name,
            cpu_capacity_per_rank / (1024**3),
            world_size,
            "lazy" if lazy_offload else "eager",
        )

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_manager = SimpleCPUOffloadScheduler(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                lazy_offload=lazy_offload,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SimpleCPUOffloadWorker(
                vllm_config, kv_cache_config, cpu_capacity_per_rank
            )

    # --- Worker-side methods ---

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

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

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self.worker_handler is not None:
            assert isinstance(kv_connector_metadata, SimpleCPUOffloadMetadata)
            self.worker_handler.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass  # Launch loads ops in get_finished() after launching model execution

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

    def build_connector_worker_meta(self):
        if self.worker_handler is not None:
            return self.worker_handler.build_connector_worker_meta()
        return None

    # --- Scheduler-side methods ---

    # NOTE: New API only for SimpleCPUOffloadConnector.
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

    # NOTE: New API only for SimpleCPUOffloadConnector.
    def has_pending_transfers(self) -> bool:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.has_pending_stores()
        return False

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.take_events()
        return []
