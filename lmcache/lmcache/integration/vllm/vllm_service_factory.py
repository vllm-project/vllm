# SPDX-License-Identifier: Apache-2.0
"""
VllmServiceFactory: Creates LMCache service components for vLLM integration.

This factory encapsulates all vLLM-specific component creation logic,
keeping the LMCacheManager agnostic to the serving engine.
"""

# Standard
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig

    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer
    from lmcache.v1.manager import LMCacheManager

# First Party
from lmcache.integration.base_service_factory import BaseServiceFactory
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.internal_api_server.api_server import InternalAPIServer
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher

logger = init_logger(__name__)


class VllmServiceFactory(BaseServiceFactory):
    """Creates LMCache service components for vLLM integration.

    Handles role-based component creation:
    - scheduler: lookup client, PrometheusLogger (no engine if bypass disabled)
    - worker: engine, lookup server, offload server
    - DP rank 0: API server, plugin launcher
    - all roles: health monitor (created during post_init)
    """

    def __init__(
        self,
        lmcache_config: LMCacheEngineConfig,
        vllm_config: "VllmConfig",
        role: str,
    ):
        self.lmcache_config = lmcache_config
        self.vllm_config = vllm_config
        self.role = role
        self.metadata: Optional[LMCacheMetadata] = None
        self.lmcache_engine: Optional[LMCacheEngine] = None

    def get_engine_instance_id(self) -> str:
        # First Party
        from lmcache.integration.vllm.utils import ENGINE_NAME

        return ENGINE_NAME

    def get_or_create_metadata(self) -> Optional[LMCacheMetadata]:
        if self.metadata is not None:
            return self.metadata

        # First Party
        from lmcache.integration.vllm.utils import (
            calculate_draft_layers,
            calculate_local_rank_and_world_size,
            mla_enabled,
            validate_mla_config,
        )

        try:
            # Third Party
            from vllm.utils.torch_utils import get_kv_cache_torch_dtype
        except ImportError:
            # Third Party
            from vllm.utils import get_kv_cache_torch_dtype

        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        cache_config = self.vllm_config.cache_config

        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        use_mla = mla_enabled(model_config)
        validate_mla_config(self.lmcache_config, use_mla)

        num_layer = model_config.get_num_layers(parallel_config)
        num_draft_layers = calculate_draft_layers(self.vllm_config)
        num_layer += num_draft_layers
        chunk_size = self.lmcache_config.chunk_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        kv_shape = (
            num_layer,
            1 if use_mla else 2,
            chunk_size,
            num_kv_head,
            head_size,
        )

        logger.info(
            "num_layer: %d, chunk_size: %d, num_kv_head (per gpu): %d, "
            "head_size: %d, hidden_dim (D) for KV (per gpu): %d, "
            "use mla: %s, kv shape: %s, num_draft_layers: %d",
            num_layer,
            chunk_size,
            num_kv_head,
            head_size,
            num_kv_head * head_size,
            use_mla,
            kv_shape,
            num_draft_layers,
        )

        # Extract engine_id and kv_connector_extra_config from vllm_config
        engine_id = None
        kv_connector_extra_config = None
        if hasattr(self.vllm_config, "kv_transfer_config"):
            kv_transfer_config = self.vllm_config.kv_transfer_config
            if kv_transfer_config is not None:
                engine_id = getattr(kv_transfer_config, "engine_id", None)
                kv_connector_extra_config = getattr(
                    kv_transfer_config, "kv_connector_extra_config", None
                )

        if self.role == "scheduler":
            # Avoid GPU probing for scheduler-only metadata path;
            # scheduler may run on CPU-only control-plane nodes.
            local_worker_id = parallel_config.rank
            local_world_size = parallel_config.world_size
        else:
            local_worker_id, local_world_size = calculate_local_rank_and_world_size(
                self.vllm_config
            )
        self.metadata = LMCacheMetadata(
            model_name=model_config.model,
            world_size=parallel_config.world_size,
            local_world_size=local_world_size,
            worker_id=parallel_config.rank,
            local_worker_id=local_worker_id,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role=self.role,
            served_model_name=model_config.served_model_name,
            chunk_size=self.lmcache_config.chunk_size,
            engine_id=engine_id,
            kv_connector_extra_config=kv_connector_extra_config,
        )
        return self.metadata

    def get_or_create_lmcache_engine(self) -> Optional[LMCacheEngine]:
        self._ensure_metadata()
        assert self.metadata is not None

        # Scheduler without bypass lookup does not need an engine
        if (
            self.role == "scheduler"
            and not self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            # Create PrometheusLogger for scheduler without engine
            # First Party
            from lmcache.observability import PrometheusLogger

            PrometheusLogger.GetOrCreate(
                self.metadata,
                config=self.lmcache_config,
            )
            return None

        # First Party
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.utils import EngineType
        from lmcache.v1.gpu_connector import CreateGPUConnector

        if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
            self.lmcache_engine = curr_engine
            return curr_engine

        if self.role == "scheduler":
            tpg = SimpleNamespace()
            tpg.broadcast = lambda tensor, src: tensor
            tpg.broadcast_object = lambda obj, src: obj
            vllm_gpu_connector = None
        else:
            # Third Party
            from vllm.distributed.parallel_state import get_tp_group

            tpg = get_tp_group()
            # First Party
            from lmcache.integration.vllm.utils import vllm_layout_hints

            vllm_gpu_connector = CreateGPUConnector(
                self.lmcache_config,
                self.metadata,
                EngineType.VLLM,
                layout_hints=vllm_layout_hints(),
            )

        engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            self.lmcache_config,
            self.metadata,
            vllm_gpu_connector,
            tpg.broadcast,
            tpg.broadcast_object,
        )
        self.lmcache_engine = engine

        if (
            self.role == "scheduler"
            and self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            assert engine.save_only_first_rank or (
                self.lmcache_config.get_extra_config_value(
                    "remote_enable_mla_worker_id_as0", self.metadata.use_mla
                )
            ), (
                "enable_scheduler_bypass_lookup is only supported with "
                "save_only_first_rank or remote_enable_mla_worker_id_as0"
            )

        return engine

    def _ensure_metadata(self):
        if self.metadata is None:
            self.get_or_create_metadata()

    def _ensure_engine(self):
        if self.lmcache_engine is None:
            self.get_or_create_lmcache_engine()

    def maybe_create_prometheus_logger(self):
        # PrometheusLogger is created on-demand within other components
        # (e.g., engine creation for scheduler, health monitor setup).
        return None

    def maybe_create_lookup_client(self) -> Optional[LookupClientInterface]:
        # Only scheduler needs lookup client
        if self.role != "scheduler":
            return None

        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        self._ensure_metadata()
        assert self.metadata is not None
        return LookupClientFactory.create_lookup_client(
            self.lmcache_config,
            self.metadata,
            self.lmcache_engine,
        )

    def maybe_create_lookup_server(
        self,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        # Only worker needs lookup server
        if self.role != "worker":
            return None

        # First Party
        from lmcache.v1.lookup_client.factory import LookupClientFactory

        self._ensure_metadata()
        self._ensure_engine()
        assert self.metadata is not None
        assert self.lmcache_engine is not None
        return LookupClientFactory.create_lookup_server(
            self.lmcache_engine,
            self.metadata,
        )

    def maybe_create_offload_server(self) -> Optional[ZMQOffloadServer]:
        # Only worker needs offload server
        if self.role != "worker":
            return None

        # Third Party
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        self._ensure_engine()
        assert self.lmcache_engine is not None
        return ZMQOffloadServer(
            self.lmcache_engine,
            get_tensor_model_parallel_rank(),
        )

    def maybe_create_runtime_plugin_launcher(
        self,
    ) -> Optional[RuntimePluginLauncher]:
        # First Party
        from lmcache.integration.vllm.utils import is_dp_rank0

        # Only DP rank 0 needs runtime plugin launcher
        if not is_dp_rank0(self.vllm_config):
            return None

        worker_id = (
            -1
            if self.lmcache_engine is None or self.metadata is None
            else self.metadata.worker_id
        )
        return RuntimePluginLauncher(
            self.lmcache_config,
            self.role,
            self.vllm_config.parallel_config.tensor_parallel_size,
            worker_id,
        )

    def maybe_create_internal_api_server(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional[InternalAPIServer]:
        # First Party
        from lmcache.integration.vllm.utils import is_dp_rank0

        # Only DP rank 0 needs internal API server
        if not is_dp_rank0(self.vllm_config):
            return None

        return InternalAPIServer(lmcache_manager)

    def maybe_create_health_monitor(
        self, lmcache_manager: "LMCacheManager"
    ) -> Optional[HealthMonitor]:
        return self._create_health_monitor(
            lmcache_manager, self.lmcache_config, self.lmcache_engine
        )
