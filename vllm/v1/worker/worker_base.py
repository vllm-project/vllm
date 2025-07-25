# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn as nn

from vllm.config import IntermediateLoggingConfig, VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.intermediates_logging import register_intermediate_hooks
from vllm.worker.worker_base import WorkerBase as WorkerBaseV0

logger = init_logger(__name__)


class WorkerBase(WorkerBaseV0):
    """
    Abstract class for v1 worker, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in v0 WorkerBase
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """
        Initialize common worker components.
        
        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver 
            responsibilities
        """
        # Configuration storage
        super().__init__(vllm_config=vllm_config)

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: Optional[torch.device] = None
        self.model_runner: Optional[nn.Module] = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def register_intermediate_hooks(
            self,
            config: Optional[IntermediateLoggingConfig] = None,
            **kwargs) -> None:
        """Register hooks for intermediate tensor logging.
        
        This method is called via collective_rpc from the engine core.
        It registers hooks on the model to dump intermediate tensors during execution.
        
        Args:
            config: Configuration for intermediate logging. If provided, this takes precedence over kwargs.
        """
        if self.model_runner is None or not hasattr(
                self.model_runner, "model") or self.model_runner.model is None:
            logger.error(
                "Could not register intermediate hooks: model_runner.model is not accessible"
            )
            return
        model = self.model_runner.model
        try:
            # Register hooks
            register_intermediate_hooks(model, config, **kwargs)
            # Store the logger instance for potential later hook removal
        except Exception:
            logger.info("Successfully registered intermediate hooks")
            logger.error("Error registering intermediate hooks", exc_info=True)
