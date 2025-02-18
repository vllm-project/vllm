# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheSpec
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

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return
