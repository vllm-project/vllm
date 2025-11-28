# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MPS (Metal Performance Shaders) worker for Apple Silicon GPUs.

This worker is designed for single-device inference on Apple Silicon Macs,
leveraging PyTorch's MPS backend for GPU acceleration.
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.mps_model_runner import MPSModelRunner

logger = init_logger(__name__)


class MPSWorker(Worker):
    """Worker for MPS (Metal Performance Shaders) backend on Apple Silicon."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # MPS doesn't support custom all-reduce
        self.parallel_config.disable_custom_all_reduce = True

    def init_device(self):
        """Initialize MPS device and model runner."""
        # Check if MPS is available
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend is not available. "
                "Make sure you are running on an Apple Silicon Mac with macOS 12.3+ "
                "and PyTorch with MPS support."
            )

        logger.info("Initializing MPS worker on Apple Silicon GPU")

        # Initialize the distributed environment (though MPS doesn't support multi-GPU)
        if self.parallel_config.world_size > 1:
            logger.warning(
                "MPS backend does not support distributed execution. "
                "Running on single device only."
            )

        # Note: MPS backend doesn't require special initialization like CUDA
        # Just use the device directly
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,  # Will be "gloo"
        )

        # Set random seed
        set_random_seed(self.model_config.seed)

        # Construct the model runner with MPS device
        # For now, use CPUModelRunner as it should work with MPS device
        # since PyTorch MPS operations are similar to CPU operations
        mps_device = torch.device("mps")
        logger.info(f"Using device: {mps_device}")

        # Use MPS model runner with MPS device
        self.model_runner: MPSModelRunner = MPSModelRunner(
            self.vllm_config, mps_device
        )
        logger.info("Successfully initialized MPS model runner")

    def sleep(self, level: int = 1) -> None:
        """Sleep mode is not supported on MPS."""
        logger.warning("Sleep mode is not supported on MPS, ignoring.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up mode is not supported on MPS."""
        logger.warning("Wake up mode is not supported on MPS, ignoring.")

    def determine_available_memory(self) -> int:
        """
        Determine available memory on MPS device.

        Apple Silicon uses unified memory shared between CPU and GPU.
        We use the platform's reported total memory (which accounts for
        system reservation).
        """
        total_memory = current_platform.get_device_total_memory()
        logger.info(
            f"MPS unified memory available: {total_memory / (1024**3):.2f} GB"
        )
        return total_memory

    def compile_or_warm_up_model(self) -> None:
        """Warm up the model."""
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        logger.info("Warming up model on MPS device...")
        try:
            self.model_runner.warming_up_model()
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"Model warmup encountered an error: {e}")
            logger.info("Proceeding anyway, some operations may be slower initially")

    def get_current_memory_usage(self) -> float:
        """Get current memory usage on MPS device."""
        try:
            if hasattr(torch.mps, 'current_allocated_memory'):
                return float(torch.mps.current_allocated_memory())
            else:
                # MPS memory tracking may not be available in all PyTorch versions
                logger.debug("MPS memory tracking not available")
                return 0.0
        except Exception as e:
            logger.debug(f"Failed to get MPS memory usage: {e}")
            return 0.0
