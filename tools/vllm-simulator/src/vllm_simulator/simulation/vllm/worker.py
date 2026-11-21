"""
Worker hook: overrides only device-lifecycle methods.

Model work goes through the native Worker into _ModelRunnerStub
(model_runner_stub.py).
"""

from contextlib import nullcontext

import torch

from vllm_simulator.hook import BaseHook
from vllm_simulator.simulation.manager import ConfigManager
from vllm_simulator.simulation.utils import profile_device_available_bytes
from vllm_simulator.simulation.vllm.model_runner_stub import (
    _ModelRunnerStub,
    build_kv_cache_spec,
)
from vllm_simulator.utils import get_logger

logger = get_logger()


class C_VLLMWorkerHook(BaseHook):
    """Hook Worker to run on CPU without model/CUDA dependencies."""

    HOOK_CLASS_NAME = "Worker"
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"

    @classmethod
    def hook(cls, target):
        def override_init_device(self):
            """Minimal init: distributed env (gloo) + stub model runner."""
            from vllm.v1.worker.gpu_worker import (
                init_worker_distributed_environment,
                init_workspace_manager,
                set_random_seed,
            )

            ConfigManager.init_from_vllm_config(self.vllm_config)

            # gloo: no NCCL/CUDA in simulation.
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                "gloo",
            )
            set_random_seed(self.vllm_config.model_config.seed)

            self.device = torch.device("cpu")
            # Before the meta build: MLA backends reserve workspaces at
            # construction.
            init_workspace_manager(self.device, 1)
            self.model_runner = _ModelRunnerStub(
                self.vllm_config, build_kv_cache_spec(self.vllm_config)
            )
            logger.info("[vLLM Hijack] Worker.init_device: stub initialized")

        def override_determine_available_memory(self):
            """Size the KV pool from the simulated device instead of profiling."""
            available_bytes = profile_device_available_bytes(
                ConfigManager.get_model_config(),
                ConfigManager.get_accelerator_info(),
                ConfigManager.get_scheduler_config(),
            )
            logger.info(
                "[vLLM Hijack] Worker.determine_available_memory: %d gibibytes. "
                "The available memory will be used for kv cache allocation.",
                available_bytes // (1 << 30),
            )
            return available_bytes

        def override_maybe_get_memory_pool_context(self, tag):
            """No CuMem pool: the mock platform is not cuda-alike, so the
            native method would touch the CUDA driver."""
            return nullcontext()

        def override_compile_or_warm_up_model(self):
            """Skip compilation and warmup entirely."""
            from vllm.v1.worker.worker_base import CompilationTimes

            logger.info("[vLLM Hijack] Worker.compile_or_warm_up_model: skipped")
            return CompilationTimes(language_model=0.0, encoder=0.0)

        def override_sleep(self, level=1):
            pass

        def override_wake_up(self, tags=None):
            pass

        target.init_device = override_init_device
        target.determine_available_memory = override_determine_available_memory
        target._maybe_get_memory_pool_context = override_maybe_get_memory_pool_context
        target.compile_or_warm_up_model = override_compile_or_warm_up_model
        target.sleep = override_sleep
        target.wake_up = override_wake_up
