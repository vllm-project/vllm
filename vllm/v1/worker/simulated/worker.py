# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker for simulated execution: no weights, no KV tensors, no forward."""

# ruff: noqa: E402
import torch

from vllm.platforms import current_platform

if current_platform.is_cpu():
    # Stubs torch's CUDA stream/event APIs for the GPU runner, as in
    # cpu_worker.py; must precede its import.
    import vllm.v1.worker.cpu.shm  # noqa: F401

from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.simulated.model_runner import SimulatedModelRunner
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)


class SimulatedWorker(Worker):
    """Worker for simulated execution: no weights, no KV tensors, no forward."""

    _kv_connector: KVConnectorBaseType | None = None

    def init_device(self) -> None:
        if current_platform.is_cuda_alike():
            self._bind_device()
        else:
            from vllm.v1.worker.cpu_worker import set_dist_ident

            self.device = torch.device("cpu")
            set_dist_ident(self.distributed_init_method)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        set_random_seed(self.model_config.seed)
        logger.info(
            "Simulated execution on %s (timing=%s)",
            self.device,
            self.vllm_config.simulation_config.timing,
        )
        self._init_workspace_and_model_runner()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # Engines simulated in one process (disaggregated instances) each
        # build their own connector, which is a process global.
        kv_transfer_state.set_kv_transfer_group(None)
        super().initialize_from_config(kv_cache_config)
        if kv_transfer_state.has_kv_transfer_group():
            self._kv_connector = kv_transfer_state.get_kv_transfer_group()

    def shutdown(self) -> None:
        # Shut down this engine's connector, not the last one built.
        kv_transfer_state.set_kv_transfer_group(self._kv_connector)
        super().shutdown()

    def execute_model(self, scheduler_output):  # type: ignore[override]
        """No hidden states flow between pipeline stages: every stage runs
        its own bookkeeping and timing from the same scheduler output."""
        with self.annotate_profile(scheduler_output):
            return self.model_runner.execute_model(scheduler_output)

    def _make_model_runner(self) -> SimulatedModelRunner:  # type: ignore[override]
        return SimulatedModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        """Virtual KV capacity. Nothing is profiled: the KV cache is sized from
        `kv_cache_memory_bytes`, or from the device's total memory and
        `gpu_memory_utilization` when that is unset."""
        kv_bytes = self.cache_config.kv_cache_memory_bytes
        if kv_bytes is None:
            assert self.device is not None
            if self.device.type == "cpu":
                import psutil

                total = psutil.virtual_memory().total
            else:
                total = torch.accelerator.get_device_properties(
                    self.device
                ).total_memory  # type: ignore[attr-defined]
            kv_bytes = int(total * self.cache_config.gpu_memory_utilization)
            logger.warning(
                "kv_cache_memory_bytes is unset; simulating %s GiB of KV cache "
                "from the local device. Pass --kv-cache-memory-bytes to model "
                "the target deployment.",
                format_gib(kv_bytes),
            )
        logger.info("Using %s GiB of virtual KV cache.", format_gib(kv_bytes))
        return kv_bytes

    def compile_or_warm_up_model(self) -> CompilationTimes:
        assert isinstance(self.model_runner, SimulatedModelRunner)
        self.model_runner.warm_up_step_kernels()
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode is a no-op under simulated execution.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Wake-up is a no-op under simulated execution.")
