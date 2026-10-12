# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker for simulated forward: no weights, no KV tensors, no forward pass."""

# ruff: noqa: E402
import os

import torch

from vllm.platforms import current_platform

if current_platform.is_cpu():
    # Turns torch's CUDA stream/event/pin_memory APIs into no-ops so the GPU
    # model runner's buffers work on a CPU-only host. Must precede the GPU
    # worker and runner imports, as in cpu_worker.py.
    import vllm.v1.worker.cpu.shm  # noqa: F401

from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.simulated.model_runner import SimulatedModelRunner
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)


class SimulatedWorker(Worker):
    """Runs the V2 model runner's request bookkeeping while skipping weight
    loading, KV-cache allocation and the forward pass.

    On a CUDA-alike host the worker still binds a device: the runner keeps its
    small request-state and block-table buffers there and uses the stock
    Triton input-preparation kernels. On the CPU platform the same kernels run
    through Triton's CPU backend. The model lives on the meta device either
    way."""

    def init_device(self) -> None:
        # Nothing to reduce across ranks; the custom all-reduce needs P2P probes.
        self.parallel_config.disable_custom_all_reduce = True

        if current_platform.is_cuda_alike():
            assert self.local_rank < torch.accelerator.device_count(), (
                f"local rank {self.local_rank} is out of bounds for "
                f"{torch.accelerator.device_count()} devices"
            )
            visible = current_platform.logical_device_id_to_visible_device_id(
                self.local_rank
            )
            self.device = torch.device(f"{current_platform.device_type}:{visible}")
            torch.accelerator.set_device_index(self.device)
        else:
            self.device = torch.device("cpu")
            init_method = self.distributed_init_method
            os.environ["VLLM_DIST_IDENT"] = (
                os.path.basename(init_method.removeprefix("file://"))
                if init_method.startswith("file://")
                else init_method.split(":")[-1]
            )

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        set_random_seed(self.model_config.seed)
        logger.info("Simulated forward on %s", self.device)
        self._init_workspace_and_model_runner()

    def _make_model_runner(self) -> SimulatedModelRunner:  # type: ignore[override]
        return SimulatedModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        """Virtual KV capacity: nothing is profiled. The KV cache is sized from
        `kv_cache_memory_bytes`, or from the device's total memory and
        `gpu_memory_utilization` when that is unset."""
        kv_bytes = self.cache_config.kv_cache_memory_bytes
        if kv_bytes is None:
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
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode is a no-op under simulated forward.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Wake-up is a no-op under simulated forward.")
