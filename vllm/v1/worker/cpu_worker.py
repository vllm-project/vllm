# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import platform
from collections.abc import Callable
from typing import Any

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment

logger = init_logger(__name__)


class CPUWorker(Worker):
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

        self.parallel_config.disable_custom_all_reduce = True

        # Torch profiler. Enabled and configured through profiler_config.
        self.profiler: Any | None = None
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )

    def init_device(self):
        # Setup OpenMP threads affinity.
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        # Under numa binding some cores reserved for kv transfer in nixl_connector.py
        if omp_cpuids == "auto" and platform.system() == "Linux":
            cpu_arch = current_platform.get_cpu_architecture()
            if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
                # For S390X/POWERPC SMT-8/4/2
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4]
                )
            elif cpu_arch == CpuArchEnum.X86:
                # For x86 SMT-2, use 1 CPU per core
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: cpus[-1:]
                )
            elif cpu_arch == CpuArchEnum.ARM:
                # For AArch64, no SMT
                self.local_omp_cpuid = self._get_autobind_cpu_ids(lambda cpus: cpus)
            else:
                self.local_omp_cpuid = "nobind"
        elif omp_cpuids == "nobind":
            self.local_omp_cpuid = "nobind"
        else:
            local_dp_rank = self.parallel_config.data_parallel_rank_local
            omp_cpuids_list = omp_cpuids.split("|")
            if local_dp_rank is not None:
                world_size = self.parallel_config.world_size
                omp_cpuids_list = omp_cpuids_list[
                    local_dp_rank * world_size : (local_dp_rank + 1) * world_size
                ]
            self.local_omp_cpuid = omp_cpuids_list[self.rank]

        if self.local_omp_cpuid != "nobind":
            ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
            if ret:
                logger.info(ret)

        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: CPUModelRunner = CPUModelRunner(
            self.vllm_config, torch.device("cpu")
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        return self.cache_config.cpu_kvcache_space_bytes or 0

    def compile_or_warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        self.model_runner.warming_up_model()

    def _get_autobind_cpu_ids(
        self, cpu_selector: Callable[[list[LogicalCPUInfo]], list[LogicalCPUInfo]]
    ) -> str:
        """
        Return CPU ids to bind based on NUMA nodes.
        Currently for rank N, only CPU ids on the N-th node in available NUMA
        node list will be selected.
        Args:
            cpu_selector: a callable object to select CPUs from a CPU list
            of a physical core. The input is a LogicalCPUInfo list, sorted by
            the LogicalCPUInfo.id. A selected LogicalCPUInfo list should be
            returned.
        """
        # simulate multiple numa nodes, for testing
        sim_multi_numa_nodes = os.environ.get("VLLM_CPU_SIM_MULTI_NUMA", "0") != "0"

        allowed_numa_nodes, logical_cpu_list = (
            CpuPlatform.get_allowed_cpu_core_node_list()
        )
        assert (
            len(allowed_numa_nodes) >= self.parallel_config.world_size
            or sim_multi_numa_nodes
        ), (
            f"Not enough allowed NUMA nodes to bind threads of "
            f"{self.parallel_config.world_size} CPUWorkers. "
            f"Allowed NUMA nodes are {allowed_numa_nodes}. "
            "Please try to bind threads manually."
        )

        if not sim_multi_numa_nodes:
            # Get CPUs on NUMA node `allowed_numa_nodes[local_rank]`
            selected_numa_node = allowed_numa_nodes[self.local_rank]  # type: ignore
            logical_cpu_list = [
                x for x in logical_cpu_list if x.numa_node == selected_numa_node
            ]
        else:
            assert len(logical_cpu_list) >= self.parallel_config.world_size
            logical_cpu_list = sorted(logical_cpu_list, key=lambda x: x.numa_node)
            sim_cpu_num_per_node = (
                len(logical_cpu_list) // self.parallel_config.world_size
            )
            start_idx = self.local_rank * sim_cpu_num_per_node
            logical_cpu_list = logical_cpu_list[
                start_idx : (start_idx + sim_cpu_num_per_node)
            ]

        # Select CPUs from each physical core via cpu_selector
        core_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
        for cpu_info in logical_cpu_list:
            if cpu_info.physical_core not in core_to_cpus:
                core_to_cpus[cpu_info.physical_core] = []
            core_to_cpus[cpu_info.physical_core].append(cpu_info)
        logical_cpu_list = []
        for cpu_list in core_to_cpus.values():
            cpu_list = sorted(cpu_list, key=lambda x: x.id)
            logical_cpu_list.extend(cpu_selector(cpu_list))
        logical_cpu_list = sorted(logical_cpu_list, key=lambda x: x.id)

        # Reserve CPUs for other processes
        reserve_cpu_num = envs.VLLM_CPU_NUM_OF_RESERVED_CPU
        if reserve_cpu_num is None:
            need_reserve = (
                self.parallel_config.world_size > 1
                or self.parallel_config.data_parallel_size_local > 1
            )
            reserve_cpu_num = 1 if need_reserve else 0
        assert len(logical_cpu_list) > reserve_cpu_num, (
            f"VLLM_CPU_NUM_OF_RESERVED_CPU ({reserve_cpu_num}) "
            f"should less than {len(logical_cpu_list)}."
        )
        if reserve_cpu_num != 0:
            logical_cpu_list = logical_cpu_list[:-reserve_cpu_num]

        logger.info(
            "auto thread-binding list (id, physical core): %s",
            [(x.id, x.physical_core) for x in logical_cpu_list],
        )
        return ",".join([str(x.id) for x in logical_cpu_list])

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
