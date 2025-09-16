# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import platform
from typing import Callable, Optional

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import CpuArchEnum, current_platform
from vllm.platforms.cpu import CpuPlatform, LogicalCPUInfo
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import (Worker,
                                       init_worker_distributed_environment)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

logger = init_logger(__name__)


class CPUWorker(Worker):

    def __init__(self,
                 vllm_config: VllmConfig,
                 local_rank: int,
                 rank: int,
                 distributed_init_method: str,
                 is_driver_worker: bool = False):
        super().__init__(vllm_config,
                         local_rank,
                         rank,
                         distributed_init_method,
                         is_driver_worker=is_driver_worker)

        self.parallel_config.disable_custom_all_reduce = True

    def init_device(self):
        # Setup OpenMP threads affinity.
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        if omp_cpuids == "auto" and platform.system() == "Linux":
            cpu_arch = current_platform.get_cpu_architecture()
            if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
                # For S390X/POWERPC SMT-8/4/2
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4])
            elif current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                # For x86 SMT-2, use 1 CPU per core
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: cpus[-1:])
            else:
                self.local_omp_cpuid = "all"
        else:
            local_dp_rank = self.parallel_config.data_parallel_rank_local
            omp_cpuids = omp_cpuids.split("|")
            if local_dp_rank is not None:
                world_size = self.parallel_config.world_size
                omp_cpuids = omp_cpuids[local_dp_rank *
                                        world_size:(local_dp_rank + 1) *
                                        world_size]
            self.local_omp_cpuid = omp_cpuids[self.rank]

        if self.local_omp_cpuid != "all":
            ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
            if ret:
                logger.info(ret)

        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(
            ":")[-1]
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            current_platform.dist_backend)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: CPUModelRunner = CPUModelRunner(
            self.vllm_config, torch.device("cpu"))

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        return self.cache_config.cpu_kvcache_space_bytes  # type: ignore

    def compile_or_warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        self.model_runner.warming_up_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = self.model_runner._get_num_input_tokens(
            num_scheduled_tokens)
        all_gather_tensors = {
            "residual":
            not is_residual_scattered_for_sp(self.vllm_config,
                                             num_input_tokens)
        }
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors)
            return None

        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None

    def _get_autobind_cpu_ids(
        self, cpu_selector: Callable[[list[LogicalCPUInfo]],
                                     list[LogicalCPUInfo]]
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

        allowed_numa_nodes, logical_cpu_list = \
            CpuPlatform.get_allowed_cpu_core_node_list()
        assert len(allowed_numa_nodes) >= self.parallel_config.world_size, (
            f"No enough allowed NUMA nodes to bind threads of "
            f"{self.parallel_config.world_size} CPUWorkers. "
            f"Allowed NUMA nodes are {allowed_numa_nodes}. "
            "Please try to bind threads manually.")

        # Get CPUs on NUMA node `allowed_numa_nodes[local_rank]``
        selected_numa_node = allowed_numa_nodes[
            self.local_rank]  # type: ignore
        logical_cpu_list = [
            x for x in logical_cpu_list if x.numa_node == selected_numa_node
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
            need_reserve = (self.parallel_config.world_size > 1 or
                            self.parallel_config.data_parallel_size_local > 1)
            reserve_cpu_num = 1 if need_reserve else 0
        assert len(logical_cpu_list) > reserve_cpu_num, (
            f"VLLM_CPU_NUM_OF_RESERVED_CPU ({reserve_cpu_num}) "
            f"should less than {len(logical_cpu_list)}.")
        if reserve_cpu_num != 0:
            logical_cpu_list = logical_cpu_list[:-reserve_cpu_num]

        logger.info("auto thread-binding list (id, physical core): %s",
                    [(x.id, x.physical_core) for x in logical_cpu_list])
        return ",".join([str(x.id) for x in logical_cpu_list])
