# SPDX-License-Identifier: Apache-2.0
import os
from importlib import util
from typing import Optional

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import (Worker,
                                       init_worker_distributed_environment)

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
        self.local_omp_cpuid = "all"
        if omp_cpuids == "auto":
            self.local_omp_cpuid = self.get_cpus_id_binding_based_on_numa_nodes(
            )
        else:
            self.local_omp_cpuid = omp_cpuids.split("|")[self.rank]

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
                                            self.local_rank, "gloo")
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
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None

        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None

    def get_cpus_id_binding_based_on_numa_nodes(self) -> str:
        """Return CPUs id binding based on NUMA nodes.
        """
        rank_to_cpus = self.local_omp_cpuid
        # Setup OpenMP thread affinity based on NUMA nodes automatically
        world_size = self.vllm_config.parallel_config.world_size
        libnuma_found = util.find_spec("numa") is not None
        psutil_found = util.find_spec("psutil") is not None
        if libnuma_found and psutil_found:
            import psutil
            from numa import info
            cpu_count = psutil.cpu_count(logical=False)
            cpus_allow_list = psutil.Process().cpu_affinity()
            numa_size = info.get_num_configured_nodes()
            cpu_count_per_numa = cpu_count // numa_size
            num_of_reserved_cpu = min(envs.VLLM_CPU_NUM_OF_RESERVED_CPU,
                                      cpu_count_per_numa // 2)

            # check allow node_to_cpus list
            node_to_cpus = []
            for i in range(numa_size):
                node_intersect = set(
                    info.node_to_cpus(i)).intersection(cpus_allow_list)
                if bool(node_intersect):
                    node_to_cpus.append(list(node_intersect))

            if world_size > len(node_to_cpus):
                logger.error(
                    "Auto thread-binding failed due to "
                    "world size: %d is larger than "
                    "allowed NUMA nodes number: %d."
                    "Please try to bind threads manually.", world_size,
                    len(node_to_cpus))
            else:
                end = cpu_count_per_numa - num_of_reserved_cpu
                rank_to_cpus_list = node_to_cpus[self.rank][:end]
                rank_to_cpus = ','.join(str(x) for x in rank_to_cpus_list)
                logger.info("auto thread-binding list: %s", rank_to_cpus)
        else:
            logger.warning(
                "Auto thread-binding is not supported due to "
                "the lack of package numa and psutil,"
                "fallback to no thread-binding. To get better performance,"
                "please try to manually bind threads.")
        return rank_to_cpus
