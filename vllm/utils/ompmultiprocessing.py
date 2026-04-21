# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OMP Aware Multiprocessing manager for running multiprocessing.Process()
Copyright (c) 2026 Red Hat Inc
Copyright (c) 2026 Cambridge Greys Ltd
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

import vllm.utils.cpu_resource_utils as cr_utils
from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.cpu_resource_utils import LogicalCPUInfo

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class OMPProcessManager:
    def __init__(self, config: "VllmConfig"):
        if not current_platform.is_cpu():
            return

        self.local_world_size = config.parallel_config.local_world_size
        self.local_dp_rank = config.parallel_config.data_parallel_rank_local
        # This is a bit tricky because the internal DP size
        # is always 1 for non-MoE models
        self.internal_dp_size = config.parallel_config._api_process_count

        self.simulate_multi_node = os.environ.get("VLLM_CPU_SIM_MULTI_NUMA", "0") != "0"
        ld_preload_str = os.getenv("LD_PRELOAD", "")
        self.use_iomp = "libiomp" in ld_preload_str or "libomp" in ld_preload_str
        self.use_gomp = "libgomp" in ld_preload_str

        assert not (self.use_iomp and self.use_gomp)

        # at least reserve 1/local_world_size(for ARM) core for scheduler
        # proc as always use MP executor
        # TODO: make scheduler proc sleep when idle
        self.reserve_cpu_num = (
            self.local_world_size
            if current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            else 1
        )
        # reserve at one more core for nixl_connector under p/d case
        if config.kv_transfer_config:
            self.reserve_cpu_num += 1

        if envs.VLLM_CPU_NUM_OF_RESERVED_CPU is not None:
            if self.reserve_cpu_num > envs.VLLM_CPU_NUM_OF_RESERVED_CPU:
                msg = (
                    f"VLLM_CPU_NUM_OF_RESERVED_CPU is less than "
                    "the minimum requirement"
                    f": {self.reserve_cpu_num} cores"
                )
                logger.warning(msg=msg)
            self.reserve_cpu_num = envs.VLLM_CPU_NUM_OF_RESERVED_CPU

        self._parse_omp_threads_bind_env()

        assert not self.simulate_multi_node or self.auto_setup

    @contextmanager
    def configure_omp_envs(self, rank: int, local_rank: int):
        if not current_platform.is_cpu() or self.skip_setup:
            yield
            return

        envs_dict = {}
        cpu_list = [str(i) for i in self.cpu_lists[local_rank]]
        envs_dict["OMP_NUM_THREADS"] = str(len(cpu_list))
        if self.use_iomp:
            # set IOMP envs
            cpu_list_str = ",".join(cpu_list)
            envs_dict["KMP_AFFINITY"] = (
                f"granularity=fine,explicit,proclist=[{cpu_list_str}]"
            )
            # The time(milliseconds) that a thread should wait after
            # completing the execution of a parallel region, before sleeping.
            envs_dict["KMP_BLOCKTIME"] = "1"
            # Prevents the CPU to run into low performance state
            envs_dict["KMP_TPAUSE"] = "0"
            # Provides fine granularity parallelism
            envs_dict["KMP_FORKJOIN_BARRIER_PATTERN"] = "dist,dist"
            envs_dict["KMP_PLAIN_BARRIER_PATTERN"] = "dist,dist"
            envs_dict["KMP_REDUCTION_BARRIER_PATTERN"] = "dist,dist"
        elif self.use_gomp:
            # set GOMP envs
            # likes '0 1 2 ...'
            cpu_list_str = " ".join(cpu_list)
            envs_dict["GOMP_CPU_AFFINITY"] = cpu_list_str
        else:
            # set OMP envs
            # likes '{0,1,2,...}'
            cpu_list_str = ",".join(cpu_list)
            envs_dict["OMP_PLACES"] = f"{{{cpu_list_str}}}"
            envs_dict["OMP_PROC_BIND"] = "true"

        # backup envs
        old_envs_dict = {}
        for k in envs_dict:
            old_envs_dict[k] = os.environ.get(k)

        try:
            # set envs
            for k, v in envs_dict.items():
                os.environ[k] = v
            yield
        finally:
            # restore old envs
            for k, v in old_envs_dict.items():  # type: ignore
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _parse_omp_threads_bind_env(self):
        vllm_mask = envs.VLLM_CPU_OMP_THREADS_BIND
        self.skip_setup = vllm_mask == "nobind"
        self.auto_setup = vllm_mask == "auto"
        self.reserved_cpu_list = []
        self.cpu_lists = []

        if self.auto_setup:
            # auto generate CPU lists
            cpu_arch = current_platform.get_cpu_architecture()
            if cpu_arch == CpuArchEnum.POWERPC:
                # For POWERPC SMT-8/4/2
                cpu_list, reserve_list = self._get_autobind_cpu_ids(
                    lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4]
                )
            elif cpu_arch in (CpuArchEnum.X86, CpuArchEnum.S390X):
                # For x86/S390X SMT-2, use 1 logical CPU per physical core
                cpu_list, reserve_list = self._get_autobind_cpu_ids(
                    lambda cpus: cpus[-1:]
                )
            elif cpu_arch == CpuArchEnum.ARM:
                # For AArch64, no SMT, use all logical CPU
                cpu_list, reserve_list = self._get_autobind_cpu_ids(lambda cpus: cpus)
            else:
                cpu_list, reserve_list = [], []
                raise RuntimeError(f"{cpu_arch} doesn't support auto CPU binding.")

            for item in cpu_list:
                self.cpu_lists.append([x.id for x in item])
            self.reserved_cpu_list = [x.id for x in reserve_list]
        elif not self.skip_setup:
            # user defined CPU lists
            omp_cpuids_list = vllm_mask.split("|")
            if self.local_dp_rank is not None:
                local_dp_rank = self.local_dp_rank
                world_size = self.local_world_size
                # Rank mapping [DP, PP, TP]
                omp_cpuids_list = omp_cpuids_list[
                    local_dp_rank * world_size : (local_dp_rank + 1) * world_size
                ]

            assert len(omp_cpuids_list) == self.local_world_size, (
                "Given "
                f"number of CPU id list {omp_cpuids_list} doesn't match "
                f"local world size {self.local_world_size}."
            )

            # parse CPU list strings like "5,2-4" to [5, 2, 3, 4]
            self.cpu_lists = [cr_utils.parse_id_list(s) for s in omp_cpuids_list]
        else:
            # skip
            self.cpu_lists = []

        msg = "OpenMP thread binding info: \n"
        for i in range(self.local_world_size):
            msg += f"\tlocal_rank={i}, core ids={self.cpu_lists[i]}\n"
        msg += f"\treserved_cpus={self.reserved_cpu_list}"
        logger.info(msg)

    def _get_autobind_cpu_ids(
        self, cpu_selector: Callable[[list[LogicalCPUInfo]], list[LogicalCPUInfo]]
    ) -> tuple[list[list[LogicalCPUInfo]], list[LogicalCPUInfo]]:
        """
        Return CPU ids to bind based on NUMA nodes, and CPU ids reserved for
        other processes.
        Currently for rank N, only CPU ids on the N-th node in available NUMA
        node list will be selected.
        Args:
            cpu_selector: a callable object to select CPUs from a CPU list
            of a physical core. The input is a LogicalCPUInfo list contains
            logical CPUs of a physical CPU, sorted by the LogicalCPUInfo.id.
            A selected LogicalCPUInfo list should be returned.
        """

        # this memory node list has been sliced for DP offset
        allowed_numa_nodes = cr_utils.get_visible_memory_node()
        logical_cpu_list = cr_utils.get_allowed_cpu_list()

        local_world_size = self.local_world_size
        assert (
            len(allowed_numa_nodes) >= local_world_size or self.simulate_multi_node
        ), (
            f"Not enough allowed NUMA nodes to bind threads of "
            f"{local_world_size} local CPUWorkers. "
            f"Allowed NUMA nodes are {allowed_numa_nodes}. "
            "Please try to bind threads manually or decrease DP/TP/PP."
        )

        # Generate OMP CPU list for each rank
        cpu_lists_of_ranks = []
        reserved_cpu_list = []
        total_cpu_num = 0
        for local_rank in range(self.local_world_size):
            if not self.simulate_multi_node:
                selected_numa_node = allowed_numa_nodes[local_rank]
                selected_logical_cpu_list = [
                    x for x in logical_cpu_list if x.numa_node == selected_numa_node
                ]
            else:
                world_size_across_dp = self.local_world_size * self.internal_dp_size
                assert len(logical_cpu_list) >= world_size_across_dp
                selected_logical_cpu_list = sorted(
                    logical_cpu_list, key=lambda x: x.numa_node
                )
                sim_cpu_num_per_node = (
                    len(selected_logical_cpu_list) // world_size_across_dp
                )
                assert self.local_dp_rank is not None
                start_idx = (
                    local_rank + self.local_world_size * self.local_dp_rank
                ) * sim_cpu_num_per_node
                selected_logical_cpu_list = selected_logical_cpu_list[
                    start_idx : (start_idx + sim_cpu_num_per_node)
                ]

            # Select logical CPUs on same physical cores via cpu_selector
            core_to_cpus: dict[int, list[LogicalCPUInfo]] = {}
            for cpu_info in selected_logical_cpu_list:
                if cpu_info.physical_core not in core_to_cpus:
                    core_to_cpus[cpu_info.physical_core] = []
                core_to_cpus[cpu_info.physical_core].append(cpu_info)
            selected_logical_cpu_list = []
            for cpu_list in core_to_cpus.values():
                cpu_list = sorted(cpu_list, key=lambda x: x.id)
                selected_logical_cpu_list.extend(cpu_selector(cpu_list))

            # sort selected cores based on core id
            selected_logical_cpu_list = sorted(
                selected_logical_cpu_list, key=lambda x: x.id
            )

            cpu_lists_of_ranks.append(selected_logical_cpu_list)
            total_cpu_num += len(selected_logical_cpu_list)

        # Reserve CPUs for other processes
        if total_cpu_num <= self.reserve_cpu_num:
            logger.warning(
                "Selected CPU core number (%s) "
                "should be greater than reserved CPU core "
                "number (%s).",
                total_cpu_num,
                self.reserve_cpu_num,
            )
            return cpu_lists_of_ranks, []

        reserve_num_per_rank = [
            self.reserve_cpu_num // self.local_world_size
        ] * self.local_world_size
        # last rank first
        for i in range(
            self.local_world_size - 1,
            self.local_world_size - 1 - self.reserve_cpu_num % self.local_world_size,
            -1,
        ):
            reserve_num_per_rank[i] += 1
        for i in range(self.local_world_size):
            num = reserve_num_per_rank[i]
            if num > 0:
                reserved_cpu_list.extend(cpu_lists_of_ranks[i][-num:])
                cpu_lists_of_ranks[i] = cpu_lists_of_ranks[i][:-num]

        return cpu_lists_of_ranks, reserved_cpu_list
