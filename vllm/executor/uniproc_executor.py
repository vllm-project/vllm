# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import worker_receiver_cache_from_config
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        run_method)
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.utils import get_and_update_mm_cache
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class UniProcExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        is_driver_worker = True
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.mm_receiver_cache = worker_receiver_cache_from_config(
            self.vllm_config, MULTIMODAL_REGISTRY, Lock())

        self.async_output_thread: Optional[ThreadPoolExecutor] = None
        if self.max_concurrent_batches > 1:
            self.async_output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="WorkerAsyncOutput")

        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(
            ":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        return 2 if self.scheduler_config.async_scheduling else 1

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None,
                       non_block: bool = False) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        if self.mm_receiver_cache is not None and method == "execute_model":
            get_and_update_mm_cache(self.mm_receiver_cache, args)

        if not non_block:
            return [run_method(self.driver_worker, method, args, kwargs)]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                if (async_thread := self.async_output_thread) is not None:
                    return [async_thread.submit(result.get_output)]
                result = result.get_output()
            future = Future[Any]()
            future.set_result(result)
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return [future]

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        self.driver_worker.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            self.shutdown()
        return

    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()


UniProcExecutorAsync = UniProcExecutor


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """
    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        if envs.VLLM_USE_V1:
            assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, \
            ("To get deterministic execution in V1, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0")
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks.
        Add an additional all_reduce to get the min across all ranks.
        Note that even if we have the same `gpu_memory_utilization` and 
        `swap_space`, the available memory in every rank might still 
        differ because NCCL can take different amounts of memory in 
        different ranks. Therefore, it is necessary to test if all ranks 
        agree on the same KV cache configuration.
        """
        a, b = super().determine_num_available_blocks()
        from vllm.distributed.parallel_state import get_world_group
        cpu_group = get_world_group().cpu_group
        a_tensor = torch.tensor([a], device="cpu", dtype=torch.int64)
        b_tensor = torch.tensor([b], device="cpu", dtype=torch.int64)
        dist.all_reduce(a_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        dist.all_reduce(b_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return a_tensor.item(), b_tensor.item()
