# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import cast

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    GroupCoordinator,
    cleanup_dist_env_and_memory,
    get_all_model_groups,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.v1.fault_tolerance.sentinel import BaseSentinel


class WorkerSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        pause_event: threading.Event,
        init_distributed_env_callback: Callable,
        clear_input_batch_callback: Callable,
        device: torch.cuda.device,
    ):
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity = f"PP{pp_rank}_TP{tp_rank}"
        super().__init__(
            upstream_cmd_addr=vllm_config.fault_tolerance_config.worker_cmd_addr,
            downstream_cmd_addr=None,
            sentinel_identity=identity.encode(),
            sentinel_tag=f"{dp_rank}_{identity}",
            vllm_config=vllm_config,
        )
        self.init_distributed_env_callback = init_distributed_env_callback
        self.clear_input_batch_callback = clear_input_batch_callback
        self.device = device

        self.pause_event = pause_event
        self.communicator_aborted = False
        torch.cuda.set_device(self.device)
        threading.Thread(
            target=self.run, daemon=True, name="WorkerSentinelMonitorThread"
        ).start()

    def run(self):
        # Wait for fault tolerance instructions from EngineCoreSentinel
        while not self.sentinel_dead:
            self.poll_and_execute_upstream_cmd()

    def pause(self, timeout: int = 1, **kwargs) -> bool:
        soft_pause = kwargs.get("soft_pause", False)
        if soft_pause:
            self._set_device_communicator_status(False)
            self.pause_event.set()
            self.logger("Pause signal sent.")
            return True
        # Abort all NCCL communicators and
        # process groups in parallel using a thread pool.
        if self.communicator_aborted:
            return True
        self.pause_event.set()
        self._set_device_communicator_status(False)
        torch.cuda.set_device(self.device)
        model_groups = get_all_model_groups()
        futures = []

        def _abort_nccl_comm(group: GroupCoordinator):
            if group.device_communicator is not None:
                device_comm = cast(CudaCommunicator, group.device_communicator)
                nccl_comm = device_comm.pynccl_comm
                assert nccl_comm is not None
                nccl_comm.nccl_abort_comm()

        def _abort_process_group(group: GroupCoordinator):
            backend = group.device_group._get_backend(self.device)
            backend.abort()

        executor = ThreadPoolExecutor(max_workers=len(model_groups) * 2)
        try:
            for group in model_groups:
                futures.append(executor.submit(_abort_nccl_comm, group))
                futures.append(executor.submit(_abort_process_group, group))

            done, not_done = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)
            if not_done:
                self.logger(
                    "%d abort calls did not finish in total %s seconds",
                    len(not_done),
                    timeout,
                    level="warning",
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        exception_count = sum(1 for f in done if f.exception() is not None)
        self.communicator_aborted = len(not_done) == 0 and exception_count == 0
        if self.communicator_aborted:
            cleanup_dist_env_and_memory()
            self.logger("Communicators are aborted.")
        else:
            self.logger(
                "Communicator abort failed: %d NCCL comm abort calls timed out,"
                " %d tasks threw exceptions. This may leave NCCL communicators "
                "or process groups in an inconsistent state. Subsequent "
                "distributed operations could be unsafe.",
                len(not_done),
                exception_count,
                level="error",
            )
        return self.communicator_aborted

    def _set_device_communicator_status(self, active: bool):
        model_groups = get_all_model_groups()
        for group in model_groups:
            if group.device_communicator is not None:
                device_comm = cast(CudaCommunicator, group.device_communicator)
                nccl_comm = device_comm.pynccl_comm
                assert nccl_comm is not None
                nccl_comm.available = active
                nccl_comm.disabled = not active

    def retry(self, timeout: int = 1, **kwargs) -> bool:
        if self.communicator_aborted:
            torch.cuda.set_device(self.device)
            with set_current_vllm_config(self.vllm_config):
                self.init_distributed_env_callback()
                self.communicator_aborted = False
            torch.cuda.synchronize()
        self.clear_input_batch_callback()
        self.pause_event.clear()
        return True
