# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from typing import cast

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.system_utils import decorate_logs, get_mp_context, set_process_title
from vllm.v1.engine.utils import set_assigned_physical_gpu_ids_for_dp_rank
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.utils import shutdown

logger = init_logger(__name__)


def _run_headless_executor(
    vllm_config: VllmConfig,
    dp_rank: int,
    local_dp_rank: int,
    user_assigned_gpu_ids: list[int] | None,
) -> None:
    parallel_config = vllm_config.parallel_config
    parallel_config.data_parallel_rank = dp_rank
    parallel_config.data_parallel_index = dp_rank
    parallel_config.data_parallel_rank_local = local_dp_rank
    set_assigned_physical_gpu_ids_for_dp_rank(
        vllm_config, local_dp_rank, user_assigned_gpu_ids
    )

    process_name = f"HeadlessExecutor_DP{dp_rank}"
    set_process_title(process_name)
    decorate_logs()

    executor = MultiprocExecutor(vllm_config, monitor_workers=False)
    try:
        executor.start_worker_monitor(inline=True)
    finally:
        executor.shutdown()


class HeadlessMultiprocExecutorManager:
    def __init__(self, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        user_assigned_gpu_ids = parallel_config.assigned_physical_gpu_ids
        context = get_mp_context()
        self.processes: list[BaseProcess] = []
        for dp_rank in range(parallel_config.data_parallel_size):
            process = context.Process(
                target=_run_headless_executor,
                name=f"HeadlessExecutor_DP{dp_rank}",
                args=(vllm_config, dp_rank, dp_rank, user_assigned_gpu_ids),
            )
            self.processes.append(process)

        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        try:
            for process in self.processes:
                process.start()
        finally:
            if self.finished_processes():
                self.shutdown()

    def monitor_liveness(self) -> None:
        sentinel_to_process = {
            cast(int, process.sentinel): process for process in self.processes
        }
        live_sentinels = set(sentinel_to_process)
        while live_sentinels:
            exited = connection.wait(live_sentinels, timeout=1)
            if not exited:
                continue
            for sentinel in exited:
                process = sentinel_to_process[cast(int, sentinel)]
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"Headless executor {process.name} exited with code "
                        f"{process.exitcode}."
                    )
            break

    def shutdown(self, timeout: float | None = None) -> None:
        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)

    def finished_processes(self) -> dict[str, int]:
        return {
            process.name: process.exitcode
            for process in self.processes
            if process.exitcode is not None
        }
