# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import platform
import sys

import pytest

from vllm.utils.network_utils import get_file_store_init_method
from vllm.v1.executor import UniProcExecutor
from vllm.v1.worker.worker_base import WorkerWrapperBase

RUNAI_PLATFORM_SUPPORTED = sys.platform == "linux" and platform.machine() in {
    "aarch64",
    "x86_64",
}
RUNAI_PLATFORM_SKIP_REASON = (
    "runai-model-streamer only provides Linux x86_64 and aarch64 wheels"
)
requires_runai = pytest.mark.skipif(
    not RUNAI_PLATFORM_SUPPORTED,
    reason=RUNAI_PLATFORM_SKIP_REASON,
)


# This is a dummy executor for patching in test_runai_model_streamer_s3.py.
# We cannot use vllm_runner fixture here, because it spawns worker process.
# The worker process reimports the patched entities, and the patch is not applied.
class RunaiDummyExecutor(UniProcExecutor):
    def _init_executor(self) -> None:
        distributed_init_method = get_file_store_init_method()

        local_rank = 0
        rank = 0
        is_driver_worker = True

        device_info = self.vllm_config.device_config.device.__str__().split(":")
        if len(device_info) > 1:
            local_rank = int(device_info[1])

        worker_rpc_kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.driver_worker = WorkerWrapperBase()

        self.collective_rpc("init_worker", args=([worker_rpc_kwargs],))
        self.collective_rpc("init_device")
