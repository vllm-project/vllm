# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for EPLB (Expert Parallel Load Balancing)."""

import os
import threading

import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class CpuGpuEvent:
    """
    Combines a CUDA event with a CPU threading event to enforce record->wait
    ordering across two threads.

    CUDA events alone are insufficient for cross-thread synchronization because
    waiting on an unrecorded CUDA event is a no-op. The wait will return
    immediately instead of blocking. This class adds a threading.Event so
    that the waiting thread blocks on the CPU side until record() has
    actually been called, at which point the CUDA event is guaranteed to be
    in-flight and event.wait() will correctly synchronize the GPU stream.
    """

    def __init__(self):
        self._event = torch.cuda.Event()
        self._recorded = threading.Event()

    def wait(self, stream: torch.cuda.Stream | None = None):
        """
        Blocks the calling thread until record finishes. Used to guarantee that the
        record kernel is called before wait.

        Should only be called by the Async Eplb thread.
        """
        self._recorded.wait()
        self._event.wait(stream)
        self._recorded.clear()

    def record(self, stream: torch.cuda.Stream | None = None):
        """
        Unblocks the waiting thread after calling event.record().

        Should only be called by the main thread.
        """
        assert not self._recorded.is_set()
        self._event = torch.cuda.Event()
        self._event.record(stream)
        self._recorded.set()


def override_envs_for_eplb(parallel_config: ParallelConfig) -> None:
    """
    Override environment variables for EPLB when specific conditions are met.

    Args:
        parallel_config: The parallel configuration object.
    """
    is_data_parallel = parallel_config.data_parallel_size > 1
    is_eplb_enabled = parallel_config.enable_eplb
    async_eplb = parallel_config.eplb_config.use_async
    is_deepep_ll = parallel_config.all2all_backend == "deepep_low_latency"

    # Override NCCL_MAX_CTAS to avoid hangs when using async EPLB with the
    # DeepEP low-latency backend.
    #
    # The hang happens when two ranks interleave kernel launches differently
    # between NCCL collectives (used by async EPLB weight exchange) and DeepEP
    # low-latency (LL) kernels. DeepEP LL uses a cooperative launch and tries
    # to reserve a large fraction of the GPU's SMs; if those SMs are currently
    # occupied by NCCL, the DeepEP LL launch blocks until enough SMs are
    # freed.
    #
    # If rank A enters DeepEP LL in main thread while rank B is still executing
    # NCCL in async thread, rank A can block waiting for SMs, while rank B can
    # block inside NCCL waiting for rank A to participate in the collective.
    # This circular wait causes a deadlock.
    # Limiting NCCL occupancy via NCCL_MAX_CTAS leaves space for the DeepEP
    # cooperative kernel to launch and complete, breaking the deadlock.
    # See: https://github.com/deepseek-ai/DeepEP/issues/496
    if is_data_parallel and is_eplb_enabled and is_deepep_ll and async_eplb:
        current_value_str = os.getenv("NCCL_MAX_CTAS")

        if current_value_str and current_value_str.isdigit():
            return

        override_value = 8
        os.environ["NCCL_MAX_CTAS"] = str(override_value)
        logger.info_once(
            f"EPLB: Setting NCCL_MAX_CTAS={override_value} "
            "for expert parallel with EPLB and deepep_low_latency backend",
            scope="global",
        )
