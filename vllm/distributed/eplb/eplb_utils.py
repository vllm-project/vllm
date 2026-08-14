# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for EPLB (Expert Parallel Load Balancing)."""

import os
import threading
from collections.abc import Callable
from typing import Any

from vllm.config import ParallelConfig
from vllm.distributed.eplb.platform_backend import EplbDeviceEvent
from vllm.logger import init_logger

logger = init_logger(__name__)


class CrossThreadDeviceEvent:
    """
    Combines a device event with a CPU threading event to enforce record->wait
    ordering across two threads.

    This class is designed for exactly two threads: one producer that calls
    record() and one consumer that calls wait(). Using it with more than two
    threads is not supported and will produce undefined behavior.

    Device events alone may not block when waiting before the event is recorded.
    The CPU event ensures record happens before wait across the two EPLB threads.
    """

    def __init__(self, event_factory: Callable[[], EplbDeviceEvent]):
        self._event_factory = event_factory
        self._event = event_factory()
        self._recorded = threading.Event()

    def wait(self, stream: Any | None = None) -> None:
        """
        Blocks the calling thread until record finishes. Used to guarantee that the
        record kernel is called before wait.

        Should only be called by the Async Eplb thread.
        """
        self._recorded.wait()
        self._event.wait(stream)
        self._recorded.clear()

    def record(self, stream: Any | None = None) -> None:
        """
        Unblocks the waiting thread after calling event.record().

        Should only be called by the main thread.
        """
        if self._recorded.is_set():
            raise RuntimeError(
                "CrossThreadDeviceEvent.record() called before the previous "
                "event was "
                "consumed by wait()"
            )
        self._event = self._event_factory()
        self._event.record(stream)
        self._recorded.set()


def override_envs_for_eplb(
    parallel_config: ParallelConfig,
    moe_backend: str | None = None,
) -> None:
    """
    Override environment variables for EPLB when specific conditions are met.

    Args:
        parallel_config: The parallel configuration object.
        moe_backend: The configured MoE backend (e.g. ``deep_gemm_mega_moe``).
    """
    is_data_parallel = parallel_config.data_parallel_size > 1
    is_eplb_enabled = parallel_config.enable_eplb
    is_mega_moe = moe_backend == "deep_gemm_mega_moe"
    is_nccl_based_eplb_communicator = parallel_config.eplb_config.communicator in (
        "torch_nccl",
        "pynccl",
    )

    # Override NCCL_MAX_CTAS to avoid hangs when EPLB's NCCL weight exchange
    # contends with MoE backend's cooperative-launch on GPU SMs.
    #
    # DeepGEMM Mega MoE uses cooperative launch, which tries to reserve a
    # large fraction of the GPU's SMs. If those SMs are occupied by NCCL,
    # the cooperative launch blocks until enough SMs are freed, causing a
    # deadlock. Limiting NCCL occupancy via NCCL_MAX_CTAS leaves space for
    # the cooperative kernel to launch and complete.
    if (
        is_data_parallel
        and is_eplb_enabled
        and is_nccl_based_eplb_communicator
        and is_mega_moe
    ):
        current_value_str = os.getenv("NCCL_MAX_CTAS")

        if current_value_str and current_value_str.isdigit():
            return

        override_value = 8
        os.environ["NCCL_MAX_CTAS"] = str(override_value)
        logger.info_once(
            f"EPLB: Setting NCCL_MAX_CTAS={override_value} "
            f"for expert parallel with NCCL-based EPLB communicator and "
            f"cooperative MoE backend (deep_gemm_mega_moe)",
            scope="global",
        )
