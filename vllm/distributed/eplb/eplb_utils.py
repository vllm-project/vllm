# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for EPLB (Expert Parallel Load Balancing)."""

import os

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


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
