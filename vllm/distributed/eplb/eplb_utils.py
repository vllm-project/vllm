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

    # Override NCCL_MAX_CTAS to overcome hangs when using async EPLB with
    # DeepEP low latency mode. We reduce the number of SMs used by NCCL so
    # that weight exchange in async EPLB doesn't block the cooperative launch
    # of DeepEP LL kernels.
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
