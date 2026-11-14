# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for EPLB (Expert Parallel Load Balancing)."""

import contextlib
import os

import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def device_stream(stream: torch.Stream | None):
    """Platform-agnostic context manager that activates *stream* as the
    current accelerator stream for the duration of the `with` block.
    A no-op when *stream* is `None`."""
    if stream is None:
        yield
        return
    prev = torch.accelerator.current_stream()
    torch.accelerator.set_stream(stream)
    try:
        yield
    finally:
        torch.accelerator.set_stream(prev)


def override_envs_for_eplb(
    parallel_config: ParallelConfig,
    moe_backend: str | None = None,
) -> None:
    """Override environment variables for EPLB when specific conditions are met.

    Args:
        parallel_config: The parallel configuration object.
        moe_backend: The configured MoE backend (e.g. `deep_gemm_mega_moe`).

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
