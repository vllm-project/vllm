# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.cache import CacheConfig
from vllm.platforms import current_platform

from ..utils import (
    compare_two_settings,
    create_new_process_for_each_test,
    wait_for_memory_to_settle,
)


@pytest.mark.parametrize(
    "PP_SIZE, MODEL_NAME",
    [
        (2, "JackFram/llama-160m"),
    ],
)
@pytest.mark.parametrize(
    "ATTN_BACKEND",
    [None] if current_platform.is_rocm() else ["FLASH_ATTN"],
)
@create_new_process_for_each_test()
def test_pp_cudagraph(
    PP_SIZE: int,
    MODEL_NAME: str,
    ATTN_BACKEND: str | None,
):
    if current_platform.is_rocm():
        # ROCm CI can hand this test a device that is still lazily
        # releasing VRAM from a previous process; wait before starting.
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - CacheConfig.gpu_memory_utilization
        )

    cudagraph_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--distributed-executor-backend",
        "mp",
    ]
    if current_platform.is_rocm():
        # MI250 reports ~2.7 GiB in use per idle GPU, which leaves too little
        # headroom for the default 0.92 utilization once workers start.
        cudagraph_args += ["--gpu-memory-utilization", "0.85"]
    # On ROCm, defer to the platform attention selector instead of forcing a backend.
    if ATTN_BACKEND is not None:
        cudagraph_args.append(f"--attention-backend={ATTN_BACKEND}")

    eager_args = cudagraph_args + ["--enforce-eager"]

    compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)
