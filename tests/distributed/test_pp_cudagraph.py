# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest
import torch

from vllm.platforms import current_platform

from ..utils import compare_two_settings, create_new_process_for_each_test


def _wait_for_idle_rocm_gpus(
    num_devices: int, timeout: float = 60.0, poll_interval: float = 5.0
) -> None:
    """Give a just-exited process's VRAM a moment to clear on a shared
    ROCm CI host before starting a server, so a stale reading doesn't
    trip vLLM's startup free-memory check. Not a hard gate -- proceeds
    after ``timeout`` either way.
    """
    headroom_bytes = 4 * 1024**3
    deadline = time.time() + timeout
    while time.time() < deadline:
        busy = any(
            total - free > headroom_bytes
            for free, total in (
                torch.accelerator.get_memory_info(i) for i in range(num_devices)
            )
        )
        if not busy:
            return
        time.sleep(poll_interval)
    print(f"[GPU memory preflight] Still busy after {timeout:.0f}s; proceeding.")


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
        _wait_for_idle_rocm_gpus(num_devices=PP_SIZE)

    cudagraph_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--distributed-executor-backend",
        "mp",
    ]
    # On ROCm, defer to the platform attention selector instead of forcing a backend.
    if ATTN_BACKEND is not None:
        cudagraph_args.append(f"--attention-backend={ATTN_BACKEND}")

    eager_args = cudagraph_args + ["--enforce-eager"]

    compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)
