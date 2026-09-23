# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from vllm.platforms import current_platform


@pytest.fixture()
def gpu_memory_cleared():
    """Wait for GPU memory to settle before a test.

    On ROCm with NPS2 GPU partitioning (DPX mode, e.g. MI355/gfx950), when
    multiple test pods run on the same physical node simultaneously, a prior
    test can leave residual GPU memory that prevents the next test from
    initializing. Tests that need a clean GPU state should request this
    fixture explicitly rather than relying on autouse.
    """
    if current_platform.is_rocm() and "gfx950" in (
        current_platform.get_device_name() or ""
    ):
        wait_for_gpu_memory_to_clear(
            devices=[0],
            threshold_ratio=0.08,
            timeout_s=30,
            stable_duration_s=1,
        )
    yield
