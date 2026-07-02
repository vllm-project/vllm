# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for ROCm hipMemGetInfo over-reporting total VRAM.

On multi-GPU ROCm systems with XGMI/HIVE peer access, hipMemGetInfo may
return the entire accessible VRAM pool as ``total`` (e.g. 6 × 32 GiB = 192
GiB for GPU 0 on a 6-GPU box).  ROCmPlatform.mem_get_info() must cap that to
the physical local-device VRAM so vLLM does not attempt to allocate more
memory than the local GPU actually has.

See https://github.com/vllm-project/vllm/issues/36890
"""
import pytest
from unittest.mock import patch

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-specific mem_get_info cap test",
)
def test_rocm_mem_get_info_caps_total_to_local_vram():
    """mem_get_info must return local VRAM as total, not the XGMI pool."""
    # Simulate hipMemGetInfo returning a 6× inflated total (192 GiB) while
    # the physical VRAM is 32 GiB.
    VRAM_32GIB = 32 * (1 << 30)
    POOL_192GIB = 6 * VRAM_32GIB
    FREE_20GIB = 20 * (1 << 30)

    from vllm.platforms.rocm import ROCmPlatform

    with (
        patch("torch.cuda.mem_get_info", return_value=(FREE_20GIB, POOL_192GIB)),
        patch.object(
            ROCmPlatform,
            "get_device_total_memory",
            return_value=VRAM_32GIB,
        ),
    ):
        free, total = ROCmPlatform.mem_get_info(0)

    assert total == VRAM_32GIB, (
        f"total should be capped to local VRAM ({VRAM_32GIB // (1 << 30)} GiB), "
        f"got {total // (1 << 30)} GiB"
    )
    assert free <= total, "free must not exceed total after capping"


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-specific mem_get_info cap test",
)
def test_rocm_mem_get_info_passthrough_when_not_inflated():
    """mem_get_info must not modify values when total already equals local VRAM."""
    VRAM_32GIB = 32 * (1 << 30)
    FREE_20GIB = 20 * (1 << 30)

    from vllm.platforms.rocm import ROCmPlatform

    with (
        patch("torch.cuda.mem_get_info", return_value=(FREE_20GIB, VRAM_32GIB)),
        patch.object(
            ROCmPlatform,
            "get_device_total_memory",
            return_value=VRAM_32GIB,
        ),
    ):
        free, total = ROCmPlatform.mem_get_info(0)

    assert total == VRAM_32GIB
    assert free == FREE_20GIB
