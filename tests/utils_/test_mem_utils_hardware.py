# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-gated UMA (integrated-GPU) memory invariants.

Runs only on a real integrated GPU (e.g. the DGX Spark / GB10 CI lane) and is
skipped everywhere else. Guards the premise behind the UMA free-memory
correction used by ``get_device_memory_info`` and
``release_device_memory_under_pressure`` in ``vllm/utils/mem_utils.py``.
"""

import psutil
import pytest
import torch


def _on_integrated_gpu() -> bool:
    # Gate on the raw device property rather than vLLM's wrapper, so a regression
    # in ``current_platform.is_integrated_gpu`` makes the test FAIL below instead
    # of silently skipping.
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).is_integrated
        )
    except Exception:
        return False


@pytest.mark.skipif(
    not _on_integrated_gpu(), reason="integrated (UMA) GPU hardware only"
)
def test_integrated_gpu_free_memory_underreport():
    from vllm.platforms import current_platform

    # vLLM must classify this UMA device as integrated; otherwise the free-memory
    # correction silently no-ops and KV/cudagraph sizing can OOM (see #44740).
    assert current_platform.is_integrated_gpu(0) is True

    cuda_free = torch.accelerator.get_memory_info(0)[0]
    psutil_available = psutil.virtual_memory().available

    # On UMA, get_memory_info() reports only physically-free memory (== MemFree)
    # and excludes reclaimable OS memory, while psutil.available (== MemAvailable)
    # counts it. MemAvailable >= MemFree is a kernel invariant, so psutil is the
    # truer, never-smaller allocatable figure -- which is why the correction
    # exists. Assert the direction (deterministic); log the magnitude (varies).
    gap_gib = (psutil_available - cuda_free) / 1024**3
    print(
        f"UMA free-memory underreport: get_memory_info="
        f"{cuda_free / 1024**3:.1f} GiB vs psutil="
        f"{psutil_available / 1024**3:.1f} GiB (gap {gap_gib:.1f} GiB)"
    )
    assert psutil_available >= cuda_free, (cuda_free, psutil_available)
