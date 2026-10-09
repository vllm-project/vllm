# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for experimental Hopper mHC dispatch; no CUDA kernels run here."""

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v41.nvidia.ops import mhc


@pytest.mark.parametrize(
    "capability,enabled,deep_gemm,hidden,hc,ubatching,expected",
    [
        (90, None, True, 5120, 4, False, False),
        (90, False, True, 5120, 4, False, False),
        (90, True, True, 5120, 4, False, True),
        (100, None, True, 5120, 4, False, True),
        (103, None, True, 5120, 4, False, True),
        (100, False, True, 5120, 4, False, False),
        (80, True, True, 5120, 4, False, False),
        (120, True, True, 5120, 4, False, False),
        (90, True, False, 5120, 4, False, False),
        (90, True, True, 5120, 4, True, False),
        (90, True, True, 5121, 4, False, False),
        (90, True, True, 5120, 3, False, False),
        (90, True, True, 5120, 6, False, False),
    ],
)
def test_overlap_opt_in_preserves_kernel_restrictions(
    monkeypatch, capability, enabled, deep_gemm, hidden, hc, ubatching, expected
):
    monkeypatch.setattr(
        mhc.current_platform,
        "is_device_capability_family",
        lambda family: capability // 10 == family // 10,
    )
    monkeypatch.setattr(
        mhc.current_platform, "is_device_capability", lambda cc: capability == cc
    )
    monkeypatch.setattr(mhc, "is_deep_gemm_supported", lambda: deep_gemm)
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(enable_mhc_overlap=enabled),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=hidden, hc_mult=hc)
        ),
        parallel_config=SimpleNamespace(use_ubatching=ubatching),
    )
    assert mhc.supports_mhc_overlap(config) is expected


def test_hopper_overlap_does_not_select_sm100_collective(monkeypatch):
    """Opting into SM90 coefficient overlap must not construct AllReduceMHC."""
    monkeypatch.setattr(
        mhc.current_platform, "is_device_capability_family", lambda _: False
    )
    # No TP group or model config is needed when the architecture declines.
    assert not mhc.supports_mhc_all_reduce(SimpleNamespace())
