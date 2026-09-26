# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free capability gate for the CUTLASS int8 w8a8 kernel (#54311).

CUTLASS int8 is only wired for SM < 100; the SM100/SM120 dispatchers pass a
nullptr int8 func and the C++ hard-errors. ``is_supported`` must decline on
SM100+ so kernel selection can reach the Triton int8 fallback. These assert the
gate directly by mocking the platform predicate, no GPU needed.
"""

import pytest

import vllm.model_executor.kernels.linear.scaled_mm.cutlass as cutlass_mod
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
)


@pytest.fixture(autouse=True)
def _force_cuda(monkeypatch):
    # The first check is is_cuda(); force it True so the compute-capability
    # gate is the only thing under test (CI runners are CPU-only).
    monkeypatch.setattr(cutlass_mod.current_platform, "is_cuda", lambda: True)


@pytest.mark.parametrize("cc", [121, 120, 100])
def test_cutlass_int8_declines_on_sm100_plus(cc):
    # sm_100 (B200), sm_120 (RTX 5090), sm_121 (GB10 / DGX Spark).
    supported, reason = CutlassInt8ScaledMMLinearKernel.is_supported(cc)
    assert supported is False
    assert reason is not None


@pytest.mark.parametrize("cc", [90, 89, 80, 75])
def test_cutlass_int8_supported_below_sm100(cc):
    supported, reason = CutlassInt8ScaledMMLinearKernel.is_supported(cc)
    assert supported is True
    assert reason is None


def test_cutlass_int8_supported_when_capability_unknown():
    # compute_capability=None must not spuriously decline (back-compat).
    supported, _ = CutlassInt8ScaledMMLinearKernel.is_supported(None)
    assert supported is True
