# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 linear kernel auto-selection.

On SM 12x the CUTLASS-based NVFP4 kernels produce incorrect output
(https://github.com/vllm-project/vllm/issues/48898) and FlashInfer's JIT can
fail to target consumer 12x CUDA toolchains at engine startup. Auto-selection
must prefer Marlin there while both kernels remain reachable via an explicit
``--linear-backend`` opt-in.
"""

import pytest

import vllm.model_executor.kernels.linear as linear_kernels
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
    CutlassNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernel selection"
)


@pytest.fixture
def nvfp4_kernels_supported(monkeypatch):
    """Pretend every CUTLASS/Marlin NVFP4 kernel is supported so selection is
    decided purely by the priority list and the SM 12x exclusion."""

    def supported(cls, compute_capability=None):
        return True, None

    def implementable(cls, config):
        return True, None

    for kernel in (
        FlashInferCutlassNvFp4LinearKernel,
        CutlassNvFp4LinearKernel,
        MarlinNvFp4LinearKernel,
    ):
        monkeypatch.setattr(kernel, "is_supported", classmethod(supported))
        monkeypatch.setattr(kernel, "can_implement", classmethod(implementable))
    monkeypatch.setattr(
        FlashInferCuteDslNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (False, "requires sm_10x")),
    )


def _set_family(monkeypatch, family: int | None):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == family,
    )


def test_auto_selection_prefers_marlin_on_sm12x(monkeypatch, nvfp4_kernels_supported):
    _set_family(monkeypatch, 120)
    kernel = init_nvfp4_linear_kernel()
    assert isinstance(kernel, MarlinNvFp4LinearKernel)


def test_auto_selection_unchanged_off_sm12x(monkeypatch, nvfp4_kernels_supported):
    _set_family(monkeypatch, 100)
    kernel = init_nvfp4_linear_kernel()
    assert isinstance(kernel, FlashInferCutlassNvFp4LinearKernel)


def test_explicit_backend_opt_in_preserved_on_sm12x(
    monkeypatch, nvfp4_kernels_supported
):
    """--linear-backend flashinfer_cutlass must still work on SM 12x for
    debugging and evidence-gathering (as used in #48898)."""
    _set_family(monkeypatch, 120)
    monkeypatch.setattr(
        linear_kernels, "_get_linear_backend", lambda: "flashinfer_cutlass"
    )
    kernel = init_nvfp4_linear_kernel()
    assert isinstance(kernel, FlashInferCutlassNvFp4LinearKernel)
