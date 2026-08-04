# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``KernelConfig.nvfp4_input_quant_backend`` routing into the NVFP4 linear
kernels (opt-in FlashInfer CuTe-DSL activation quantization)."""

import pytest

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.cutlass import CutlassNvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer,
    has_flashinfer_cutedsl_nvfp4_quant,
)

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 requires compute capability of 10 or above.",
        allow_module_level=True,
    )


def _config(nvfp4_input_quant_backend: str) -> VllmConfig:
    return VllmConfig(
        kernel_config=KernelConfig(nvfp4_input_quant_backend=nvfp4_input_quant_backend)
    )


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer CuTe-DSL NVFP4 quantization is not available.",
)
def test_nvfp4_input_quant_backend_routes_only_flashinfer_kernels() -> None:
    """A FlashInfer NVFP4 linear kernel resolves its activation quant to CuTe-DSL
    when ``nvfp4_input_quant_backend=flashinfer_cutedsl``. A non-FlashInfer kernel
    cannot honor the request and raises (matching linear_backend's convention for
    an unsatisfiable explicit backend). Under the default ("auto"), no kernel opts
    in."""
    layer_config = NvFp4LinearLayerConfig()

    with set_current_vllm_config(_config("flashinfer_cutedsl")):
        fi_kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
        assert fi_kernel.input_quant_backend == "flashinfer_cutedsl"

        # A non-FlashInfer NVFP4 linear kernel cannot honor the knob and raises.
        with pytest.raises(ValueError, match="does not route activation quant"):
            CutlassNvFp4LinearKernel(layer_config)

    with set_current_vllm_config(_config("auto")):
        fi_kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
        assert fi_kernel.input_quant_backend == "auto"


@pytest.mark.skipif(
    not has_flashinfer(),
    reason="FlashInfer (needed to construct the kernel) is not available.",
)
def test_nvfp4_input_quant_backend_requires_flashinfer(monkeypatch) -> None:
    """Selecting flashinfer_cutedsl for a FlashInfer kernel without CuTe-DSL
    nvfp4_quantize available fails at kernel setup. Gated on has_flashinfer()
    (not the CuTe-DSL check) so it also runs where CuTe-DSL is genuinely
    unavailable; the monkeypatch forces the unavailable branch either way."""
    import vllm.model_executor.kernels.linear.nvfp4.base as nvfp4_base

    # base.py imports has_flashinfer_cutedsl_nvfp4_quant at module scope, so patch
    # it there (where the resolver looks it up), not on vllm.utils.flashinfer.
    monkeypatch.setattr(nvfp4_base, "has_flashinfer_cutedsl_nvfp4_quant", lambda: False)
    layer_config = NvFp4LinearLayerConfig()

    with (
        set_current_vllm_config(_config("flashinfer_cutedsl")),
        pytest.raises(ValueError, match="requires SM100"),
    ):
        FlashInferCutlassNvFp4LinearKernel(layer_config)


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer CuTe-DSL NVFP4 quantization is not available.",
)
def test_flashinfer_cutedsl_disables_input_prequant() -> None:
    """Guards the fused-MLP path: under flashinfer_cutedsl the CUTLASS NVFP4
    kernel must not advertise a pre-quantized input key (a producer would
    pre-quantize with the C++ kernel and bypass cute-dsl). So input_quant_key()
    is None under flashinfer_cutedsl and kNvfp4Dynamic under the default."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
    )

    layer_config = NvFp4LinearLayerConfig()

    with set_current_vllm_config(_config("flashinfer_cutedsl")):
        kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
        assert kernel.input_quant_key() is None

    with set_current_vllm_config(_config("auto")):
        kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
        assert kernel.input_quant_key() == kNvfp4Dynamic
