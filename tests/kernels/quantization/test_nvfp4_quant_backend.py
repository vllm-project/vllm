# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``KernelConfig.nvfp4_input_quant_backend`` routing into the NVFP4 linear
kernels (FlashInfer CuTe-DSL activation quantization, default on Blackwell)."""

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
    under the default ("auto", when CuTe-DSL is available) and under an explicit
    "flashinfer_cutedsl"; "cuda" forces the built-in kernel. A non-FlashInfer
    kernel never routes through CuTe-DSL: it stays on the CUDA kernel under "auto"
    and raises for an explicit "flashinfer_cutedsl" (matching linear_backend's
    convention for an unsatisfiable explicit backend)."""
    layer_config = NvFp4LinearLayerConfig()

    with set_current_vllm_config(_config("flashinfer_cutedsl")):
        fi_kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
        assert fi_kernel.input_quant_backend == "flashinfer_cutedsl"

        # A non-FlashInfer NVFP4 linear kernel cannot honor the knob and raises.
        with pytest.raises(ValueError, match="does not route activation quant"):
            CutlassNvFp4LinearKernel(layer_config)

    with set_current_vllm_config(_config("auto")):
        # Default prefers CuTe-DSL for FlashInfer kernels when it is available.
        fi_kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
        assert fi_kernel.input_quant_backend == "flashinfer_cutedsl"
        # Non-FlashInfer kernels keep the built-in CUDA kernel under "auto".
        cutlass_kernel = CutlassNvFp4LinearKernel(layer_config)
        assert cutlass_kernel.input_quant_backend == "auto"

    with set_current_vllm_config(_config("cuda")):
        # Explicit opt-out keeps the built-in CUDA kernel even for FlashInfer.
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
    """Guards the fused-MLP path: whenever the CUTLASS NVFP4 kernel resolves to
    CuTe-DSL (the default "auto" when available, or an explicit "flashinfer_cutedsl")
    it must not advertise a pre-quantized input key, or a producer would
    pre-quantize with the C++ kernel and bypass cute-dsl. So input_quant_key() is
    None there and kNvfp4Dynamic only under an explicit "cuda"."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
    )

    layer_config = NvFp4LinearLayerConfig()

    for backend in ("auto", "flashinfer_cutedsl"):
        with set_current_vllm_config(_config(backend)):
            kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
            assert kernel.input_quant_key() is None

    with set_current_vllm_config(_config("cuda")):
        kernel = FlashInferCutlassNvFp4LinearKernel(layer_config)
        assert kernel.input_quant_key() == kNvfp4Dynamic
