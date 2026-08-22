# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 E8M0 kernel selection logic (CPU-only)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x import (
    B12xFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8BlockScaledMMKernel,
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.humming import (
    HummingFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockE8M0Sym,
)

pytestmark = pytest.mark.cpu_test


_E8M0_BACKEND_SUPPORT = [
    (FlashInferFp8DeepGEMMDynamicBlockScaledKernel, "cuda", True),
    (DeepGemmFp8BlockScaledMMKernel, "cuda", True),
    (CutlassFp8BlockScaledMMKernel, "cuda", False),
    (B12xFp8BlockScaledMMKernel, "cuda", True),
    (MarlinFP8ScaledMMLinearKernel, "cuda", True),
    (HummingFP8ScaledMMLinearKernel, "cuda", True),
    (TritonFp8BlockScaledMMKernel, "cuda", False),
    (BlockWiseTorchFP8ScaledMMLinearKernel, "cuda", False),
    (AiterFp8BlockScaledMMKernel, "rocm", True),
    (TritonFp8BlockScaledMMKernel, "rocm", True),
    (CPUFp8BlockScaledMMKernel, "cpu", False),
    (XPUFp8BlockScaledMMKernel, "xpu", False),
    (TritonFp8BlockScaledMMKernel, "xpu", True),
    (BlockWiseTorchFP8ScaledMMLinearKernel, "xpu", False),
]


@pytest.fixture
def e8m0_block_config() -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockE8M0Sym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(128, 128),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("kernel_cls,platform,e8m0_supported", _E8M0_BACKEND_SUPPORT)
def test_fp8_block_backends_e8m0_support(
    kernel_cls, platform, e8m0_supported, e8m0_block_config, monkeypatch
):
    """Every registered block-FP8 backend declares its E8M0 compatibility."""
    if kernel_cls is FlashInferFp8DeepGEMMDynamicBlockScaledKernel:
        monkeypatch.setattr(
            FlashInferFp8BlockScaledMMKernel,
            "can_implement",
            classmethod(lambda cls, config: (True, None)),
        )
        monkeypatch.setattr(
            DeepGemmFp8BlockScaledMMKernel,
            "can_implement",
            classmethod(lambda cls, config: (True, None)),
        )
    elif kernel_cls is DeepGemmFp8BlockScaledMMKernel:
        monkeypatch.setattr(
            "vllm.model_executor.kernels.linear.scaled_mm.deep_gemm."
            "get_current_vllm_config",
            lambda: SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type="test")
                )
            ),
        )
        monkeypatch.setattr(
            "vllm.model_executor.kernels.linear.scaled_mm.deep_gemm."
            "should_auto_disable_deep_gemm",
            lambda model_type: False,
        )
        monkeypatch.setattr(
            "vllm.model_executor.kernels.linear.scaled_mm.deep_gemm."
            "should_use_deepgemm_for_fp8_linear",
            lambda out_dtype, weight_shape: True,
        )
    if kernel_cls is TritonFp8BlockScaledMMKernel:
        monkeypatch.setattr(
            "vllm.model_executor.kernels.linear.scaled_mm.triton."
            "current_platform.is_cuda",
            lambda: platform == "cuda",
        )
    can_implement, reason = kernel_cls.can_implement(e8m0_block_config)

    assert can_implement is e8m0_supported, reason
