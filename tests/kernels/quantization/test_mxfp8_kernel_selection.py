# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 linear kernel selection logic (CPU-only)

Run `pytest tests/kernels/quantization/test_mxfp8_kernel_selection.py`.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    MarlinMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test

# Kernels backed by FlashInfer mm_mxfp8, which requires N, K >= 128 and
# K % 32 == 0.
_MM_MXFP8_KERNELS = [
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
]

_SUPPORTED_SHAPE = (4096, 4096)
# (N, K): N < 128, K < 128, K % 32 != 0.
_UNSUPPORTED_SHAPES = [(64, 4096), (4096, 64), (4096, 4112)]

# (N, K) with N < 128 or K < 128, which only Marlin runs.
_MARLIN_ONLY_SHAPES = [(32, 256), (64, 256), (96, 256), (256, 64)]
# (N, K): N <= 0, K <= 0, K % 32 != 0.
_MARLIN_UNSUPPORTED_SHAPES = [(0, 256), (256, 0), (256, 100)]

_SELECTION_CASES = [
    (100, (128, 128), FlashInferCutedslMxfp8LinearKernel),
    (100, (130, 256), FlashInferCutedslMxfp8LinearKernel),
    (103, (128, 128), FlashInferCutedslMxfp8LinearKernel),
    (103, (130, 256), FlashInferCutedslMxfp8LinearKernel),
    (120, (128, 128), FlashInferCutlassMxfp8LinearKernel),
    (120, (130, 256), MarlinMxfp8LinearKernel),
    (121, (128, 128), FlashInferCutlassMxfp8LinearKernel),
    (121, (130, 256), MarlinMxfp8LinearKernel),
    *[
        (capability, shape, MarlinMxfp8LinearKernel)
        for capability in (100, 103, 120, 121)
        for shape in _MARLIN_ONLY_SHAPES
    ],
]


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
def test_mm_mxfp8_kernels_accept_supported_shape(kernel_cls):
    config = Mxfp8LinearLayerConfig(weight_shape=_SUPPORTED_SHAPE)
    can_implement, reason = kernel_cls.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
@pytest.mark.parametrize("weight_shape", _UNSUPPORTED_SHAPES)
def test_mm_mxfp8_kernels_reject_unsupported_shape(kernel_cls, weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize("weight_shape", _MARLIN_UNSUPPORTED_SHAPES)
def test_marlin_mxfp8_rejects_unsupported_shape(weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = MarlinMxfp8LinearKernel.can_implement(config)
    assert not can_implement
    assert reason


@contextmanager
def _patch_cuda_platform(capability: int):
    """Patch the platform so kernel selection runs as on a CUDA GPU with the
    given compute capability. The FlashInfer and Marlin kernels report as
    supported, CuTe-DSL only on SM100/103."""
    with (
        patch("vllm.model_executor.kernels.linear.current_platform") as platform,
        patch(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer.current_platform."
            "is_device_capability_family",
            side_effect=lambda cap, device_id=0: cap // 10 == capability // 10,
        ),
        patch.object(
            FlashInferCutedslMxfp8LinearKernel,
            "is_supported",
            return_value=(capability in (100, 103), None),
        ),
        patch.object(
            FlashInferCutlassMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        patch.object(
            MarlinMxfp8LinearKernel, "is_supported", return_value=(True, None)
        ),
    ):
        platform._enum = PlatformEnum.CUDA
        yield


@pytest.mark.parametrize(
    ("capability", "expected"),
    [(100, True), (103, True), (120, False), (121, False)],
)
def test_cutlass_mxfp8_requires_aligned_n_only_on_sm12x(capability, expected):
    """(130, 256) passes the generic mm_mxfp8 limits. Only the SM12x CUTLASS
    backend needs N % 32 == 0, so SM100/103 must keep accepting it."""
    config = Mxfp8LinearLayerConfig(weight_shape=(130, 256))
    with _patch_cuda_platform(capability):
        can_implement, _ = FlashInferCutlassMxfp8LinearKernel.can_implement(config)
    assert can_implement == expected


@pytest.mark.parametrize(
    ("capability", "weight_shape", "expected_kernel_cls"), _SELECTION_CASES
)
def test_init_mxfp8_linear_kernel(capability, weight_shape, expected_kernel_cls):
    """A layer that mm_mxfp8 cannot handle must fall through to the next
    kernel in the CUDA priority list instead of being selected and failing
    in apply_weights."""
    with _patch_cuda_platform(capability):
        kernel = init_mxfp8_linear_kernel(weight_shape=weight_shape)

    assert isinstance(kernel, expected_kernel_cls)
    assert kernel.config == Mxfp8LinearLayerConfig(weight_shape=weight_shape)


_AITER_FLYDSL_TUNED_SHAPE = (2560, 6144)
_AITER_FLYDSL_UNTUNED_SHAPE = (4992, 6144)


@contextmanager
def _patch_rocm_gfx950(linear_backend: str = "auto"):
    """Patch the platform so kernel selection runs as on gfx950 with AITER
    linear enabled and AITER MXFP8 configs tuned for `_AITER_FLYDSL_TUNED_SHAPE`."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.kernels.linear import (
        AiterFlyDSLMxfp8LinearKernel,
        RocmDotScaledMxfp8LinearKernel,
    )

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            patch("vllm.model_executor.kernels.linear.current_platform") as platform,
            patch(
                "vllm.model_executor.kernels.linear._get_linear_backend",
                return_value=linear_backend,
            ),
            patch(
                "vllm.model_executor.kernels.linear.mxfp8.aiter_flydsl.current_platform"
            ) as aiter_platform,
            patch.object(rocm_aiter_ops, "is_linear_enabled", return_value=True),
            patch.object(
                rocm_aiter_ops,
                "is_mxfp8_bpreshuffle_gemm_tuned",
                side_effect=lambda n, k: (n, k) == _AITER_FLYDSL_TUNED_SHAPE,
            ),
            patch.object(
                RocmDotScaledMxfp8LinearKernel,
                "is_supported",
                return_value=(True, None),
            ),
        ):
            platform._enum = PlatformEnum.ROCM
            aiter_platform.is_rocm.return_value = True
            aiter_platform.supports_mx.return_value = True
            yield AiterFlyDSLMxfp8LinearKernel, RocmDotScaledMxfp8LinearKernel
    finally:
        torch.set_default_dtype(prev_dtype)


@pytest.mark.parametrize("linear_backend", ["auto", "aiter"])
def test_rocm_mxfp8_aiter_flydsl_claims_only_tuned_shapes(monkeypatch, linear_backend):
    """AITER takes the tuned (N, K); an untuned one falls through to the
    Triton kernel, including under --linear-backend=aiter."""
    with _patch_rocm_gfx950(linear_backend) as (aiter_cls, triton_cls):
        tuned = init_mxfp8_linear_kernel(weight_shape=_AITER_FLYDSL_TUNED_SHAPE)
        untuned = init_mxfp8_linear_kernel(weight_shape=_AITER_FLYDSL_UNTUNED_SHAPE)
    assert type(tuned) is aiter_cls
    assert type(untuned) is triton_cls


def test_rocm_mxfp8_aiter_flydsl_off_under_batch_invariance(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with _patch_rocm_gfx950() as (_, triton_cls):
        kernel = init_mxfp8_linear_kernel(weight_shape=_AITER_FLYDSL_TUNED_SHAPE)
    assert type(kernel) is triton_cls


def test_rocm_mxfp8_aiter_flydsl_rejects_non_bf16_and_bmm(monkeypatch):
    with _patch_rocm_gfx950() as (aiter_cls, triton_cls):
        bmm = Mxfp8LinearLayerConfig(
            weight_shape=_AITER_FLYDSL_TUNED_SHAPE, bmm_batch_size=2
        )
        assert not aiter_cls.can_implement(bmm)[0]
        torch.set_default_dtype(torch.float16)
        kernel = init_mxfp8_linear_kernel(weight_shape=_AITER_FLYDSL_TUNED_SHAPE)
    assert type(kernel) is triton_cls


def test_rocm_mxfp8_aiter_flydsl_tuned_lookup_failure_is_untuned():
    """A failing AITER tuned-config lookup reports the shape as untuned
    instead of failing kernel selection."""
    import sys

    from vllm._aiter_ops import rocm_aiter_ops

    fake = MagicMock()
    fake.get_mxscale_bpreshuffle_config.side_effect = KeyError("gfx")
    lookup = rocm_aiter_ops.is_mxfp8_bpreshuffle_gemm_tuned.__wrapped__
    with (
        patch.dict(sys.modules, {"aiter.ops.gemm_op_a8w8": fake}),
        patch("vllm._aiter_ops.current_platform") as platform,
    ):
        platform.is_rocm.return_value = True
        assert lookup(*_AITER_FLYDSL_TUNED_SHAPE) is False
