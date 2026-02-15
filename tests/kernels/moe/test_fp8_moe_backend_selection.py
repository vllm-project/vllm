# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for FP8 MoE backend selection priority ordering.

Validates that the backend selection logic in `select_fp8_moe_backend`
correctly prioritizes backends based on GPU architecture:
- Blackwell (SM100+): FlashInfer > DeepGEMM
- Hopper (SM90) and older: DeepGEMM > FlashInfer
"""

from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)


def _make_test_config() -> FusedMoEConfig:
    """Create a minimal FusedMoEConfig for testing backend selection."""
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=4096,
        intermediate_size_per_partition=14336,
        num_local_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation="silu",
        in_dtype="bfloat16",
        device="cuda",
        routing_method=RoutingMethodType.TopK,
    )


def _mock_supported(backend_name: Fp8MoeBackend):
    """
    Create a mock kernel class whose is_supported_config returns True
    only for the given backend.
    """

    def _factory(backend: Fp8MoeBackend):
        mock_cls = MagicMock()
        if backend == backend_name:
            mock_cls.is_supported_config.return_value = (True, None)
        else:
            mock_cls.is_supported_config.return_value = (
                False,
                f"{backend.value} not supported in test",
            )
        return mock_cls

    return _factory


@pytest.mark.parametrize(
    "device_cap, expected_backend",
    [
        # On Blackwell (SM100+), FlashInfer CUTLASS should be preferred
        (100, Fp8MoeBackend.FLASHINFER_CUTLASS),
        # On Hopper (SM90), DeepGEMM should be preferred
        (90, Fp8MoeBackend.DEEPGEMM),
        # On Ada (SM89), DeepGEMM should be preferred
        (89, Fp8MoeBackend.DEEPGEMM),
    ],
)
def test_fp8_moe_backend_priority_by_arch(device_cap, expected_backend):
    """
    Test that when both FLASHINFER_CUTLASS and DEEPGEMM are supported,
    the correct one is selected based on GPU architecture.
    """
    config = _make_test_config()

    # Both FLASHINFER_CUTLASS and DEEPGEMM report as supported
    supported_backends = {
        Fp8MoeBackend.FLASHINFER_CUTLASS,
        Fp8MoeBackend.DEEPGEMM,
    }

    def mock_backend_to_kernel_cls(backend):
        mock_cls = MagicMock()
        if backend in supported_backends:
            mock_cls.is_supported_config.return_value = (True, None)
        else:
            mock_cls.is_supported_config.return_value = (
                False,
                f"{backend.value} not supported in test",
            )
        return mock_cls

    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8"
            ".current_platform.has_device_capability",
            side_effect=lambda cap: device_cap >= cap,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform.is_rocm",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8"
            ".is_supported_config_trtllm_fp8",
            return_value=(False, "trtllm not supported in test"),
        ),
        patch("vllm.model_executor.layers.fused_moe.oracle.fp8.envs") as mock_envs,
    ):
        # No env vars explicitly set
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_TEST_FORCE_FP8_MARLIN = False

        backend, _ = select_fp8_moe_backend(
            config=config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )

        assert backend == expected_backend, (
            f"Expected {expected_backend.value} on SM{device_cap}, got {backend.value}"
        )


@pytest.mark.parametrize(
    "device_cap",
    [90, 100],
)
def test_fp8_moe_backend_deepgemm_only(device_cap):
    """
    Test that DeepGEMM is selected on any arch when FlashInfer is
    not supported (e.g., not installed).
    """
    config = _make_test_config()

    def mock_backend_to_kernel_cls(backend):
        mock_cls = MagicMock()
        if backend == Fp8MoeBackend.DEEPGEMM:
            mock_cls.is_supported_config.return_value = (True, None)
        else:
            mock_cls.is_supported_config.return_value = (
                False,
                f"{backend.value} not supported in test",
            )
        return mock_cls

    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8"
            ".current_platform.has_device_capability",
            side_effect=lambda cap: device_cap >= cap,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform.is_rocm",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8"
            ".is_supported_config_trtllm_fp8",
            return_value=(False, "trtllm not supported in test"),
        ),
        patch("vllm.model_executor.layers.fused_moe.oracle.fp8.envs") as mock_envs,
    ):
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_TEST_FORCE_FP8_MARLIN = False

        backend, _ = select_fp8_moe_backend(
            config=config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )

        assert backend == Fp8MoeBackend.DEEPGEMM, (
            f"Expected DEEPGEMM when only DeepGEMM is supported on "
            f"SM{device_cap}, got {backend.value}"
        )


@pytest.mark.parametrize(
    "device_cap",
    [90, 100],
)
def test_fp8_moe_backend_flashinfer_only(device_cap):
    """
    Test that FLASHINFER_CUTLASS is selected on any arch when DeepGEMM
    is not supported.
    """
    config = _make_test_config()

    def mock_backend_to_kernel_cls(backend):
        mock_cls = MagicMock()
        if backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
            mock_cls.is_supported_config.return_value = (True, None)
        else:
            mock_cls.is_supported_config.return_value = (
                False,
                f"{backend.value} not supported in test",
            )
        return mock_cls

    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8"
            ".current_platform.has_device_capability",
            side_effect=lambda cap: device_cap >= cap,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform.is_rocm",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8"
            ".is_supported_config_trtllm_fp8",
            return_value=(False, "trtllm not supported in test"),
        ),
        patch("vllm.model_executor.layers.fused_moe.oracle.fp8.envs") as mock_envs,
    ):
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_TEST_FORCE_FP8_MARLIN = False

        backend, _ = select_fp8_moe_backend(
            config=config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )

        assert backend == Fp8MoeBackend.FLASHINFER_CUTLASS, (
            f"Expected FLASHINFER_CUTLASS when only FlashInfer is supported "
            f"on SM{device_cap}, got {backend.value}"
        )


def test_fp8_moe_backend_triton_fallback_when_lora_enabled():
    """
    Test that TRITON is always selected when LoRA is enabled,
    regardless of GPU architecture.
    """
    config = _make_test_config()
    # Enable LoRA
    config.is_lora_enabled = True

    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.fp8.backend_to_kernel_cls",
    ) as mock_backend_to_cls:
        mock_cls = MagicMock()
        mock_backend_to_cls.return_value = mock_cls

        backend, _ = select_fp8_moe_backend(
            config=config,
            weight_key=kFp8Static128BlockSym,
            activation_key=kFp8Dynamic128Sym,
        )

        assert backend == Fp8MoeBackend.TRITON, (
            f"Expected TRITON when LoRA is enabled, got {backend.value}"
        )
