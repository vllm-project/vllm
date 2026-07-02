# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    AiterExperts,
)
from vllm.platforms import current_platform

skipif_not_cuda_rocm = pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_rocm()),
    reason="Only supported on CUDA/ROCm platforms.",
)


@pytest.mark.parametrize(
    "platform_method,expected_backend",
    [
        ("is_cuda", UnquantizedMoeBackend.TRITON),  # Default CUDA without FlashInfer
        ("is_rocm", UnquantizedMoeBackend.TRITON),  # ROCm without AITER
        ("is_cpu", UnquantizedMoeBackend.CPU),
        ("is_xpu", UnquantizedMoeBackend.XPU),
        ("is_tpu", UnquantizedMoeBackend.TPU),
        ("is_out_of_tree", UnquantizedMoeBackend.OOT),
    ],
)
@patch(
    "vllm.utils.flashinfer.has_flashinfer",
    return_value=False,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.rocm_aiter_ops.is_fused_moe_enabled",
    return_value=False,
)
def test_select_default_backend_by_platform(
    mock_aiter_enabled,
    mock_has_flashinfer,
    monkeypatch,
    platform_method,
    expected_backend,
):
    """Test default backend selection per platform with all optional
    accelerators (FlashInfer, AITER) disabled."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        # Set all platform checks to False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_rocm.return_value = False
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        # Set only the specified platform to True
        getattr(mock_platform, platform_method).return_value = True

    with (
        patch.object(current_platform, "is_cuda", return_value=False),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(current_platform, "is_tpu", return_value=False),
        patch.object(current_platform, "is_out_of_tree", return_value=False),
        patch.object(current_platform, platform_method, return_value=True),
    ):
        moe_config = make_dummy_moe_config()
        selected_backend, expert_cls = select_unquantized_moe_backend(
            moe_config=moe_config
        )

        assert selected_backend == expected_backend
        if expected_backend in [
            UnquantizedMoeBackend.CPU,
            UnquantizedMoeBackend.OOT,
            UnquantizedMoeBackend.TPU,
        ]:
            assert expert_cls is None
        else:
            assert expert_cls is not None


@patch(
    "vllm.utils.flashinfer.has_flashinfer",
    return_value=False,
)
@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.rocm_aiter_ops.is_fused_moe_enabled",
    return_value=True,
)
@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific backend selection test"
)
def test_select_rocm_aiter_backend(mock_aiter_enabled, mock_has_flashinfer):
    """Test ROCm backend selection when AITER is available."""
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda.return_value = False
        mock_platform.is_rocm.return_value = True
        mock_platform.is_cpu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        moe_config = make_dummy_moe_config()
        selected_backend, expert_cls = select_unquantized_moe_backend(
            moe_config=moe_config,
        )

        assert selected_backend == UnquantizedMoeBackend.AITER
        assert expert_cls is not None


@patch(
    "vllm.model_executor.layers.fused_moe.oracle.unquantized.rocm_aiter_ops.is_fused_moe_enabled",
    return_value=True,
)
def test_select_rocm_swiglustep_falls_back_from_explicit_aiter(
    mock_aiter_enabled,
    monkeypatch,
):
    """AITER has no native SWIGLUSTEP activation, so ROCm should use Triton."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")

    with (
        patch.object(current_platform, "is_cuda", return_value=False),
        patch.object(current_platform, "is_rocm", return_value=True),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(current_platform, "is_tpu", return_value=False),
        patch.object(current_platform, "is_out_of_tree", return_value=False),
        patch.object(current_platform, "is_cuda_alike", return_value=True),
    ):
        moe_config = make_dummy_moe_config()
        moe_config.activation = MoEActivation.SWIGLUSTEP

        selected_backend, experts_cls = select_unquantized_moe_backend(
            moe_config=moe_config,
        )

        assert selected_backend == UnquantizedMoeBackend.TRITON
        assert experts_cls is not None


def test_aiter_does_not_claim_swiglustep_support():
    assert not AiterExperts._supports_activation(MoEActivation.SWIGLUSTEP)


@patch(
    "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.TrtLlmBf16Experts.is_supported_config",
    return_value=(True, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_trtllm_backend(mock_is_supported_trtllm):
    """Test CUDA backend selection when FlashInfer TRTLLM is available and enabled."""
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(current_platform, "is_tpu", return_value=False),
        patch.object(current_platform, "is_out_of_tree", return_value=False),
        patch.object(current_platform, "has_device_capability", return_value=True),
    ):
        moe_config = make_dummy_moe_config()
        moe_config.moe_backend = "flashinfer_trtllm"
        # TRTLLM requires EP and does not support DP
        moe_config.moe_parallel_config.use_ep = True
        moe_config.moe_parallel_config.use_dp = False

        selected_backend, experts_cls = select_unquantized_moe_backend(
            moe_config=moe_config
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM
        assert experts_cls is not None


@patch(
    "vllm.utils.flashinfer.has_flashinfer",
    return_value=True,
)
@patch(
    "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.TrtLlmBf16Experts.is_supported_config",
    return_value=(False, None),
)
@patch(
    "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe.FlashInferExperts.is_supported_config",
    return_value=(True, None),
)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Only supported on NVIDIA platforms."
)
def test_select_cuda_flashinfer_cutlass_backend(
    mock_has_flashinfer,
    mock_is_supported_trtllm,
    mock_is_supported_cutlass,
):
    """Test CUDA backend selection when FlashInfer TRTLLM is not available
    and FlashInfer CUTLASS is available."""
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(current_platform, "is_tpu", return_value=False),
        patch.object(current_platform, "is_out_of_tree", return_value=False),
        patch.object(current_platform, "has_device_capability", return_value=True),
    ):
        moe_config = make_dummy_moe_config()
        # Select FlashInfer CUTLASS explicitly
        moe_config.moe_backend = "flashinfer_cutlass"
        # CUTLASS requires EP and does not support DP
        moe_config.moe_parallel_config.use_ep = True
        moe_config.moe_parallel_config.use_dp = False

        selected_backend, experts_cls = select_unquantized_moe_backend(
            moe_config=moe_config
        )

        assert selected_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS
        assert experts_cls is not None


@skipif_not_cuda_rocm
def test_select_lora_backend_prefers_triton():
    """LoRA-enabled unquantized MoE should select Triton backend."""
    moe_config = make_dummy_moe_config()
    moe_config.is_lora_enabled = True
    selected_backend, experts_cls = select_unquantized_moe_backend(
        moe_config=moe_config
    )

    assert selected_backend == UnquantizedMoeBackend.TRITON
    assert experts_cls is not None


@skipif_not_cuda_rocm
def test_select_lora_explicit_non_triton_backend():
    """LoRA should override explicit non-Triton backend to Triton."""
    moe_config = make_dummy_moe_config()
    moe_config.is_lora_enabled = True

    # Use string from mapping in function map_unquantized_backend()
    moe_config.moe_backend = "flashinfer_cutlass"

    selected_backend, experts_cls = select_unquantized_moe_backend(
        moe_config=moe_config
    )

    assert selected_backend == UnquantizedMoeBackend.TRITON
    assert experts_cls is not None


@skipif_not_cuda_rocm
@pytest.mark.parametrize("is_lora_enabled", [False, True])
def test_select_explicit_triton_backend(is_lora_enabled):
    """Explicit triton backend selection should return Triton."""
    moe_config = make_dummy_moe_config()
    moe_config.is_lora_enabled = is_lora_enabled
    moe_config.moe_backend = "triton"

    selected_backend, experts_cls = select_unquantized_moe_backend(
        moe_config=moe_config
    )

    assert selected_backend == UnquantizedMoeBackend.TRITON
    assert experts_cls is not None
