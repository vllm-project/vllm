# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-selection of the NvFp4 MoE backend on SM12x (no GPU required)."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform


def _sm12x_env(
    device_capability: tuple[int, int],
    b12x: bool = True,
    fi_cutlass: bool = False,
) -> ExitStack:
    """Mock a CUDA SM12x device with controlled backend availability."""
    stack = ExitStack()
    for cm in (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(
            current_platform,
            "is_device_capability",
            side_effect=lambda cap, device_id=0: cap == device_capability,
        ),
        patch.object(
            current_platform,
            "is_device_capability_family",
            side_effect=lambda family, device_id=0: family == 120,
        ),
        patch.object(
            current_platform,
            "has_device_capability",
            side_effect=lambda cap, device_id=0: True,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe"
            ".has_flashinfer_b12x_moe",
            return_value=b12x,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe"
            ".has_flashinfer_cutlass_fused_moe",
            return_value=fi_cutlass,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.experts.cutlass_moe"
            ".cutlass_group_gemm_supported",
            return_value=False,
        ),
    ):
        stack.enter_context(cm)
    return stack


def _select(config, weight_key, activation_key):
    backend, _ = select_nvfp4_moe_backend(config, weight_key, activation_key)
    return backend


def test_sm120_w4a16_auto_selects_b12x():
    """SM120 W4A16 NVFP4 checkpoints get the native b12x backend instead of
    the Marlin weight-only fallback."""
    with _sm12x_env((12, 0)):
        backend = _select(make_dummy_moe_config(), kNvfp4Static, None)
    assert backend == NvFp4MoeBackend.FLASHINFER_B12X


def test_sm120_w4a4_keeps_flashinfer_cutlass_priority():
    """b12x sits behind FLASHINFER_CUTLASS, so W4A4 selection is unchanged."""
    with _sm12x_env((12, 0), fi_cutlass=True):
        backend = _select(make_dummy_moe_config(), kNvfp4Static, kNvfp4Dynamic)
    assert backend == NvFp4MoeBackend.FLASHINFER_CUTLASS


def test_sm121_auto_stays_on_marlin():
    """SM121 (DGX Spark) keeps the Marlin fallback; b12x remains an explicit
    opt-in there."""
    with _sm12x_env((12, 1)):
        backend = _select(make_dummy_moe_config(), kNvfp4Static, None)
    assert backend == NvFp4MoeBackend.MARLIN


def test_sm120_ep_deployment_rejects_b12x():
    """The b12x kernel expects global-expert-count weights, so EP deployments
    must not select it."""
    config = make_dummy_moe_config(num_experts=4, num_local_experts=2)
    config.moe_parallel_config.use_ep = True
    config.moe_parallel_config.ep_size = 2
    with _sm12x_env((12, 0)):
        backend = _select(config, kNvfp4Static, None)
    assert backend == NvFp4MoeBackend.MARLIN


def test_sm120_explicit_b12x_with_ep_raises():
    """Explicit --moe-backend flashinfer_b12x under EP fails fast with a
    clear error instead of a shape mismatch inside the kernel."""
    config = make_dummy_moe_config(num_experts=4, num_local_experts=2)
    config.moe_parallel_config.use_ep = True
    config.moe_parallel_config.ep_size = 2
    config.moe_backend = "flashinfer_b12x"
    with _sm12x_env((12, 0)), pytest.raises(ValueError, match="parallel config"):
        _select(config, kNvfp4Static, None)
