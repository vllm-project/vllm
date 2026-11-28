# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HummingLinearKernel's environment guards (CPU-only).

`HummingLinearKernel` is the last entry in `_POSSIBLE_KERNELS[CUDA]`, so
`choose_mp_linear_kernel` reaches it only after every other mixed-precision
kernel has declined. At that point `can_implement` is the single thing standing
between a non-CUDA build, a build without the Humming extension, or an
unsupported 8-bit activation dtype and a kernel that cannot run them. Each of
those three guards returns a distinct reason, and the reasons are what shows up
in the `Failed to find a kernel ...` error, so their order matters too.

The existing `test_w4a16_kernel_selection.py` covers this selection surface on
ROCm only (both tests are skipped off ROCm), and Humming is not in
`_POSSIBLE_KERNELS[ROCM]`.

Run `pytest tests/kernels/quantization/test_humming_kernel_guards.py`.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.kernels.linear import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.humming import (
    HummingLinearKernel,
)
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.cpu_test

_MODULE = "vllm.model_executor.kernels.linear.mixed_precision.humming"

# 128 keeps every config here on a group size Humming accepts, so these tests
# exercise the environment guards and nothing else.
_GROUP_SIZE = 128


def _config(act_type: torch.dtype) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(512, 256),
        partition_weight_shape=(512, 256),
        weight_type=scalar_types.uint4b8,
        act_type=act_type,
        group_size=_GROUP_SIZE,
        zero_points=False,
    )


def _can_implement(
    config: MPLinearLayerConfig,
    *,
    is_cuda: bool = True,
    has_humming: bool = True,
) -> tuple[bool, str | None]:
    platform = MagicMock()
    platform.is_cuda.return_value = is_cuda
    with (
        patch(f"{_MODULE}.current_platform", platform),
        patch(f"{_MODULE}.has_humming", return_value=has_humming),
    ):
        return HummingLinearKernel.can_implement(config)


def test_rejects_non_cuda_platforms():
    can_implement, reason = _can_implement(_config(torch.bfloat16), is_cuda=False)
    assert not can_implement
    assert reason is not None and "CUDA" in reason


def test_rejects_cuda_builds_without_the_humming_extension():
    can_implement, reason = _can_implement(_config(torch.bfloat16), has_humming=False)
    assert not can_implement
    assert reason is not None and "not installed" in reason


def test_the_platform_guard_runs_before_the_install_check():
    # On a non-CUDA platform `has_humming()` may probe a CUDA-only extension,
    # so the platform guard has to answer first -- and the reason the user sees
    # should name the platform, not a missing install.
    platform = MagicMock()
    platform.is_cuda.return_value = False
    with (
        patch(f"{_MODULE}.current_platform", platform),
        patch(f"{_MODULE}.has_humming", return_value=False) as has_humming,
    ):
        can_implement, reason = HummingLinearKernel.can_implement(
            _config(torch.bfloat16)
        )

    assert not can_implement
    assert reason is not None and "CUDA" in reason
    has_humming.assert_not_called()


@pytest.mark.parametrize("act_type", [torch.bfloat16, torch.float16, torch.float32])
def test_accepts_activation_dtypes_wider_than_one_byte(act_type: torch.dtype):
    can_implement, reason = _can_implement(_config(act_type))
    assert can_implement, reason


@pytest.mark.parametrize("act_type", [torch.int8, torch.float8_e4m3fn])
def test_accepts_the_8bit_activation_dtypes_with_a_quant_key(act_type: torch.dtype):
    # These two are the keys of `_ACT_QUANT_KEYS`; per-token quantization of the
    # activations is requested through them.
    can_implement, reason = _can_implement(_config(act_type))
    assert can_implement, reason


@pytest.mark.parametrize("act_type", [torch.float8_e5m2, torch.uint8])
def test_rejects_8bit_activation_dtypes_without_a_quant_key(act_type: torch.dtype):
    can_implement, reason = _can_implement(_config(act_type))
    assert not can_implement
    assert reason is not None and str(act_type) in reason


def test_min_capability_stays_at_turing():
    # `choose_mp_linear_kernel` compares this against the device capability
    # before it ever calls `can_implement`. Raising it would silently drop the
    # Humming path on every pre-SM80 device without any other test noticing.
    assert HummingLinearKernel.get_min_capability() == 75
