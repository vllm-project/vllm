# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.platforms import current_platform

HEAD_DIM = 128
CONV_WIDTH = 4
# Tokens per step that `KdaFused`, the fallback of every step size, verifies.
MIN_WINDOW, MAX_WINDOW = 2, 8


def can_use_deferred_recoverssm(
    *,
    window: int,
    head_dim: int,
    conv_width: int,
    has_conv_bias: bool,
    activation_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    f_b_dtype: torch.dtype,
    lower_bound: float | None,
) -> tuple[bool, str | None]:
    """Whether the deferred verify (`KdaDeferredVerify`) can serve a KDA layer
    with an FP32 SSM state, and if not, why."""
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return False, "it needs SM10x tensor cores (tcgen05)"
    if not MIN_WINDOW <= window <= MAX_WINDOW:
        return False, f"it verifies {MIN_WINDOW} to {MAX_WINDOW} tokens per step"
    if head_dim != HEAD_DIM or conv_width != CONV_WIDTH:
        return False, (
            f"it needs head_dim {HEAD_DIM} and a width-{CONV_WIDTH} convolution"
        )
    if has_conv_bias:
        return False, "it does not apply a convolution bias"
    if lower_bound is None:
        return False, "it needs a bounded KDA gate (gate_lower_bound)"
    if (activation_dtype, conv_state_dtype, f_b_dtype) != (torch.bfloat16,) * 3:
        return False, "it needs bfloat16 activations, conv state and F_B weight"
    return True, None
