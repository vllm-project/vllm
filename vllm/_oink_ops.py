# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small helper wrappers for external Oink Blackwell custom ops.

vLLM does not depend on the external Oink repository/package. When an external
plugin registers torch.library.custom_op entrypoints under the `oink::`
namespace (e.g. via vLLM's general_plugins mechanism) and
`VLLM_USE_OINK_OPS=1` is set, vLLM can route eligible calls to those ops.

This module provides:
- A single place to probe Oink op availability at module init time
  (outside torch.compile tracing), and
- Thin wrappers around the torch.ops entrypoints for use in CUDA fast paths,
  without introducing graph breaks.

Important:
  Do not call the availability helpers in a compiled region. They may call
  functions decorated with `torch._dynamo.disable` to safely check
  conditions that should not be traced.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

try:
    from torch._dynamo import disable as _dynamo_disable  # type: ignore[attr-defined]
except Exception:  # pragma: no cover

    def _dynamo_disable(fn: Callable):  # type: ignore[misc]
        return fn


def _has_oink_op(op_name: str) -> bool:
    """Check if a specific oink op is registered."""
    return hasattr(torch.ops, "oink") and hasattr(torch.ops.oink, op_name)


@_dynamo_disable
def is_oink_available_for_device(device_index: int) -> bool:
    """Return True if Oink ops are registered and device is SM100+.

    This function is intended to be called during module initialization
    (e.g., in RMSNorm.__init__), not in the forward path.

    External plugins are expected to gate registration on SM100+ and
    VLLM_USE_OINK_OPS=1, so if the ops are present they should be usable.
    """
    if not torch.cuda.is_available():
        return False

    try:
        major, minor = torch.cuda.get_device_capability(device_index)
        sm = 10 * major + minor
        if sm < 100:
            return False
    except Exception:
        return False

    return _has_oink_op("rmsnorm")


def has_fused_add_rms_norm() -> bool:
    """Return True if the in-place fused op is registered."""
    return _has_oink_op("fused_add_rms_norm")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Call `torch.ops.oink.rmsnorm`.

    This wrapper is safe to call in torch.compile regions.
    """
    return torch.ops.oink.rmsnorm(x, weight, eps)


def fused_add_rms_norm_(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Call `torch.ops.oink.fused_add_rms_norm` (mutates x and residual)."""
    torch.ops.oink.fused_add_rms_norm(x, residual, weight, eps)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper returning (x, residual) after in-place mutation."""
    fused_add_rms_norm_(x, residual, weight, eps)
    return x, residual
