# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-forward NVFP4 W4A16 / W4A4 selection.

At small M a weight-only (W4A16) GEMM beats a fully quantised (W4A4) one:
decode is memory bound, so quantising activations buys little while costing an
extra quantise pass. At large M the FP4 tensor cores win. The crossover is a
property of the *kernel pair*, not of the hardware, so it is a tunable.

The blocker is weight layout, not dispatch. Every W4A16 backend available today
rewrites the checkpoint weights during ``process_weights_after_loading``
(Marlin shuffles; the FlashInfer CuTe-DSL path pads, swizzles and runs a
prepare step). Pairing such a kernel with a W4A4 kernel that wants a different
layout would require keeping both resident, which costs more memory than the
dispatch saves and defeats the purpose.

A kernel therefore opts in by declaring ``preserves_checkpoint_layout = True``,
meaning its ``process_weights_after_loading`` leaves the quantised weight and
scales in checkpoint-native form. No in-tree kernel declares that yet, so this
dispatch stays disabled by default and fails loudly rather than silently
producing wrong numbers or double-allocating.
"""

import torch

from vllm.logger import init_logger

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class LayoutMismatchError(RuntimeError):
    """Paired kernels do not agree on an on-device weight layout."""


def _layout_signature(layer: torch.nn.Module) -> tuple:
    """Shape/dtype fingerprint of the quantised weight and its scales."""
    out = []
    for name in ("weight", "weight_scale", "weight_global_scale"):
        p = getattr(layer, name, None)
        out.append(None if p is None else (tuple(p.shape), str(p.dtype)))
    return tuple(out)


class DynamicNvFp4LinearKernel(NvFp4LinearKernel):
    """Route each forward to a W4A16 or W4A4 kernel based on M."""

    def __init__(
        self,
        c: NvFp4LinearLayerConfig,
        a16_kernel: NvFp4LinearKernel,
        a4_kernel: NvFp4LinearKernel,
        a16_max_m: int,
    ) -> None:
        super().__init__(c)
        self.a16_kernel = a16_kernel
        self.a4_kernel = a4_kernel
        self.a16_max_m = a16_max_m

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Support is a property of the paired kernels, checked when they are
        # constructed, not of this wrapper.
        return True, None

    @classmethod
    def can_implement(cls, c: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    @staticmethod
    def check_pairable(
        a16_kernel: NvFp4LinearKernel, a4_kernel: NvFp4LinearKernel
    ) -> tuple[bool, str | None]:
        """Both kernels must consume the checkpoint-native layout."""
        for kernel in (a16_kernel, a4_kernel):
            if not getattr(type(kernel), "preserves_checkpoint_layout", False):
                return False, (
                    f"{type(kernel).__name__} rewrites weights in "
                    "process_weights_after_loading, so it cannot share one "
                    "weight layout with its dispatch partner"
                )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        ok, reason = self.check_pairable(self.a16_kernel, self.a4_kernel)
        if not ok:
            raise LayoutMismatchError(reason)

        before = _layout_signature(layer)
        self.a16_kernel.process_weights_after_loading(layer)
        after = _layout_signature(layer)
        if after != before:
            # The kernel claimed to preserve the layout but did not. Catch it
            # here rather than let the partner kernel read rewritten weights.
            raise LayoutMismatchError(
                f"{type(self.a16_kernel).__name__} declares "
                f"preserves_checkpoint_layout but changed the layout "
                f"{before} -> {after}"
            )
        self.a4_kernel.process_weights_after_loading(layer)
        if _layout_signature(layer) != before:
            raise LayoutMismatchError(
                f"{type(self.a4_kernel).__name__} declares "
                "preserves_checkpoint_layout but changed the layout"
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = x.numel() // x.shape[-1]
        kernel = self.a16_kernel if m <= self.a16_max_m else self.a4_kernel
        return kernel.apply_weights(layer, x, bias)
