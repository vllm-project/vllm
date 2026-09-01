# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-forward NVFP4 W4A16 / W4A4 selection.

At small M a weight-only (W4A16) GEMM beats a fully quantised (W4A4) one:
decode is memory bound, so quantising activations buys little while costing an
extra quantise pass. At large M the FP4 tensor cores win. The crossover is a
property of the *kernel pair*, not of the hardware, so it is a tunable.

The blocker is weight layout, not dispatch. Kernels prepare weights into
backend-specific forms during ``process_weights_after_loading``: Marlin applies
its own shuffle, and the FlashInfer CuTe-DSL W4A16 path runs
``prepare_bf16_fp4_weights``, which returns int32 rather than the packed uint8
of the checkpoint. Pairing two kernels that disagree would mean keeping both
layouts resident, which costs more memory than the dispatch saves.

A kernel therefore declares ``nvfp4_weight_layout``, naming the layout it
consumes. Two kernels may be paired only if the names match. This deliberately
does *not* require checkpoint-native weights: a shared prepared layout works
just as well, and is the likelier route, since it only needs one backend to
consume the layout another already produces. Kernels that declare nothing are
never paired, because a silent mismatch yields wrong numerics rather than an
error.
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
        """Both kernels must declare the same on-device weight layout."""
        declared: dict[str, str] = {}
        for kernel in (a16_kernel, a4_kernel):
            layout = getattr(type(kernel), "nvfp4_weight_layout", None)
            if layout is None:
                return False, (
                    f"{type(kernel).__name__} does not declare "
                    "nvfp4_weight_layout, so it cannot be paired: a kernel "
                    "must name the layout it consumes before the dispatcher "
                    "can know one set of weights serves both"
                )
            declared[type(kernel).__name__] = layout
        if len(set(declared.values())) > 1:
            pairs = ", ".join(f"{k}={v}" for k, v in sorted(declared.items()))
            return False, (
                f"paired kernels consume different weight layouts ({pairs}); "
                "one set of weights cannot serve both"
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
                f"nvfp4_weight_layout but changed the layout "
                f"{before} -> {after}"
            )
        self.a4_kernel.process_weights_after_loading(layer)
        if _layout_signature(layer) != before:
            raise LayoutMismatchError(
                f"{type(self.a4_kernel).__name__} declares "
                "nvfp4_weight_layout but changed the layout"
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        m = x.numel() // x.shape[-1]
        kernel = self.a16_kernel if m <= self.a16_max_m else self.a4_kernel
        return kernel.apply_weights(layer, x, bias, **kwargs)
