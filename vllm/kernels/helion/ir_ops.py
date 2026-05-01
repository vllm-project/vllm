# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion IrOpImpl registrations.

Registers Helion kernels as ``"helion"`` provider implementations of vLLM IR ops.

Two registration paths are available:

* **Single-kernel (helper)**: use :func:`register_as_simple_vllm_ir_impl` when the
  entire IrOpImpl body is a single :class:`HelionKernelWrapper`. ``supported`` and
  ``supports_args`` are derived automatically.

* **Composite (manual)**: call ``ir_op.register_impl("helion", ...)`` directly when
  the impl body calls multiple Helion kernels or mixes in non-Helion code. The
  developer must author an explicit ``supports_args`` that covers the union of
  constraints across the whole composition.
"""

import functools
import inspect
from typing import TYPE_CHECKING

from vllm.utils.import_utils import has_helion

if TYPE_CHECKING:
    from vllm.ir.op import IrOp, IrOpImpl
    from vllm.kernels.helion.register import HelionKernelWrapper

HELION_SUPPORTED = has_helion()


def register_as_simple_vllm_ir_impl(
    ir_op: "IrOp",
    kernel: "HelionKernelWrapper",
) -> "IrOpImpl":
    """Register a single HelionKernelWrapper as the ``"helion"`` IrOpImpl for an IR op.

    Automates ``supported``, ``supports_args``, and ``impl_fn`` wiring for the
    single-kernel case. For composite impls (multiple kernels or non-Helion code),
    use ``ir_op.register_impl("helion", ...)`` directly with an explicit
    ``supports_args``.

    The kernel's ``raw_kernel_func`` must have the same signature as the IR op's
    native implementation — they compute the same operation, so this is a natural
    requirement.
    """
    sig = inspect.signature(kernel.raw_kernel_func)

    def _supports_args(*args, **kwargs) -> bool:
        if kernel._disabled:
            return False
        configured = kernel._configured_kernel
        config_keys = list(configured.configs.keys())
        selected = configured.config_picker(args, config_keys)
        return selected is not None or "default" in config_keys

    _supports_args.__signature__ = sig

    @functools.wraps(kernel.raw_kernel_func)
    def helion_impl(*args, **kwargs):
        return kernel(*args, **kwargs)

    helion_impl.__signature__ = sig

    return ir_op.register_impl(
        "helion",
        supported=not kernel._disabled,
        supports_args=_supports_args,
    )(helion_impl)


# ---------------------------------------------------------------------------
# Per-op registrations
# ---------------------------------------------------------------------------

if HELION_SUPPORTED:
    import vllm.ir as ir
    from vllm.kernels.helion.ops.silu_mul_fp8 import silu_mul_fp8 as _silu_mul_fp8

    register_as_simple_vllm_ir_impl(ir.ops.silu_and_mul_fp8, _silu_mul_fp8)


__all__: list[str] = ["register_as_simple_vllm_ir_impl"]
