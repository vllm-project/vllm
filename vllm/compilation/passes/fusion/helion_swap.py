# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Post-grad pass that swaps native fused-quant CUDA ops for Helion kernels.

Several Helion kernels in ``vllm/kernels/helion/ops`` are registered as custom
ops (``torch.ops.vllm_helion.*``) that are *drop-in* replacements for the native
``torch.ops._C.*`` fused-quant ops: they share the same schema (argument names,
types, and mutated-argument layout). Because the schemas match, we can rewrite
the compiled graph in place -- retargeting the ``auto_functionalized`` node (or a
direct mutating call) from the native op to its Helion equivalent, leaving all
keyword arguments and ``getitem`` users untouched.

This pass runs after the RMSNorm/activation quant fusion passes, so it catches
both the fused ops those passes emit and any standalone quant ops the model's
quant methods emit directly. It is gated by ``VLLM_USE_HELION_KERNELS`` (on by
default when ``helion`` is installed).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..inductor_pass import InductorPass
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

# auto_functionalized_v2 exists on newer PyTorch; fall back gracefully.
try:
    from torch._higher_order_ops.auto_functionalize import (
        auto_functionalized_v2,
    )

    _AF_TARGETS: tuple = (auto_functionalized, auto_functionalized_v2)
except ImportError:  # pragma: no cover - depends on torch version
    _AF_TARGETS = (auto_functionalized,)


# Native op name -> Helion op name. Names are identical except for silu, which
# has a different (functional) signature and is therefore not swapped here.
_SWAPPABLE_OP_NAMES: list[str] = [
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant",
    "dynamic_per_token_scaled_fp8_quant",
    "per_token_group_fp8_quant",
]


def _arg_signature(op: torch._ops.OpOverload) -> tuple:
    """Return (names, write-flags) for an op's arguments.

    A safe in-place retarget of an ``auto_functionalized`` node requires the two
    ops to agree on argument names (so the node's kwargs remain valid) and on the
    layout of mutated (written) arguments (so the ``auto_functionalized`` output
    tuple and its ``getitem`` users still line up). Argument *types* (e.g. int vs
    SymInt) and defaults are irrelevant since values are already bound in the
    graph, so they are intentionally excluded.
    """
    args = op._schema.arguments
    names = tuple(a.name for a in args)
    writes = tuple(bool(a.alias_info and a.alias_info.is_write) for a in args)
    return names, writes


def build_helion_op_map() -> dict[torch._ops.OpOverload, torch._ops.OpOverload]:
    """Map native ``torch.ops._C`` op overloads to Helion equivalents.

    Only includes ops whose native and Helion versions are both registered and
    whose schemas are compatible. Importing ``vllm.kernels.helion`` triggers the
    ``@register_kernel`` decorators that register the ``vllm_helion`` custom ops.
    """
    import vllm.kernels.helion  # noqa: F401  triggers Helion op registration
    from vllm.kernels.helion.register import _HOP_AVAILABLE

    if _HOP_AVAILABLE:
        # The HOP path does not register torch.ops.vllm_helion custom ops, so
        # there is nothing to swap in the post-grad graph. HOP integration must
        # happen at the Python call site during Dynamo tracing instead.
        logger.warning(
            "HelionKernelSwapPass is a no-op because the Helion HOP path is "
            "enabled (no vllm_helion custom ops to swap)."
        )
        return {}

    def _resolve(ns: Any, name: str) -> Any:
        try:
            return getattr(ns, name)
        except (AttributeError, RuntimeError):
            return None

    op_map: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}
    for name in _SWAPPABLE_OP_NAMES:
        native = _resolve(torch.ops._C, name)
        helion = _resolve(torch.ops.vllm_helion, name)
        if native is None or helion is None:
            continue
        native_ov = native.default
        helion_ov = helion.default
        if _arg_signature(native_ov) != _arg_signature(helion_ov):
            logger.warning(
                "Skipping Helion swap for '%s': incompatible arg signature "
                "(native=%s, helion=%s)",
                name,
                native_ov._schema,
                helion_ov._schema,
            )
            continue
        op_map[native_ov] = helion_ov
    return op_map


class HelionKernelSwapPass(VllmInductorPass):
    """Swap native fused-quant ops for Helion kernels (VLLM_USE_HELION_KERNELS)."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        helion_map = build_helion_op_map()
        # Route through cudagraph-aware ops: Helion only under CUDA-graph capture,
        # native _C in eager (avoids Helion's per-call dispatch overhead in eager).
        if helion_map:
            from vllm.kernels.helion.routing import build_routed_op_map

            self._op_map = build_routed_op_map(helion_map)
            logger.info(
                "HelionKernelSwapPass enabled (cudagraph-routed) for ops: %s",
                sorted(str(op) for op in self._op_map),
            )
        else:
            self._op_map = {}

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if not self._op_map:
            return

        count = 0
        for node in graph.nodes:
            if node.op != "call_function":
                continue

            # Direct mutating call: torch.ops._C.<op>.default(...)
            if node.target in self._op_map:
                node.target = self._op_map[node.target]
                count += 1
                continue

            # auto_functionalized(<op>, **kwargs): the op is the first arg.
            # Retargeting the op in place is safe because the Helion op shares
            # the native op's arg names and mutated-arg layout, so the node's
            # kwargs and getitem users stay valid.
            if node.target in _AF_TARGETS and node.args:
                helion_op = self._op_map.get(node.args[0])
                if helion_op is not None:
                    node.args = (helion_op, *node.args[1:])
                    count += 1

        if count:
            logger.info("HelionKernelSwapPass: swapped %d op(s) to Helion", count)

    def uuid(self) -> str:
        return InductorPass.hash_dict(
            {
                "pass": "HelionKernelSwapPass",
                "ops": sorted(str(op) for op in self._op_map),
            }
        )
