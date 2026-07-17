# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph-aware routing for Helion kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs

# Fusion-only Helion ops and the native op each one replaces. Keys are Helion
# kernel names (torch.ops.vllm_helion.<key>); values are the native op emitted
# by vLLM's post-grad fusion passes (torch.ops._C.<value>). The correspondence
# is declared explicitly rather than assuming the two always share a name. The
# remaining registered Helion kernels have eager call sites or an incompatible
# schema and need a different routing path.
_HELION_TO_NATIVE_OP: dict[str, str] = {
    "rms_norm_dynamic_per_token_quant": "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant": "rms_norm_per_block_quant",
    "silu_and_mul_per_block_quant": "silu_and_mul_per_block_quant",
    "fused_qk_norm_rope": "fused_qk_norm_rope",
}


def _schema_tail(op: torch._ops.OpOverload) -> str:
    schema = str(op._schema)
    return schema[schema.index("(") :]


def _make_routed_impl(
    native_op: torch._ops.OpOverload,
    helion_op: torch._ops.OpOverload,
) -> Callable[..., Any]:
    schema_args = list(helion_op._schema.arguments)
    names = [arg.name for arg in schema_args]
    defaults = {
        arg.name: arg.default_value for arg in schema_args if arg.has_default_value()
    }

    def impl(*args: object, **kwargs: object) -> Any:
        values = list(args)
        for name in names[len(args) :]:
            values.append(kwargs[name] if name in kwargs else defaults[name])
        if torch.cuda.is_current_stream_capturing():
            return helion_op(*values)
        return native_op(*values)

    return impl


def build_compiled_helion_op_map() -> dict[
    torch._ops.OpOverload, torch._ops.OpOverload
]:
    """Return native-to-routed mappings for compatible fusion-only ops."""
    from vllm.kernels.helion.ops import import_all_kernels
    from vllm.kernels.helion.register import _HOP_AVAILABLE, vllm_helion_lib

    if _HOP_AVAILABLE:
        return {}

    import_all_kernels()
    routed: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}

    for helion_name, native_name in _HELION_TO_NATIVE_OP.items():
        native_packet = getattr(torch.ops._C, native_name, None)
        helion_packet = getattr(torch.ops.vllm_helion, helion_name, None)
        if native_packet is None or helion_packet is None:
            continue

        native_op = native_packet.default
        helion_op = helion_packet.default

        routed_name = f"routed_{helion_name}"
        if not hasattr(torch.ops.vllm_helion, routed_name):
            vllm_helion_lib.define(routed_name + _schema_tail(helion_op))
            vllm_helion_lib.impl(
                routed_name,
                _make_routed_impl(native_op, helion_op),
                "CUDA",
            )
            vllm_helion_lib._register_fake(routed_name, lambda *args, **kwargs: None)

        routed[native_op] = getattr(torch.ops.vllm_helion, routed_name).default

    return routed


def register_compiled_routed_helion_ops() -> None:
    """Eagerly define the routed Helion ops (idempotent).

    Assumes helion is installed when ``VLLM_USE_HELION_KERNELS`` is set;
    ``VllmConfig.__post_init__`` fails fast otherwise.

    ``build_compiled_helion_op_map`` defines the ``vllm_helion.routed_*`` ops as
    a side effect, but it is only reached when ``HelionFusionRoutingPass`` runs
    at compile time. On an AOT compile-cache hit the pass never runs, yet the
    loaded graph still references the routed ops, so they must already exist in
    the process. Called from the AOT-artifact load path (see decorators.py) to
    keep cached graphs resolvable.
    """
    if not envs.VLLM_USE_HELION_KERNELS:
        return
    build_compiled_helion_op_map()
