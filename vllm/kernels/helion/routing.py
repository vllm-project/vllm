# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph-aware routing for Helion kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

# Helion ops and the platform fallback op each one replaces. CUDA fallbacks are
# mutation-based vLLM ops; ROCm fallbacks are functional AITER ops. The
# correspondence is explicit because names and namespaces differ.
_HELION_TO_NATIVE_OP: dict[str, str] = {
    "rms_norm_per_block_quant": "rms_norm_per_block_quant",
    "silu_and_mul_per_block_quant": "silu_and_mul_per_block_quant",
    # Also emitted directly (not only by fusion) — a standalone activation quant
    # that survives fusion is retargeted here; its eager call sites are routed
    # separately in input_quant_fp8.QuantFP8.forward_cuda.
    "per_token_group_fp8_quant": "per_token_group_fp8_quant",
}

# ROCm Helion kernels use the same functional schemas as the corresponding
# AITER ops. Routing these functional ops avoids adapting AITER's returned
# tensors through mutation-based output buffers and copies.
_HELION_TO_ROCM_AITER_OP: dict[str, str] = {
    "rms_norm_per_block_quant": "rocm_aiter_rmsnorm_fp8_group_quant",
    "silu_and_mul_per_block_quant": "rocm_aiter_act_mul_and_fp8_group_quant",
    "per_token_group_fp8_quant": "rocm_aiter_group_fp8_quant",
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
        can_use_helion = not (
            current_platform.is_rocm()
            and "transpose_scale" in names
            and values[names.index("transpose_scale")]
        )
        if torch.cuda.is_current_stream_capturing() and can_use_helion:
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

    if current_platform.is_rocm():
        fallback_namespace = torch.ops.vllm
        helion_to_fallback = _HELION_TO_ROCM_AITER_OP
    else:
        fallback_namespace = torch.ops._C
        helion_to_fallback = _HELION_TO_NATIVE_OP

    for helion_name, fallback_name in helion_to_fallback.items():
        native_packet = getattr(fallback_namespace, fallback_name, None)
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
            if helion_op._schema.returns:

                def routed_fake(
                    *args: object,
                    _helion_op: torch._ops.OpOverload = helion_op,
                    **kwargs: object,
                ) -> Any:
                    return _helion_op(*args, **kwargs)

                vllm_helion_lib._register_fake(routed_name, routed_fake)
            else:
                vllm_helion_lib._register_fake(
                    routed_name, lambda *args, **kwargs: None
                )

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
