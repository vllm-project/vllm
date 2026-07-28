# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared dispatch support for checked-in Helion-generated kernels."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any

import torch
from torch.library import Library

from vllm.kernels.helion_generated.manifests import (
    GENERATED_KERNEL_MANIFESTS,
    PRESERVED_SPECIALIZATION_MANIFESTS,
)
from vllm.platforms import current_platform

_SUPPORTED_PLATFORM_NAMES = {
    "nvidia_b200": "nvidia_b200",
    "nvidia_h100": "nvidia_h100",
    "nvidia_h100_80gb_hbm3": "nvidia_h100",
    "nvidia_h100_nvl": "nvidia_h100",
    "nvidia_h100_pcie": "nvidia_h100",
    "nvidia_h100_sxm5": "nvidia_h100",
}

vllm_helion_generated_lib = Library("vllm_helion_generated", "FRAGMENT")


def _canonical_device_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _runtime_platform() -> str | None:
    if not current_platform.is_cuda():
        return None
    name = _canonical_device_name(current_platform.get_device_name())
    return _SUPPORTED_PLATFORM_NAMES.get(name)


@cache
def _select_bucketed_module(
    kernel_name: str,
    platform: str | None,
    static_key: tuple[int, ...],
    num_tokens: int,
) -> str | None:
    if platform is None or num_tokens < 1:
        return None
    kernels = GENERATED_KERNEL_MANIFESTS.get(kernel_name, {}).get(platform)
    if kernels is None:
        return None
    available_static_keys = {case[:-1] for case in kernels}
    if (kernel_name, platform) in PRESERVED_SPECIALIZATION_MANIFESTS:
        matching_arity = {
            candidate
            for candidate in available_static_keys
            if len(candidate) == len(static_key)
        }
        if not matching_arity:
            return None
        selected_static_key = min(
            matching_arity,
            key=lambda candidate: (
                tuple(
                    abs(actual - tuned) for actual, tuned in zip(static_key, candidate)
                ),
                candidate,
            ),
        )
    elif static_key in available_static_keys:
        selected_static_key = static_key
    else:
        return None
    buckets = sorted(case[-1] for case in kernels if case[:-1] == selected_static_key)
    if not buckets:
        return None
    token_bucket = next((size for size in buckets if size >= num_tokens), buckets[-1])
    selected_case = (*selected_static_key, token_bucket)
    return next(
        (module_path for case, module_path in kernels.items() if case == selected_case),
        None,
    )


@cache
def _load_launcher(module_path: str) -> Callable[..., None]:
    return importlib.import_module(module_path).call


def _selected_cases(
    kernel_name: str, token_counts: Iterable[int]
) -> tuple[tuple[int, ...], ...]:
    platform = _runtime_platform()
    kernels = GENERATED_KERNEL_MANIFESTS.get(kernel_name, {}).get(platform or "", {})
    counts = tuple(count for count in token_counts if count > 0)
    by_static_key: dict[tuple[int, ...], list[int]] = {}
    for case in kernels:
        by_static_key.setdefault(case[:-1], []).append(case[-1])

    selected: set[tuple[int, ...]] = set()
    for static_key, available in by_static_key.items():
        buckets = sorted(available)
        for count in counts:
            bucket = next(
                (candidate for candidate in buckets if candidate >= count),
                buckets[-1],
            )
            selected.add((*static_key, bucket))
    return tuple(sorted(selected))


def _schema_tail(op: torch._ops.OpOverload) -> str:
    schema = str(op._schema)
    return schema[schema.index("(") :]


def _mutation_signature(
    op: torch._ops.OpOverload,
) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (arg.name, bool(arg.alias_info and arg.alias_info.is_write))
        for arg in op._schema.arguments
    )


def _make_capture_routed_impl(
    native_op: torch._ops.OpOverload,
    generated_op: torch._ops.OpOverload,
) -> Callable[..., Any]:
    schema_args = list(generated_op._schema.arguments)
    names = [arg.name for arg in schema_args]
    defaults = {
        arg.name: arg.default_value for arg in schema_args if arg.has_default_value()
    }

    def impl(*args: object, **kwargs: object) -> Any:
        values = list(args)
        for name in names[len(args) :]:
            values.append(kwargs[name] if name in kwargs else defaults[name])
        if torch.cuda.is_current_stream_capturing():
            return generated_op(*values)
        return native_op(*values)

    return impl


def build_compiled_generated_op_map() -> dict[
    torch._ops.OpOverload, torch._ops.OpOverload
]:
    from vllm.kernels.helion_generated.ops import import_all_ops

    routed: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}
    platform = _runtime_platform()
    for module in import_all_ops():
        generated_name = module.OP_NAME
        native_name = module.NATIVE_OP_NAME
        if native_name is None:
            continue
        if platform not in GENERATED_KERNEL_MANIFESTS.get(generated_name, {}):
            continue
        native_packet = getattr(torch.ops._C, native_name, None)
        generated_packet = getattr(
            torch.ops.vllm_helion_generated, generated_name, None
        )
        if native_packet is None or generated_packet is None:
            continue
        native_op = native_packet.default
        generated_op = generated_packet.default
        if _mutation_signature(native_op) != _mutation_signature(generated_op):
            raise RuntimeError(
                f"Generated op mutation mismatch for {generated_name}: "
                f"native={native_op._schema}, generated={generated_op._schema}"
            )

        routed_name = f"routed_{generated_name}"
        if not hasattr(torch.ops.vllm_helion_generated, routed_name):
            vllm_helion_generated_lib.define(routed_name + _schema_tail(generated_op))
            vllm_helion_generated_lib.impl(
                routed_name,
                _make_capture_routed_impl(native_op, generated_op),
                "CUDA",
            )
            vllm_helion_generated_lib._register_fake(
                routed_name, lambda *args, **kwargs: None
            )
        routed[native_op] = getattr(
            torch.ops.vllm_helion_generated, routed_name
        ).default
    return routed
