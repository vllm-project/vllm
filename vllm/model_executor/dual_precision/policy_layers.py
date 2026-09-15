# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure name-matching and layer/module policies for the low-precision shadow store.

Nothing in this module touches torch state; every function is a pure
function of module names and policy strings so it can be unit-tested on CPU
against the module-name fixtures derived from real checkpoints.
"""

from __future__ import annotations

import re

import torch.nn as nn

QUANTIZED_WEIGHT_NAMES: tuple[str, ...] = (
    "qweight",
    "weight_packed",
    "w13_qweight",
    "w2_qweight",
    "w13_weight_packed",
    "w2_weight_packed",
)
"""Parameter names that mark a ``LinearBase`` as GPTQ-packed."""

NVFP4_WEIGHT_NAMES: tuple[str, ...] = (
    "weight_global_scale",
    "weight_scale_2",
    "w13_weight_global_scale",
    "w2_weight_global_scale",
)
"""Parameter names that mark a ``LinearBase`` as ModelOpt NVFP4.

NVFP4 keeps its packed weight under the plain name ``weight`` (uint8, two FP4
values per byte), which a BF16 linear also has, so the marker must be one of the
scales instead.  ``weight_scale_2`` is the name the checkpoint loads under and
``weight_global_scale`` the name ``process_weights_after_loading`` renames it to,
so a layer is recognised both before and after that pass.
"""

SHADOW_WEIGHT_NAMES: tuple[str, ...] = QUANTIZED_WEIGHT_NAMES + NVFP4_WEIGHT_NAMES
"""Every marker that makes a ``LinearBase`` usable as a low-precision shadow."""

MODULE_POLICY_ALL = "all"
MODULE_POLICY_MLP_ONLY = "mlp_only"
_MLP_ONLY_SUFFIXES = (
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.gate_up_proj",
    ".mlp.down_proj",
)
_TRANSFORMER_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def is_quantized_shadow_layer(layer: nn.Module) -> bool:
    """Whether ``layer`` carries a packed (quantized) weight.

    True for the GPTQ packings and for ModelOpt NVFP4; a plain BF16 linear, whose
    only weight tensor is ``weight``, is false.
    """
    return any(hasattr(layer, name) for name in SHADOW_WEIGHT_NAMES)


def transformer_layer_index(module_name: str) -> int | None:
    """Zero-based transformer block index of ``module_name`` or ``None``."""
    match = _TRANSFORMER_LAYER_PATTERN.search(module_name)
    return int(match.group(1)) if match is not None else None


def resolve_bf16_layer_indices(policy: str, num_layers: int) -> frozenset[int]:
    """Resolve a BF16 transformer-block policy into zero-based layer indices.

    ``policy`` is the ``VLLM_DUAL_PRECISION_BF16_LAYERS`` selector: ``none``,
    or a comma-separated list of ``first:N``, ``last:N``, single indices and
    ``a-b`` ranges.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")

    normalized = policy.strip().lower()
    if normalized == "none":
        return frozenset()
    if not normalized:
        raise ValueError(
            "VLLM_DUAL_PRECISION_BF16_LAYERS cannot be empty; use 'none' to "
            "switch every quantized transformer block to INT4."
        )

    indices: set[int] = set()
    for raw_term in normalized.split(","):
        term = raw_term.strip()
        if not term:
            raise ValueError(
                "VLLM_DUAL_PRECISION_BF16_LAYERS contains an empty selector."
            )

        if term.startswith("first:") or term.startswith("last:"):
            side, raw_count = term.split(":", 1)
            try:
                count = int(raw_count)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid dual precision layer selector {term!r}."
                ) from exc
            if not 0 <= count <= num_layers:
                raise ValueError(
                    f"Dual precision selector {term!r} is outside a "
                    f"{num_layers}-layer model."
                )
            if side == "first":
                indices.update(range(count))
            else:
                indices.update(range(num_layers - count, num_layers))
            continue

        if "-" in term:
            raw_start, raw_end = term.split("-", 1)
            try:
                start = int(raw_start)
                end = int(raw_end)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid dual precision layer range {term!r}."
                ) from exc
            if start > end:
                raise ValueError(
                    f"Dual precision layer range {term!r} must be ascending."
                )
            selected: range | tuple[int, ...] = range(start, end + 1)
        else:
            try:
                selected = (int(term),)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid dual precision layer selector {term!r}."
                ) from exc

        for index in selected:
            if not 0 <= index < num_layers:
                raise ValueError(
                    f"Dual precision layer index {index} is outside a "
                    f"{num_layers}-layer model."
                )
            indices.add(index)

    return frozenset(indices)


def should_attach_int4_shadow(
    module_name: str,
    bf16_layer_indices: frozenset[int],
    module_policy: str = MODULE_POLICY_ALL,
) -> bool:
    """Whether the quantized peer of ``module_name`` joins the shadow store.

    Modules outside a transformer block (no ``layers.N`` in the name) and
    modules inside a BF16-policy block never attach. ``mlp_only`` restricts
    attachment to the regular gate/up/down projections.
    """
    layer_index = transformer_layer_index(module_name)
    if layer_index is None or layer_index in bf16_layer_indices:
        return False
    normalized = module_policy.strip().lower()
    if normalized == MODULE_POLICY_ALL:
        return True
    if normalized == MODULE_POLICY_MLP_ONLY:
        return module_name.endswith(_MLP_ONLY_SUFFIXES)
    raise ValueError(
        "VLLM_DUAL_PRECISION_INT4_MODULES must be 'all' or 'mlp_only', "
        f"got {module_policy!r}."
    )


def format_layer_indices(indices: frozenset[int]) -> str:
    """Compact ``a-b,c`` rendering of a layer-index set for log lines."""
    if not indices:
        return "none"

    ranges: list[str] = []
    start = previous = min(indices)
    for index in sorted(indices)[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def plan_shadow_attachment(
    bf16_linear_names: list[str],
    int4_quantized_names: set[str] | frozenset[str],
    bf16_layer_indices: frozenset[int],
    module_policy: str = MODULE_POLICY_ALL,
) -> tuple[list[str], list[str], list[str]]:
    """Split BF16 linear names into (attached, policy_bf16, fallback).

    ``attached`` have a quantized peer and pass both policies; ``policy_bf16``
    have a quantized peer but are kept BF16 by policy; ``fallback`` have no
    quantized peer in the shadow checkpoint. Order follows
    ``bf16_linear_names``.
    """
    attached: list[str] = []
    policy_bf16: list[str] = []
    fallback: list[str] = []
    for name in bf16_linear_names:
        if name not in int4_quantized_names:
            fallback.append(name)
        elif should_attach_int4_shadow(name, bf16_layer_indices, module_policy):
            attached.append(name)
        else:
            policy_bf16.append(name)
    return attached, policy_bf16, fallback
