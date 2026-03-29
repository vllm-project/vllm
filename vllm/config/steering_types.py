# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Type definitions and helpers for steering vector composition.

The additive composition model:

    effective_prefill[hook][layer] =
        scale(steering_vectors[hook][layer])
        + scale(prefill_steering_vectors[hook][layer])

    effective_decode[hook][layer] =
        scale(steering_vectors[hook][layer])
        + scale(decode_steering_vectors[hook][layer])

Where ``scale(entry)`` means: if entry is a bare list, scale=1.0; if entry is
``{"vector": [...], "scale": float}``, multiply vector by scale.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Per-layer entry: bare list (scale=1.0) or {"vector": [...], "scale": float}
SteeringLayerEntry = list[float] | dict[str, Any]

# Full spec: {hook_point_name: {layer_idx: SteeringLayerEntry}}
SteeringVectorSpec = dict[str, dict[int, SteeringLayerEntry]]


def normalize_layer_entry(entry: SteeringLayerEntry) -> tuple[list[float], float]:
    """Return ``(vector, scale)`` from a steering layer entry.

    If *entry* is a bare ``list[float]``, returns ``(entry, 1.0)``.
    If *entry* is ``{"vector": [...], "scale": float}``, returns
    ``(entry["vector"], entry["scale"])``.
    """
    if isinstance(entry, list):
        return entry, 1.0
    if isinstance(entry, dict):
        return entry["vector"], float(entry["scale"])
    raise TypeError(
        f"SteeringLayerEntry must be a list or dict, got {type(entry).__name__}"
    )


def _scale_vector(vec: list[float], scale: float) -> list[float]:
    """Multiply each element of *vec* by *scale*."""
    return [v * scale for v in vec]


def _add_vectors(a: list[float], b: list[float]) -> list[float]:
    """Element-wise addition of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(
            f"Cannot add steering vectors of different lengths: "
            f"{len(a)} vs {len(b)}"
        )
    return [x + y for x, y in zip(a, b)]


def resolve_effective_vectors(
    base: SteeringVectorSpec | None,
    phase_specific: SteeringVectorSpec | None,
) -> dict[str, dict[int, list[float]]] | None:
    """Merge *base* and *phase_specific* steering specs additively.

    For each ``(hook, layer)`` pair, both the base and phase-specific entries
    are pre-scaled and then summed.  Non-overlapping entries pass through
    unchanged (pre-scaled).

    Returns pre-scaled flat vectors (``list[float]``, no scale wrapper).
    Returns ``None`` if both inputs are ``None`` or empty.
    """
    base_empty = not base
    phase_empty = not phase_specific
    if base_empty and phase_empty:
        return None

    result: dict[str, dict[int, list[float]]] = {}

    # Collect all hook points from both specs
    all_hooks: set[str] = set()
    if not base_empty:
        assert base is not None
        all_hooks.update(base.keys())
    if not phase_empty:
        assert phase_specific is not None
        all_hooks.update(phase_specific.keys())

    for hook in all_hooks:
        base_layers = base.get(hook, {}) if base else {}
        phase_layers = phase_specific.get(hook, {}) if phase_specific else {}

        all_layer_idxs: set[int] = set()
        all_layer_idxs.update(base_layers.keys())
        all_layer_idxs.update(phase_layers.keys())

        if not all_layer_idxs:
            continue

        hook_result: dict[int, list[float]] = {}
        for layer_idx in all_layer_idxs:
            base_entry = base_layers.get(layer_idx)
            phase_entry = phase_layers.get(layer_idx)

            if base_entry is not None and phase_entry is not None:
                base_vec, base_scale = normalize_layer_entry(base_entry)
                phase_vec, phase_scale = normalize_layer_entry(phase_entry)
                scaled_base = _scale_vector(base_vec, base_scale)
                scaled_phase = _scale_vector(phase_vec, phase_scale)
                hook_result[layer_idx] = _add_vectors(scaled_base, scaled_phase)
            elif base_entry is not None:
                vec, scale = normalize_layer_entry(base_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)
            else:
                assert phase_entry is not None
                vec, scale = normalize_layer_entry(phase_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)

        if hook_result:
            result[hook] = hook_result

    return result if result else None


def hash_steering_config(
    effective_vectors: dict[str, dict[int, list[float]]] | None,
) -> int:
    """Deterministic SHA-256 hash of pre-resolved steering vectors.

    Returns 0 if *effective_vectors* is ``None`` or empty.
    The hash is masked to fit in ``np.int64``.
    """
    if not effective_vectors:
        return 0
    canonical = {
        hp: sorted(vecs.items())
        for hp, vecs in sorted(effective_vectors.items())
    }
    data = str(sorted(canonical.items())).encode()
    return int(hashlib.sha256(data).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
