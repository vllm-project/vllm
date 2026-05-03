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

import numpy as np

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
        allowed = {"vector", "scale"}
        extra = set(entry.keys()) - allowed
        if extra:
            raise ValueError(
                f"Scaled steering entry has unexpected keys: {sorted(extra)}; "
                f"allowed keys: ['scale', 'vector']"
            )
        missing = allowed - set(entry.keys())
        if missing:
            raise ValueError(
                f"Scaled steering entry missing required key(s): "
                f"{sorted(missing)}; got keys: {sorted(entry.keys())}"
            )
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
            f"Cannot add steering vectors of different lengths: {len(a)} vs {len(b)}"
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


def scale_steering_spec(
    spec: SteeringVectorSpec | None,
    scale: float,
) -> SteeringVectorSpec | None:
    """Apply a uniform multiplier to every entry in *spec*.

    Returns a new spec where each layer entry's effective magnitude has
    been multiplied by *scale*.  Per-layer ``{"vector": ..., "scale": ...}``
    entries have their inner ``scale`` field multiplied; bare-list entries
    are wrapped in the dict form with the new scale.

    Used by the worker-side named-module resolver to apply the
    request's module-level scale before merging with inline overrides.
    Returns ``None`` if *spec* is ``None`` or empty.  When *scale* equals
    ``1.0`` the input is returned unchanged.
    """
    if not spec:
        return None
    if scale == 1.0:
        return spec
    result: SteeringVectorSpec = {}
    for hook, layer_dict in spec.items():
        if not layer_dict:
            continue
        scaled_layers: dict[int, SteeringLayerEntry] = {}
        for layer_idx, entry in layer_dict.items():
            vec, sc = normalize_layer_entry(entry)
            scaled_layers[layer_idx] = {"vector": vec, "scale": sc * scale}
        if scaled_layers:
            result[hook] = scaled_layers
    return result if result else None


def merge_steering_specs(
    a: SteeringVectorSpec | None,
    b: SteeringVectorSpec | None,
) -> SteeringVectorSpec | None:
    """Additively merge two :class:`SteeringVectorSpec` dicts.

    For overlapping ``(hook, layer)`` entries both sides are pre-scaled
    (via :func:`normalize_layer_entry` + :func:`_scale_vector`) then
    summed (via :func:`_add_vectors`).  Non-overlapping entries pass
    through pre-scaled.

    Returns ``None`` if both inputs are ``None`` or empty.
    """
    a_empty = not a
    b_empty = not b
    if a_empty and b_empty:
        return None

    result: SteeringVectorSpec = {}

    all_hooks: set[str] = set()
    if not a_empty:
        assert a is not None
        all_hooks.update(a.keys())
    if not b_empty:
        assert b is not None
        all_hooks.update(b.keys())

    for hook in all_hooks:
        a_layers = a.get(hook, {}) if a else {}
        b_layers = b.get(hook, {}) if b else {}

        all_layer_idxs: set[int] = set()
        all_layer_idxs.update(a_layers.keys())
        all_layer_idxs.update(b_layers.keys())

        if not all_layer_idxs:
            continue

        hook_result: dict[int, SteeringLayerEntry] = {}
        for layer_idx in all_layer_idxs:
            a_entry = a_layers.get(layer_idx)
            b_entry = b_layers.get(layer_idx)

            if a_entry is not None and b_entry is not None:
                a_vec, a_scale = normalize_layer_entry(a_entry)
                b_vec, b_scale = normalize_layer_entry(b_entry)
                scaled_a = _scale_vector(a_vec, a_scale)
                scaled_b = _scale_vector(b_vec, b_scale)
                hook_result[layer_idx] = _add_vectors(scaled_a, scaled_b)
            elif a_entry is not None:
                vec, scale = normalize_layer_entry(a_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)
            else:
                assert b_entry is not None
                vec, scale = normalize_layer_entry(b_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)

        if hook_result:
            result[hook] = hook_result

    return result if result else None


def hash_steering_config(
    effective_vectors: dict[str, dict[int, list[float]]] | None,
    module_ref: tuple[str, float] | None = None,
) -> int:
    """Deterministic SHA-256 hash of pre-resolved steering vectors.

    Returns 0 if both *effective_vectors* and *module_ref* are ``None``
    or empty.  The hash is masked to fit in ``np.int64``.

    *module_ref* is an optional ``(name, scale)`` reference to a
    worker-side named steering module.  When set, the reference is
    incorporated into the hash so that two requests with the same
    ``(name, scale)`` reference plus identical inline overrides produce
    the same hash, while different references (or different scales)
    produce different hashes.  When ``module_ref`` is ``None`` this
    function reduces to the original "hash inline-only vectors" behavior
    bit-for-bit, preserving prefix-cache reuse for existing requests.

    Hashes the binary representation of each layer vector (via
    ``np.asarray(...).tobytes()``) instead of stringifying the raw Python
    floats. The previous ``str(sorted(...))`` approach took ~28 ms per call
    on Gemma-3-4B (87K floats) because ``str`` invokes ``float.__repr__``
    on every element; this version is ~30x faster because ``tobytes`` is a
    memcpy and ``hashlib.sha256.update`` is hardware-accelerated.
    """
    if not effective_vectors and module_ref is None:
        return 0
    h = hashlib.sha256()
    if effective_vectors:
        for hook in sorted(effective_vectors.keys()):
            h.update(hook.encode())
            layer_dict = effective_vectors[hook]
            for layer_idx in sorted(layer_dict.keys()):
                entry = layer_dict[layer_idx]
                # An entry is either a bare list/array of floats or a dict
                # ``{"vector": [...], "scale": float}``. By the time we get
                # here the resolver has flattened the dict form into a plain
                # list, so we expect the bare form — but handle both for
                # safety.
                if isinstance(entry, dict):
                    vec = entry.get("vector", entry)
                    scale = float(entry.get("scale", 1.0))
                else:
                    vec = entry
                    scale = 1.0
                arr = np.asarray(vec, dtype=np.float32)
                h.update(layer_idx.to_bytes(4, "little", signed=True))
                h.update(arr.tobytes())
                if scale != 1.0:
                    h.update(np.float64(scale).tobytes())
    if module_ref is not None:
        # Domain-separator byte ensures a request with only a module_ref
        # cannot collide with an inline-vector request whose hook name
        # happens to match the module name.  Inline vectors are written
        # before this block, so the separator unambiguously demarcates
        # the module-ref segment of the digest.
        name, scale = module_ref
        h.update(b"\x00module_ref\x00")
        h.update(name.encode("utf-8"))
        h.update(np.float64(scale).tobytes())
    return int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
