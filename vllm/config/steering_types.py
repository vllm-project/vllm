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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

# Per-layer entry: bare list (scale=1.0) or {"vector": [...], "scale": float}.
# This is the public, user-facing shape — the type alias is exposed as a
# pydantic field type by request/response models in ``vllm.entrypoints``,
# so it must remain a narrow, schema-generable union.  Internally,
# :func:`merge_steering_specs` produces ``np.ndarray`` entries that are
# re-fed into resolvers; :func:`normalize_layer_entry` handles that
# transparently without widening this alias.
SteeringLayerEntry = list[float] | dict[str, Any]

# Full spec: {hook_point_name: {layer_idx: SteeringLayerEntry}}
SteeringVectorSpec = dict[str, dict[int, SteeringLayerEntry]]


def normalize_layer_entry(
    entry: SteeringLayerEntry | np.ndarray,
) -> tuple[list[float] | np.ndarray, float]:
    """Return ``(vector, scale)`` from a steering layer entry.

    If *entry* is a bare ``list[float]`` or ``np.ndarray``, returns
    ``(entry, 1.0)``.  If *entry* is ``{"vector": [...], "scale": float}``,
    returns ``(entry["vector"], entry["scale"])``.

    The ndarray case supports re-feeding the output of
    :func:`merge_steering_specs` (which produces pre-scaled ``np.float64``
    arrays) through downstream resolvers without converting back to lists.
    """
    if isinstance(entry, np.ndarray):
        return entry, 1.0
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
        f"SteeringLayerEntry must be a list, dict, or ndarray, "
        f"got {type(entry).__name__}"
    )


def _scale_vector(vec: list[float] | np.ndarray, scale: float) -> np.ndarray:
    """Multiply *vec* by *scale*, returning a float64 numpy array.

    Arithmetic is performed in float64 to match the legacy Python-list path
    bit-for-bit at the float64→float32 boundary in ``hash_steering_config``.
    """
    arr = np.asarray(vec, dtype=np.float64)
    return arr * scale


def _add_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition of two equal-length float64 vectors."""
    if a.shape != b.shape:
        raise ValueError(
            f"Cannot add steering vectors of different lengths: "
            f"{a.shape[0]} vs {b.shape[0]}"
        )
    return a + b


def resolve_effective_vectors(
    base: SteeringVectorSpec | None,
    phase_specific: SteeringVectorSpec | None,
) -> dict[str, dict[int, np.ndarray]] | None:
    """Merge *base* and *phase_specific* steering specs additively.

    For each ``(hook, layer)`` pair, both the base and phase-specific entries
    are pre-scaled and then summed.  Non-overlapping entries pass through
    unchanged (pre-scaled).

    Returns pre-scaled flat vectors as 1-D ``np.float64`` arrays. The
    float64 dtype is required for hash-determinism parity with the legacy
    Python-list path (``hash_steering_config`` casts to float32 once at the
    SHA boundary).  Returns ``None`` if both inputs are ``None`` or empty.
    """
    base_empty = not base
    phase_empty = not phase_specific
    if base_empty and phase_empty:
        return None

    result: dict[str, dict[int, np.ndarray]] = {}

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

        hook_result: dict[int, np.ndarray] = {}
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


def _torch_dtype_to_pack_dtype(torch_dtype: object) -> np.dtype:
    """Pick the numpy dtype to pack steering vectors as for *torch_dtype*.

    Maps the model's compute dtype to a numpy dtype for the wire-format
    packing path.  Numpy lacks a native ``bfloat16`` (without the
    optional ``ml_dtypes`` package), so bf16 models fall back to
    ``float32`` — still a ~2.25× IPC reduction over msgpack-encoded
    Python float lists.
    """
    name = getattr(torch_dtype, "__str__", lambda: "")().rsplit(".", 1)[-1]
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float64":
        return np.dtype(np.float64)
    if name in ("bfloat16",):
        return np.dtype(np.float32)
    return np.dtype(np.float32)


def pack_effective_steering(
    spec_base: SteeringVectorSpec | None,
    spec_phase: SteeringVectorSpec | None,
    dtype: np.dtype | str,
) -> dict[str, dict[int, np.ndarray]] | None:
    """Resolve and pack inline steering specs in one shot.

    Equivalent to ``resolve_effective_vectors(spec_base, spec_phase)``
    cast to *dtype*, but does the cast inline so we never allocate the
    intermediate float64 arrays purely for the cast-then-discard.
    Used by the LLM client (and HTTP server) to build the
    ``effective_*_steering_packed`` fields on :class:`SamplingParams`
    before request submission.

    Returns ``None`` when both inputs are ``None`` / empty.
    """
    if not spec_base and not spec_phase:
        return None
    np_dtype = np.dtype(dtype)
    resolved = resolve_effective_vectors(spec_base, spec_phase)
    if resolved is None:
        return None
    out: dict[str, dict[int, np.ndarray]] = {}
    for hook, layer_dict in resolved.items():
        out[hook] = {
            layer_idx: arr.astype(np_dtype, copy=False)
            for layer_idx, arr in layer_dict.items()
        }
    return out


def pack_steering_for_dtype(
    spec: SteeringVectorSpec | None,
    dtype: np.dtype | str,
) -> dict[str, dict[int, np.ndarray]] | None:
    """Pre-bake a :class:`SteeringVectorSpec` into model-dtype ``ndarray`` form.

    Converts every per-layer entry — bare-list or ``{"vector", "scale"}``
    dict — into a 1-D ``np.ndarray`` in *dtype* with the inner ``scale``
    already applied.  Returned shape:
    ``{hook: {layer_idx: ndarray[dtype]}}``.

    Used by the inline-vectors fast path: the LLM client (or HTTP server)
    converts user-supplied list-of-floats into this packed form before
    serializing into the request body, so the wire payload carries
    ``len(vec) * dtype.itemsize`` bytes per layer instead of
    ``len(vec) * 9`` bytes (msgpack-encoded floats).  The downstream
    resolver (:func:`resolve_effective_vectors` and
    :func:`merge_steering_specs`) already accepts ``ndarray`` entries
    transparently, so packed inputs flow through without further
    conversion.

    Returns ``None`` if *spec* is ``None`` or empty.
    """
    if not spec:
        return None
    np_dtype = np.dtype(dtype)
    result: dict[str, dict[int, np.ndarray]] = {}
    for hook, layer_dict in spec.items():
        if not layer_dict:
            continue
        packed: dict[int, np.ndarray] = {}
        for layer_idx, entry in layer_dict.items():
            vec, scale = normalize_layer_entry(entry)
            arr = np.asarray(vec, dtype=np_dtype)
            if scale != 1.0:
                # Cast the scale to the target dtype so the multiply
                # stays in-place (avoiding an fp32 promotion that the
                # caller would only have to cast back).
                arr = arr * np_dtype.type(scale)
                # ``arr * scalar`` on a non-fp32 ndarray sometimes
                # returns a higher-precision result; force it back.
                if arr.dtype != np_dtype:
                    arr = arr.astype(np_dtype, copy=False)
            packed[layer_idx] = arr
        if packed:
            result[hook] = packed
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
    effective_vectors: dict[str, dict[int, list[float] | np.ndarray]] | None,
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

    Accepts entries as either ``list[float]`` (legacy callers) or
    ``np.ndarray`` (the float64 arrays produced by
    :func:`resolve_effective_vectors`).  In both cases the float→float32
    cast happens exactly once at the ``tobytes`` boundary, so hashes are
    bit-for-bit identical regardless of the input container.
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
                # An entry is either a list/ndarray of floats or a dict
                # ``{"vector": [...], "scale": float}``. By the time we get here
                # the resolver has flattened the dict form into a plain
                # array — but handle both for safety.
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


def maybe_pack_inline_steering_for_request(
    sp: SamplingParams,
    torch_dtype: object,
) -> None:
    """Pre-resolve + pack inline steering vectors on *sp* in-place.

    Shared between :class:`vllm.entrypoints.LLM` (sync path) and
    :class:`vllm.v1.engine.async_llm.AsyncLLM` (HTTP path), called just
    before the request crosses the multiprocessing boundary.  See
    :meth:`SamplingParams._effective_prefill_steering_packed` for the
    contract.

    No-ops when:

    - all inline tier fields are ``None`` (named-only or no-steering);
    - packed fields are already set (idempotency for callers that
      pre-packed).

    Mutates *sp* by:
    1. Reading ``prefill_steering_config_hash`` and
       ``decode_steering_config_hash`` to fix the hash against the
       original fp64-resolved values (preserves prefix-cache reuse).
    2. Setting ``effective_*_steering_packed`` to the model-dtype
       ``ndarray`` form of the resolved per-phase specs.
    3. Clearing ``steering_vectors`` / ``prefill_steering_vectors`` /
       ``decode_steering_vectors`` so the wire payload doesn't carry
       both forms.
    4. Stashing the packed dicts as the cached values for the
       ``effective_*_steering`` cached_properties so worker-side reads
       return them directly without re-resolving.
    """
    if (
        sp.steering_vectors is None
        and sp.prefill_steering_vectors is None
        and sp.decode_steering_vectors is None
    ):
        return
    if (
        sp._effective_prefill_steering_packed is not None
        or sp._effective_decode_steering_packed is not None
    ):
        return

    np_dtype = _torch_dtype_to_pack_dtype(torch_dtype)

    # Prime the hash cached_properties against the original fp64 path
    # so a packed and an unpacked submission of the same logical request
    # share a prefix-cache hash.
    _ = sp.prefill_steering_config_hash
    _ = sp.decode_steering_config_hash

    sp._effective_prefill_steering_packed = pack_effective_steering(
        sp.steering_vectors, sp.prefill_steering_vectors, np_dtype
    )
    sp._effective_decode_steering_packed = pack_effective_steering(
        sp.steering_vectors, sp.decode_steering_vectors, np_dtype
    )
    sp.steering_vectors = None
    sp.prefill_steering_vectors = None
    sp.decode_steering_vectors = None
    sp.__dict__["effective_prefill_steering"] = sp._effective_prefill_steering_packed
    sp.__dict__["effective_decode_steering"] = sp._effective_decode_steering_packed
