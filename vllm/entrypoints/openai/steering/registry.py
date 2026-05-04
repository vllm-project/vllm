# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named steering vector registry.

Provides :class:`SteeringModuleRegistry` for loading, storing, and
resolving named steering vector configurations that can be referenced
by name in API requests.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.config.steering_types import (
    SteeringVectorSpec,
    merge_steering_specs,
    normalize_layer_entry,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)


@dataclass
class SteeringModule:
    """A named steering vector configuration."""

    name: str
    vectors: SteeringVectorSpec | None = None
    prefill_vectors: SteeringVectorSpec | None = None
    decode_vectors: SteeringVectorSpec | None = None


class SteeringModuleRegistry:
    """Registry for named steering vector configurations."""

    def __init__(self, valid_layer_indices: set[int] | None = None) -> None:
        self._modules: dict[str, SteeringModule] = {}
        self._lock = asyncio.Lock()
        self._valid_layer_indices = valid_layer_indices

    async def register(
        self,
        name: str,
        vectors: SteeringVectorSpec | None = None,
        prefill_vectors: SteeringVectorSpec | None = None,
        decode_vectors: SteeringVectorSpec | None = None,
    ) -> None:
        """Register a named steering module. Overwrites if name exists."""
        # Validate that at least one tier has vectors
        if not vectors and not prefill_vectors and not decode_vectors:
            raise ValueError(f"Steering module '{name}' has no vectors in any tier")

        # Validate hook point names and entry format
        for spec in [vectors, prefill_vectors, decode_vectors]:
            if spec:
                invalid = set(spec.keys()) - VALID_HOOK_POINT_NAMES
                if invalid:
                    raise ValueError(
                        f"Invalid hook point name(s) in module '{name}': "
                        f"{sorted(invalid)}. "
                        f"Valid: {sorted(VALID_HOOK_POINT_NAMES)}"
                    )
                # Validate entries are well-formed
                for hook_name, layers in spec.items():
                    for layer_idx, entry in layers.items():
                        self._validate_layer_index(name=name, layer_idx=layer_idx)
                        self._validate_layer_entry(
                            name=name,
                            hook_name=hook_name,
                            layer_idx=layer_idx,
                            entry=entry,
                        )

        module = SteeringModule(
            name=name,
            vectors=vectors,
            prefill_vectors=prefill_vectors,
            decode_vectors=decode_vectors,
        )
        async with self._lock:
            self._modules[name] = module
        logger.info("Registered steering module '%s'", name)

    async def unregister(self, name: str) -> bool:
        """Remove a named module. Returns True if it existed."""
        async with self._lock:
            removed = self._modules.pop(name, None)
        if removed:
            logger.info("Unregistered steering module '%s'", name)
        return removed is not None

    def get(self, name: str) -> SteeringModule | None:
        """Look up a module by name. Thread-safe for reads."""
        return self._modules.get(name)

    def list_modules(self) -> list[str]:
        """Return sorted list of registered module names."""
        return sorted(self._modules.keys())

    def dump_for_broadcast(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-safe view of every registered module.

        Used by the API server to broadcast the full registry to workers
        via ``collective_rpc`` so every worker holds an identical
        ``_steering_module_registry``.  The returned mapping is keyed by
        module name; each value is a ``dict`` with ``vectors``,
        ``prefill_vectors`` and ``decode_vectors`` entries (any of which
        may be ``None``).  All structures are plain Python collections
        suitable for cloudpickle serialization across the engine
        multiprocess boundary.

        Note: this returns the *unresolved* :class:`SteeringVectorSpec`
        form — per-layer entries may still be in
        ``{"vector": [...], "scale": float}`` form.  The worker-side
        merge step normalizes scales just like the original
        ``resolve_for_request`` did on the server.
        """
        return {
            name: {
                "vectors": module.vectors,
                "prefill_vectors": module.prefill_vectors,
                "decode_vectors": module.decode_vectors,
            }
            for name, module in self._modules.items()
        }

    async def load_from_file(self, name: str, path: str) -> None:
        """Load a steering module from a JSON file and register it.

        Expected JSON format::

            {
                "vectors": {"post_mlp": {"14": [0.1, ...]}},
                "prefill_vectors": {"pre_attn": {"14": [0.3, ...]}},
                "decode_vectors": null,
            }

        JSON uses string keys for layer indices; they are converted to int.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Steering module file not found: {path}")

        with open(file_path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Steering module file must contain a JSON object, "
                f"got {type(data).__name__}"
            )

        # Extract three tiers, converting string layer keys to int
        vectors = _convert_layer_keys(data.get("vectors"), field_name="vectors")
        prefill_vectors = _convert_layer_keys(
            data.get("prefill_vectors"), field_name="prefill_vectors"
        )
        decode_vectors = _convert_layer_keys(
            data.get("decode_vectors"), field_name="decode_vectors"
        )

        await self.register(
            name=name,
            vectors=vectors,
            prefill_vectors=prefill_vectors,
            decode_vectors=decode_vectors,
        )

    def resolve_for_request(
        self,
        steering_name: str,
        inline_vectors: SteeringVectorSpec | None,
        inline_prefill: SteeringVectorSpec | None,
        inline_decode: SteeringVectorSpec | None,
    ) -> tuple[
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
        str | None,
    ]:
        """Resolve a named module and merge with inline vectors.

        Returns ``(merged_vectors, merged_prefill, merged_decode, error)``.
        On success *error* is ``None``.  On failure the first three are
        ``None``.
        """
        module = self.get(steering_name)
        if module is None:
            return (
                None,
                None,
                None,
                (
                    f"Unknown steering module '{steering_name}'. "
                    f"Available: {self.list_modules() or 'none'}"
                ),
            )

        try:
            merged_vectors = merge_steering_specs(module.vectors, inline_vectors)
            merged_prefill = merge_steering_specs(
                module.prefill_vectors, inline_prefill
            )
            merged_decode = merge_steering_specs(module.decode_vectors, inline_decode)
        except ValueError as exc:
            return (
                None,
                None,
                None,
                (f"Invalid steering composition for module '{steering_name}': {exc}"),
            )

        return merged_vectors, merged_prefill, merged_decode, None

    @staticmethod
    def _validate_layer_entry(
        name: str,
        hook_name: str,
        layer_idx: int,
        entry: Any,
    ) -> None:
        """Validate a stored layer entry matches request-time constraints."""
        prefix = f"module {name!r}[{hook_name!r}][{layer_idx}]"
        vector: Any
        if isinstance(entry, dict):
            vector, scale = normalize_layer_entry(entry)
            if not isinstance(entry["scale"], (int, float)):
                raise ValueError(
                    f"{prefix}['scale'] must be a finite float, "
                    f"got {type(entry['scale']).__name__}."
                )
            if not math.isfinite(scale):
                raise ValueError(f"{prefix}['scale'] must be finite, got {scale}.")
        else:
            vector, _ = normalize_layer_entry(entry)

        if not isinstance(vector, list):
            target = prefix if isinstance(entry, list) else f"{prefix}['vector']"
            raise ValueError(
                f"{target} must be a list of floats, got {type(vector).__name__}."
            )

        value_prefix = prefix if isinstance(entry, list) else f"{prefix}['vector']"
        for i, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{value_prefix}[{i}] must be a finite float, "
                    f"got {type(value).__name__}."
                )
            if not math.isfinite(value):
                raise ValueError(f"{value_prefix}[{i}] must be finite, got {value}.")

    def _validate_layer_index(self, name: str, layer_idx: int) -> None:
        """Validate layer indices against the loaded model when available."""
        if self._valid_layer_indices is None:
            return
        if layer_idx not in self._valid_layer_indices:
            raise ValueError(
                f"Steering module '{name}' references unknown layer index "
                f"{layer_idx}. Valid steerable layers: "
                f"{sorted(self._valid_layer_indices) or 'none'}"
            )


def _convert_layer_keys(
    spec: dict[str, Any] | None,
    *,
    field_name: str,
) -> SteeringVectorSpec | None:
    """Convert JSON string layer keys to int."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(
            f"Steering module field '{field_name}' must be a JSON object, "
            f"got {type(spec).__name__}"
        )
    if not spec:
        return None
    result: SteeringVectorSpec = {}
    for hook_name, layers in spec.items():
        if not isinstance(layers, dict):
            raise ValueError(
                f"Steering module field '{field_name}'[{hook_name!r}] must be "
                f"a JSON object mapping layer indices to entries, got "
                f"{type(layers).__name__}"
            )
        converted: dict[int, Any] = {}
        for layer_key, entry in layers.items():
            try:
                converted[int(layer_key)] = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Steering module field '{field_name}'[{hook_name!r}] has "
                    f"invalid layer index {layer_key!r}; expected an integer"
                ) from exc
        if converted:
            result[hook_name] = converted
    return result if result else None
