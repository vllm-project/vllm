# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import math
from collections.abc import Sequence
from typing import Any

import msgspec

_MAX_FLOAT32 = (2.0 - 2.0**-23) * (2.0**127)


def _validate_lora_scale(scale: float) -> float:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(
            f"LoRA scale must be a finite number, got {type(scale).__name__}: {scale!r}"
        )
    scale_f = float(scale)
    if not math.isfinite(scale_f) or abs(scale_f) > _MAX_FLOAT32:
        raise ValueError(
            f"LoRA scale must be finite and representable in float32, got {scale!r}"
        )
    return 0.0 if scale_f == 0.0 else scale_f


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """Request for one or more LoRA adapters with optional static scales.

    Single-adapter form:
        `LoRARequest("adapter", 1, "/path/to/adapter", lora_scale=1.0)`

    Multi-adapter composite form:
        `LoRARequest.from_adapters([(adapter_a, 0.7), (adapter_b, 0.5)])`
        or `LoRARequest(adapters=[(adapter_a, 0.7), (adapter_b, 0.5)])`

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.

    load_inplace: If True, forces reloading the adapter even if one
        with the same lora_int_id already exists in the cache. This replaces
        the existing adapter in-place. If False (default), only loads if the
        adapter is not already loaded.
    """

    lora_name: str = ""
    lora_int_id: int = 0
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
    """Whether this adapter's MoE weights are stored in the 3D fused
    `gate_up_proj` / `down_proj` layout (one fused tensor per layer) or the
    2D per-expert split layout (separate `gate_proj` / `up_proj` / `down_proj`
    tensors per expert). Only consulted when the engine is started with
    `enable_mixed_moe_lora_format=True`; otherwise it is ignored and the
    on-disk format is inferred from the base model."""
    lora_scale: float = 1.0
    """Static per-request scale multiplier applied to this adapter's delta."""
    adapters: list[tuple["LoRARequest", float]] | None = None
    """Optional list of `(LoRARequest, scale)` pairs for per-request
    multi-adapter composition (`y = Wx + sum_i s_i B_i A_i x`)."""

    def __post_init__(self):
        self.lora_scale = _validate_lora_scale(self.lora_scale)

        if self.adapters is None:
            if self.lora_int_id < 1:
                raise ValueError(f"id must be > 0, got {self.lora_int_id}")

            # Ensure lora_path is not empty
            assert self.lora_path, "lora_path cannot be empty"
            return

        if not isinstance(self.adapters, (list, tuple)) or len(self.adapters) == 0:
            raise ValueError(
                "adapters must be a non-empty sequence of (LoRARequest, scale)"
            )
        if self.lora_int_id != 0:
            raise ValueError(
                "Composite LoRARequest cannot set lora_int_id; "
                "each constituent adapter provides its own lora_int_id."
            )
        if self.lora_path:
            raise ValueError(
                "Composite LoRARequest cannot set lora_path; "
                "each constituent adapter provides its own lora_path."
            )
        if self.lora_scale != 1.0:
            raise ValueError(
                "Composite LoRARequest cannot set lora_scale; "
                "specify per-adapter scales in adapters instead."
            )
        if self.load_inplace:
            raise ValueError(
                "Composite LoRARequest cannot set load_inplace; "
                "set load_inplace on each constituent adapter instead."
            )

        normalized: list[tuple[LoRARequest, float]] = []
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        for entry in self.adapters:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    "Each entry in adapters must be a (LoRARequest, scale) pair, "
                    f"got {entry!r}"
                )
            adapter, scale = entry
            if not isinstance(adapter, LoRARequest):
                raise ValueError(
                    "Each adapter in adapters must be a LoRARequest, "
                    f"got {type(adapter).__name__}"
                )
            if adapter.is_composite:
                raise ValueError("Nested composite LoRARequest is not supported.")
            if adapter.lora_scale != 1.0:
                raise ValueError(
                    f"Constituent adapter {adapter.lora_name!r} must have "
                    "lora_scale=1.0; pass its scale in the (adapter, scale) pair."
                )
            validated_scale = _validate_lora_scale(scale)
            if adapter.lora_int_id in seen_ids:
                raise ValueError(
                    f"Duplicate lora_int_id {adapter.lora_int_id} in composite "
                    "LoRARequest."
                )
            if adapter.lora_name in seen_names:
                raise ValueError(
                    f"Duplicate lora_name {adapter.lora_name!r} in composite "
                    "LoRARequest."
                )
            seen_ids.add(adapter.lora_int_id)
            seen_names.add(adapter.lora_name)
            normalized.append((adapter, validated_scale))

        self.adapters = normalized
        if not self.lora_name:
            self.lora_name = "+".join(adapter.lora_name for adapter, _ in normalized)

    @classmethod
    def from_adapters(
        cls,
        adapters: Sequence["LoRARequest | tuple[LoRARequest, float]"],
        lora_name: str = "",
    ) -> "LoRARequest":
        """Construct a composite LoRARequest from adapters and optional scales.

        Entries may be `(LoRARequest, scale)` tuples or bare `LoRARequest`
        objects (which default to scale `1.0`).
        """
        if not adapters:
            raise ValueError("adapters cannot be empty")
        pairs: list[tuple[LoRARequest, float]] = []
        for item in adapters:
            if isinstance(item, LoRARequest):
                pairs.append((item, 1.0))
            else:
                pairs.append(item)  # type: ignore[arg-type]
        return cls(lora_name=lora_name, adapters=pairs)

    @property
    def is_composite(self) -> bool:
        return self.adapters is not None

    @property
    def adapter_id(self) -> int | None:
        return None if self.is_composite else self.lora_int_id

    @property
    def name(self) -> str:
        return self.lora_name

    @property
    def path(self) -> str:
        return self.lora_path

    def get_adapters(self) -> tuple[tuple["LoRARequest", float], ...]:
        """Return `(adapter, scale)` pairs for this request."""
        if self.adapters is not None:
            return tuple(self.adapters)
        return ((self, self.lora_scale),)

    def get_adapter_requests(self) -> tuple["LoRARequest", ...]:
        """Return constituent single-adapter `LoRARequest` objects."""
        if self.adapters is not None:
            return tuple(adapter for adapter, _ in self.adapters)
        return (self,)

    def get_adapter_ids(self) -> tuple[int, ...]:
        """Return constituent adapter integer IDs."""
        if self.adapters is not None:
            return tuple(adapter.lora_int_id for adapter, _ in self.adapters)
        return (self.lora_int_id,)

    def get_adapter_scales(self) -> tuple[float, ...]:
        """Return constituent static adapter scales."""
        if self.adapters is not None:
            return tuple(scale for _, scale in self.adapters)
        return (self.lora_scale,)

    def __eq__(self, value: object) -> bool:
        """Compares lora_name and lora_int_id for single adapters (ignoring
        lora_scale so requests with different scales share the same adapter
        slot), and constituent `(adapter, scale)` entries for composites.
        """
        if not isinstance(value, self.__class__):
            return False
        if self.is_composite != value.is_composite:
            return False
        if self.is_composite:
            assert self.adapters is not None and value.adapters is not None
            return self.lora_name == value.lora_name and self.adapters == value.adapters
        return (
            self.lora_name == value.lora_name and self.lora_int_id == value.lora_int_id
        )

    def __hash__(self) -> int:
        if self.is_composite:
            assert self.adapters is not None
            return hash((self.lora_name, tuple(self.adapters)))
        return hash((self.lora_name, self.lora_int_id))


class _TensorLoRARequest(LoRARequest):
    """Worker-local request for internal integrations such as verl.

    Construct this after the framework has received and retained the complete
    adapter tensors on the worker. Do not pass it through engine requests or
    the public add_lora RPC. The framework owns update synchronization.

    Warning:
        Prefix cache keys use the adapter name and path, not tensor contents.
        Replacing weights under the same name and path can reuse stale KV cache
        and produce incorrect outputs, even if the adapter ID changes. The
        framework must invalidate the old KV cache before resuming inference;
        loading or removing an adapter does not do this automatically.

    """

    _lora_tensors: dict[str, Any] | None = None
    _peft_config: dict[str, Any] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self._lora_tensors is None or self._peft_config is None:
            raise ValueError("_lora_tensors and _peft_config must be provided together")
        if not self._lora_tensors:
            raise ValueError("_lora_tensors cannot be empty")
