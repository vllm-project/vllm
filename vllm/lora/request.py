# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    Request for a LoRA adapter.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.

    load_inplace: If True, forces reloading the adapter even if one
        with the same lora_int_id already exists in the cache. This replaces
        the existing adapter in-place. If False (default), only loads if the
        adapter is not already loaded.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
    """Whether this adapter's MoE weights are stored in the 3D fused
    `gate_up_proj` / `down_proj` layout (one fused tensor per layer) or the
    2D per-expert split layout (separate `gate_proj` / `up_proj` / `down_proj`
    tensors per expert). Only consulted when the engine is started with
    `enable_mixed_moe_lora_format=True`; otherwise it is ignored and the
    on-disk format is inferred from the base model."""

    lora_scale: float = 1.0
    """Per-request multiplier applied on top of the adapter's own `alpha / r`
    scaling. Two requests in the same batch may share the same `lora_int_id`
    while using different `lora_scale` values: the scale is kept as
    per-request state and applied inside the LoRA kernels, so a distinct
    scale does not consume an extra adapter slot. `1.0` (the default)
    reproduces the stock behavior exactly.

    NOTE: This is deliberately *not* part of `__eq__` / `__hash__`, which key
    on `lora_name` only. The adapter set passed to the LoRA manager is about
    which weights to load; the scale is looked up per request index instead."""

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        if self.lora_scale < 0:
            raise ValueError(f"lora_scale must be >= 0, got {self.lora_scale}")

        # Ensure lora_path is not empty
        assert self.lora_path, "lora_path cannot be empty"

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare LoRARequest
        instances based on lora_name. This allows for identification
        and comparison lora adapter across engines.
        """
        return isinstance(value, self.__class__) and self.lora_name == value.lora_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash LoRARequest instances
        based on lora_name. This ensures that LoRARequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.lora_name)
