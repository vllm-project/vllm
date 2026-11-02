# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """Request for a LoRA adapter.

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
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
    """Whether this adapter's MoE weights are stored in the 3D fused
    `gate_up_proj` / `down_proj` layout (one fused tensor per layer) or the
    2D per-expert split layout (separate `gate_proj` / `up_proj` / `down_proj`
    tensors per expert). Only consulted when the engine is started with
    `enable_mixed_moe_lora_format=True`; otherwise it is ignored and the
    on-disk format is inferred from the base model."""

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

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
        """Compares lora_name and lora_int_id. A name loaded again at runtime
        gets a new lora_int_id while requests for the old one may still be
        running, and the engine needs both adapters active.
        """
        return (
            isinstance(value, self.__class__)
            and self.lora_name == value.lora_name
            and self.lora_int_id == value.lora_int_id
        )

    def __hash__(self) -> int:
        return hash((self.lora_name, self.lora_int_id))
