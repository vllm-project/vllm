# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

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

    source: Selects how the adapter is loaded.
        - "path": load from lora_path.
        - "memory": load from peft_config and lora_tensors.
          In this mode, lora_path is ignored.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False
    source: Literal["path", "memory"] = "path"
    peft_config: dict | None = None
    lora_tensors: dict | None = None

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        if self.source == "path":
            if not self.lora_path:
                raise ValueError("LoRA source 'path' requires a non-empty lora_path.")
            return

        if self.peft_config is None or self.lora_tensors is None:
            raise ValueError(
                "LoRA source 'memory' requires peft_config and lora_tensors."
            )

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
