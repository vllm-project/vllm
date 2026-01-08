# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

import msgspec

from vllm.utils.counter import AtomicCounter

lora_id_counter = AtomicCounter(0)

# Sentinel value to detect if user explicitly passed lora_int_id
_LORA_INT_ID_UNSET = -111

# Registry to track lora_name -> lora_int_id mapping
_lora_id_registry: dict[str, int] = {}


def reset_lora_id_counter() -> None:
    """
    Reset the LoRA ID counter and clear the registry.

    This is useful for testing or when you need to completely
    reset the LoRA adapter state.
    """
    global lora_id_counter
    lora_id_counter = AtomicCounter(0)
    _lora_id_registry.clear()


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    Request for a LoRA adapter.
    """

    lora_name: str
    lora_int_id: int = _LORA_INT_ID_UNSET
    lora_path: str = ""

    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None

    def __post_init__(self):
        # Only warn when user explicitly passes lora_int_id
        if self.lora_int_id != _LORA_INT_ID_UNSET:
            warnings.warn(
                "The 'lora_int_id' parameter is deprecated and will be "
                "removed in a future version. It is now automatically "
                "generated and the passed value will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Reuse existing ID for the same lora_name, or generate a new one
        if self.lora_name in _lora_id_registry:
            self.lora_int_id = _lora_id_registry[self.lora_name]
        else:
            # Always use auto-generated ID
            self.lora_int_id = lora_id_counter.inc(1)
            _lora_id_registry[self.lora_name] = self.lora_int_id
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
        return isinstance(value, self.__class__) and self.lora_name == value.lora_name

    def __hash__(self) -> int:
        return hash(self.lora_name)
