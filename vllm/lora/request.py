import warnings
from dataclasses import dataclass, field
from typing import Optional

from vllm.adapter_commons.request import AdapterRequest


@dataclass
class LoRARequest(AdapterRequest):
    """
    Request for a LoRA adapter.

    Note that this class should be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized LoRA adapters.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    lora_local_path: Optional[str] = field(default=None, repr=False)
    long_lora_max_len: Optional[int] = None
    __hash__ = AdapterRequest.__hash__

    def __post_init__(self):
        if 'lora_local_path' in self.__dict__:
            warnings.warn(
                "The 'lora_local_path' attribute is deprecated "
                "and will be removed in a future version. "
                "Please use 'lora_path' instead.",
                DeprecationWarning,
                stacklevel=2)
            if not self.lora_path:
                self.lora_path = self.lora_local_path or ""

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

    @property
    def local_path(self):
        warnings.warn(
            "The 'local_path' attribute is deprecated "
            "and will be removed in a future version. "
            "Please use 'path' instead.",
            DeprecationWarning,
            stacklevel=2)
        return self.lora_path

    @local_path.setter
    def local_path(self, value):
        warnings.warn(
            "The 'local_path' attribute is deprecated "
            "and will be removed in a future version. "
            "Please use 'path' instead.",
            DeprecationWarning,
            stacklevel=2)
        self.lora_path = value
