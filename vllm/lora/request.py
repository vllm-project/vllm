from dataclasses import dataclass
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
    lora_local_path: str
    long_lora_max_len: Optional[int] = None
    __hash__ = AdapterRequest.__hash__

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def local_path(self):
        return self.lora_local_path
