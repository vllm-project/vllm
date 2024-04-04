from dataclasses import dataclass
from typing import Optional
import hashlib


def positive_hash_sha256(input_string):
    """
    function to generate positive hash from input string, which is used to identify the model variant for lora
    sha-256 is used to keep it consistent between python versions and the sheets addon
    """
    return int(hashlib.sha256(input_string.encode('utf-8')).hexdigest(), 16) % (2 ** 63)


@dataclass
class LoRARequest:
    """
    Request for a LoRA adapter.

    Note that this class should be be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized LoRA adapters.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """

    lora_name: str
    lora_local_path: str
    lora_int_id: Optional[int] = 0

    def __post_init__(self):
        # if no int_id was given, use the name hash as id
        if not self.lora_int_id:
            self.lora_int_id = positive_hash_sha256(self.lora_name)
        if self.lora_int_id < 1:
            raise ValueError(
                f"lora_int_id must be > 0, got {self.lora_int_id}")

    def __eq__(self, value: object) -> bool:
        return isinstance(
            value, LoRARequest) and self.lora_int_id == value.lora_int_id

    def __hash__(self) -> int:
        return self.lora_int_id
