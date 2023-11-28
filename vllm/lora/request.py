from dataclasses import dataclass


@dataclass
class LoRARequest:
    lora_id: str
    lora_int_id: int
    lora_local_path: str

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(
                f"lora_int_id must be > 0, got {self.lora_int_id}")

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LoRARequest) and self.lora_id == value.lora_id

    def __hash__(self) -> int:
        return self.lora_int_id
