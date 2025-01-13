"""Request class for LoRA adapters."""
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class LoRARequest:
    """A request to load and use a LoRA adapter.
    
    Args:
        lora_name: Name of the LoRA adapter.
        lora_int_id: Integer ID for the adapter.
        lora_path: Local filesystem path or URI (s3:// or local://) to the adapter.
    """
    lora_name: str
    lora_int_id: int
    lora_path: str
    
    def __post_init__(self):
        """Validate request attributes."""
        if not isinstance(self.lora_name, str):
            raise ValueError("lora_name must be a string")
        if not isinstance(self.lora_int_id, int):
            raise ValueError("lora_int_id must be an integer")
        if not isinstance(self.lora_path, str):
            raise ValueError("lora_path must be a string")
            
    def __eq__(self, other):
        """Compare requests based on lora_int_id."""
        if not isinstance(other, LoRARequest):
            return NotImplemented
        return self.lora_int_id == other.lora_int_id
