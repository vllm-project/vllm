# Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py

from dataclasses import MISSING, dataclass, field, fields
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class PEFTHelper:
    # Required fields
    r: int
    lora_alpha: int
    target_modules: Union[list[str], str]

    # Optional fields
    bias: Literal["none", "all", "lora_only"] = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)
    use_rslora: bool = field(default=False)
    use_dora: bool = field(default=False)
    # long lora field
    context_length: int = field(default=0)

    # vllm extra filed
    max_position_embeddings: Optional[int] = field(default=False)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PEFTHelper":
        # Get all field information from the class
        class_fields = {f.name: f for f in fields(cls)}
        # Check for required fields
        required_fields = {
            name
            for name, f in class_fields.items()
            if f.default is MISSING and f.default_factory is MISSING
        }

        # Identify any missing required fields
        missing_fields = required_fields - set(config_dict.keys())
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {missing_fields}")

        # Filter out fields that aren't defined in the class
        filtered_dict = {
            k: v
            for k, v in config_dict.items() if k in class_fields
        }
        return cls(**filtered_dict)
