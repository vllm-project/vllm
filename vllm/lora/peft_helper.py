# Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py

import math
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal, Optional, Union

from vllm.utils import print_info_once


@dataclass
class PEFTHelper:
    # Required fields
    r: int
    lora_alpha: int
    target_modules: Union[list[str], str]

    bias: Literal["none", "all", "lora_only"] = field(default="none")
    modules_to_save: Optional[list[str]] = field(default=None)
    # True to use Rank-Stabilized LoRA (rsLoRA, see: https://arxiv.org/abs/2312.03732)
    use_rslora: bool = field(default=False)
    # True to use Weight-Decomposed Low-Rank Adaptation (DoRA, see: https://arxiv.org/abs/2402.09353)
    use_dora: bool = field(default=False)
    # long context lora field
    context_length: int = field(default=0)
    # Extra vllm field, start with 'vllm_' to avoid conflict
    vllm_lora_scaling_factor: float = field(default=1.0)
    vllm_max_position_embeddings: Optional[int] = field(default=False)
    vllm_long_context_scaling_factor: Optional[float] = field(default=None)

    def _validate_features(self):
        error_msg = []

        if self.modules_to_save:
            error_msg.append("vLLM only supports modules_to_save being None.")

        if self.use_dora:
            error_msg.append("vLLM does not yet support DoRA.")

        if error_msg:
            raise ValueError(f"{', '.join(error_msg)}")

    def __post_init__(self):
        self._validate_features()
        if self.use_rslora:
            print_info_once("Loading LoRA weights trained with rsLoRA.")
            self.vllm_lora_scaling_factor = self.lora_alpha / math.sqrt(self.r)
        else:
            self.vllm_lora_scaling_factor = self.lora_alpha / self.r
        if self.context_length:
            if self.vllm_max_position_embeddings is None:
                self.vllm_max_position_embeddings = self.context_length
            self.vllm_long_context_scaling_factor = float(
                math.ceil(self.context_length /
                          self.vllm_max_position_embeddings))

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
