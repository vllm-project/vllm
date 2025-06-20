# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py

import json
import math
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal, Optional, Union

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

logger = init_logger(__name__)


@dataclass
class PEFTHelper:
    """ 
    A helper class for PEFT configurations, specifically designed for LoRA.
    This class handles configuration validation, compatibility checks for 
    various LoRA implementations.
    """

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

    def _validate_features(self) -> list[str]:
        """
        Check if there are any unsupported LoRA features.
        """
        error_msg = []
        if self.modules_to_save:
            error_msg.append("vLLM only supports modules_to_save being None.")
        if self.use_dora:
            error_msg.append("vLLM does not yet support DoRA.")
        return error_msg

    def __post_init__(self):
        if self.use_rslora:
            logger.info_once("Loading LoRA weights trained with rsLoRA.")
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

    @classmethod
    def from_local_dir(
            cls,
            lora_path: str,
            max_position_embeddings: Optional[int],
            tensorizer_config_dict: Optional[dict] = None) -> "PEFTHelper":
        lora_config_path = os.path.join(lora_path, "adapter_config.json")

        if tensorizer_config_dict:
            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            from tensorizer.stream_io import open_stream
            lora_config_path = os.path.join(tensorizer_config.lora_dir,
                                            "adapter_config.json")
            with open_stream(lora_config_path,
                             mode="rb",
                             **tensorizer_args.stream_params) as f:
                config = json.load(f)

            logger.info("Successfully deserialized LoRA config from %s",
                        tensorizer_config.lora_dir)

        else:
            with open(lora_config_path) as f:
                config = json.load(f)

        config["vllm_max_position_embeddings"] = max_position_embeddings
        return cls.from_dict(config)

    def validate_legal(self, lora_config: LoRAConfig) -> None:
        """
        Validates the LoRA configuration settings against application 
        constraints and requirements.
        """
        error_msg = self._validate_features()
        if self.r > lora_config.max_lora_rank:
            error_msg.append(
                f"LoRA rank {self.r} is greater than max_lora_rank"
                f" {lora_config.max_lora_rank}.")
        if self.bias != "none" and not lora_config.bias_enabled:
            error_msg.append(
                "Adapter bias cannot be used without bias_enabled.")
        if error_msg:
            raise ValueError(f"{' '.join(error_msg)}")
