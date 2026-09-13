# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py

import json
import math
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal

from vllm.config.lora import LoRAConfig
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
    target_modules: list[str] | str

    bias: Literal["none"] = field(default="none")
    modules_to_save: list[str] | None = field(default=None)
    # True to use Rank-Stabilized LoRA (rsLoRA, see: https://arxiv.org/abs/2312.03732)
    use_rslora: bool = field(default=False)
    # True to use Weight-Decomposed Low-Rank Adaptation (DoRA, see: https://arxiv.org/abs/2402.09353)
    use_dora: bool = field(default=False)
    trainable_token_indices: list[int] | dict[str, list[int]] | None = None
    ensure_weight_tying: bool = False
    # Extra vllm field, start with 'vllm_' to avoid conflict
    vllm_lora_scaling_factor: float = field(default=1.0)
    vllm_max_position_embeddings: int | None = field(default=False)

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
        if self.r <= 0:
            raise ValueError(f"LoRA rank `r` must be a positive integer, got {self.r}.")
        self._validate_trainable_token_indices()
        if self.use_rslora:
            logger.info_once("Loading LoRA weights trained with rsLoRA.")
            self.vllm_lora_scaling_factor = self.lora_alpha / math.sqrt(self.r)
        else:
            self.vllm_lora_scaling_factor = self.lora_alpha / self.r

    def _validate_trainable_token_indices(self) -> None:
        indices = self.trainable_token_indices
        if type(self.ensure_weight_tying) is not bool:
            raise ValueError("ensure_weight_tying must be a boolean.")
        if indices is None:
            return
        if isinstance(indices, dict) and (
            not indices or any(not isinstance(key, str) or not key for key in indices)
        ):
            raise ValueError("trainable_token_indices must have nonempty module names.")
        token_lists = indices.values() if isinstance(indices, dict) else [indices]
        for token_ids in token_lists:
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or any(
                    type(token_id) is not int or token_id < 0 for token_id in token_ids
                )
                or len(set(token_ids)) != len(token_ids)
            ):
                raise ValueError(
                    "trainable_token_indices must contain nonempty lists of unique "
                    "nonnegative integer token IDs."
                )

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
            raise ValueError(f"Missing required configuration fields: {missing_fields}")

        # Filter out fields that aren't defined in the class
        filtered_dict = {k: v for k, v in config_dict.items() if k in class_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_local_dir(
        cls,
        lora_path: str,
        max_position_embeddings: int | None,
        tensorizer_config_dict: dict | None = None,
    ) -> "PEFTHelper":
        lora_config_path = os.path.join(lora_path, "adapter_config.json")

        if tensorizer_config_dict:
            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            from tensorizer.stream_io import open_stream

            tensorizer_dir = tensorizer_config.tensorizer_dir
            if tensorizer_dir is None:
                raise ValueError("tensorizer_dir must be set in tensorizer config.")

            lora_config_path = os.path.join(tensorizer_dir, "adapter_config.json")
            with open_stream(
                lora_config_path, mode="rb", **tensorizer_args.stream_kwargs
            ) as f:
                config = json.load(f)

            logger.info(
                "Successfully deserialized LoRA config from %s",
                tensorizer_config.tensorizer_dir,
            )

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
                f" {lora_config.max_lora_rank}."
            )
        if self.bias != "none":
            error_msg.append("Adapter bias is not supported.")
        if self.trainable_token_indices is not None:
            token_lists = (
                self.trainable_token_indices.values()
                if isinstance(self.trainable_token_indices, dict)
                else [self.trainable_token_indices]
            )
            num_tokens = max(len(indices) for indices in token_lists)
            if num_tokens > lora_config.max_lora_trainable_tokens:
                error_msg.append(
                    f"Adapter requires {num_tokens} trainable tokens per module, "
                    "exceeding max_lora_trainable_tokens="
                    f"{lora_config.max_lora_trainable_tokens}. Set "
                    f"--max-lora-trainable-tokens to at least {num_tokens} to enable "
                    "selected-token weights."
                )
        if error_msg:
            raise ValueError(f"{' '.join(error_msg)}")
