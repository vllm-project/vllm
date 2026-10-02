# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py

import json
import math
import os
import re
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PEFTHelper:
    """A helper class for PEFT configurations, specifically designed for LoRA.
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
    # PEFT features that change the adapter's semantics and that vLLM does not
    # implement. They are read only to reject such adapters.
    lora_bias: bool = field(default=False)
    init_lora_weights: bool | str = field(default=True)
    alora_invocation_tokens: list[int] | None = field(default=None)
    layer_replication: list[list[int]] | None = field(default=None)
    use_bdlora: dict | None = field(default=None)
    use_qalora: bool = field(default=False)
    # Per-module overrides of `r` and `lora_alpha`, keyed by module name pattern
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, float] = field(default_factory=dict)
    # Extra vllm field, start with 'vllm_' to avoid conflict
    vllm_lora_scaling_factor: float = field(default=1.0)
    vllm_max_position_embeddings: int | None = field(default=False)

    def _validate_features(self) -> list[str]:
        """Check if there are any unsupported LoRA features."""
        error_msg = []
        if self.modules_to_save:
            unsupported_modules = [
                m for m in self.modules_to_save if m not in ["classifier", "score"]
            ]
            if unsupported_modules:
                error_msg.append(
                    "vLLM only supports modules_to_save being either None "
                    'or ["classifier", "score"] for classification models. '
                    f"Unsupported modules_to_save: {unsupported_modules}"
                )
        if self.use_dora:
            error_msg.append("vLLM does not yet support DoRA.")
        if self.lora_bias:
            error_msg.append("vLLM does not support LoRA bias (lora_bias).")
        if not self._init_is_supported():
            error_msg.append(
                f"vLLM does not support init_lora_weights={self.init_lora_weights!r}."
                " Inits such as PiSSA, OLoRA, CorDA and LoftQ modify the base model"
                " weights when PEFT loads the adapter. Convert it to a regular LoRA"
                " adapter with PEFT's path_initial_model_for_weight_conversion, or"
                " set init_lora_weights to true if the served base model already"
                " contains the modified weights."
            )
        if self.alora_invocation_tokens:
            error_msg.append("vLLM does not support Activated LoRA (aLoRA).")
        if self.layer_replication:
            error_msg.append("vLLM does not support layer_replication.")
        if self.use_bdlora:
            error_msg.append("vLLM does not support block-diagonal LoRA (BD-LoRA).")
        if self.use_qalora:
            error_msg.append("vLLM does not support QALoRA.")
        return error_msg

    def _init_is_supported(self) -> bool:
        # These inits only set the initial adapter weights, so a saved adapter
        # loads in PEFT as a plain LoRA. lora_ga modifies the base weights only
        # during training, when its gradients are attached.
        init = self.init_lora_weights
        return not isinstance(init, str) or init.lower() in (
            "gaussian",
            "eva",
            "orthogonal",
            "mica",
            "lora_ga",
        )

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank `r` must be a positive integer, got {self.r}.")
        self.rank_pattern = self.rank_pattern or {}
        self.alpha_pattern = self.alpha_pattern or {}
        if self.use_rslora:
            logger.info_once("Loading LoRA weights trained with rsLoRA.")
        self.vllm_lora_scaling_factor = self._scaling(self.r, self.lora_alpha)

    def _scaling(self, rank: int, alpha: float) -> float:
        if self.use_rslora:
            return alpha / math.sqrt(rank)
        return alpha / rank

    @staticmethod
    def _match_pattern(pattern: dict, name: str):
        # Same rule as PEFT's `get_pattern_key`: a key matches if it is a suffix
        # of the module name starting at a "." boundary; the first match wins.
        for key, value in pattern.items():
            if re.match(rf"(.*\.)?({key})$", name):
                return value
        return None

    def get_rank_and_scaling(self, module_name: str) -> tuple[int, float]:
        """Return the rank and scaling PEFT uses for `module_name`, taking
        `rank_pattern` and `alpha_pattern` into account.

        `module_name` is the name in the adapter checkpoint, without the
        `base_model.model.` prefix and before any weights mapping.
        """
        if not self.rank_pattern and not self.alpha_pattern:
            return self.r, self.vllm_lora_scaling_factor
        # PEFT stores fused MoE expert LoRAs (`target_parameters`) as
        # `experts.base_layer` (gate_up_proj) and `experts` (down_proj), but
        # matches the patterns against the parameter names.
        if module_name.endswith(".experts.base_layer"):
            module_name = module_name.removesuffix(".base_layer") + ".gate_up_proj"
        elif module_name.endswith(".experts"):
            module_name += ".down_proj"
        rank = self._match_pattern(self.rank_pattern, module_name) or self.r
        alpha = self._match_pattern(self.alpha_pattern, module_name)
        if alpha is None:
            alpha = self.lora_alpha
        return rank, self._scaling(rank, alpha)

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
    ) -> "PEFTHelper":
        lora_config_path = os.path.join(lora_path, "adapter_config.json")
        with open(lora_config_path) as f:
            config = json.load(f)

        config["vllm_max_position_embeddings"] = max_position_embeddings
        return cls.from_dict(config)

    def validate_legal(self, lora_config: LoRAConfig) -> None:
        """Validates the LoRA configuration settings against application
        constraints and requirements.
        """
        error_msg = self._validate_features()
        max_rank = max([self.r, *self.rank_pattern.values()])
        if max_rank > lora_config.max_lora_rank:
            error_msg.append(
                f"LoRA rank {max_rank} is greater than max_lora_rank"
                f" {lora_config.max_lora_rank}."
            )
        if self.bias != "none":
            error_msg.append("Adapter bias is not supported.")
        if error_msg:
            raise ValueError(f"{' '.join(error_msg)}")
