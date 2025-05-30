# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config.utils import config
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config.cache_config import CacheConfig
    from vllm.config.model_config import ModelConfig

LoRADType = Literal["auto", "float16", "bfloat16"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    """Configuration for LoRA."""

    max_lora_rank: int = 16
    """Max LoRA rank."""
    max_loras: int = 1
    """Max number of LoRAs in a single batch."""
    fully_sharded_loras: bool = False
    """By default, only half of the LoRA computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    max_cpu_loras: Optional[int] = None
    """Maximum number of LoRAs to store in CPU memory. Must be >= than
    `max_loras`."""
    lora_dtype: Union[torch.dtype, LoRADType] = "auto"
    """Data type for LoRA. If auto, will default to base model dtype."""
    lora_extra_vocab_size: int = 256
    """Maximum size of extra vocabulary that can be present in a LoRA adapter
    (added to the base model vocabulary)."""
    lora_vocab_padding_size: ClassVar[int] = current_platform\
        .get_lora_vocab_padding_size()
    long_lora_scaling_factors: Optional[tuple[float, ...]] = None
    """Specify multiple scaling factors (which can be different from base model
    scaling factor - see eg. Long LoRA) to allow for multiple LoRA adapters
    trained with those scaling factors to be used at the same time. If not
    specified, only adapters trained with the base model scaling factor are
    allowed."""
    bias_enabled: bool = False
    """Enable bias for LoRA adapters."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.max_lora_rank)
        factors.append(self.max_loras)
        factors.append(self.fully_sharded_loras)
        factors.append(self.lora_dtype)
        factors.append(self.lora_extra_vocab_size)
        factors.append(self.lora_vocab_padding_size)
        factors.append(self.long_lora_scaling_factors)
        factors.append(self.bias_enabled)
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        # Setting the maximum rank to 512 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
        possible_lora_extra_vocab_size = (256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")

    def verify_with_cache_config(self, cache_config: "CacheConfig"):
        if cache_config.cpu_offload_gb > 0 and not envs.VLLM_USE_V1:
            raise ValueError(
                "V0 LoRA does not support CPU offload, please use V1.")

    def verify_with_model_config(self, model_config: "ModelConfig"):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)

    def verify_lora_support(self):
        if self.long_lora_scaling_factors is not None and envs.VLLM_USE_V1:
            raise ValueError(
                "V1 LoRA does not support long LoRA, please use V0.")
