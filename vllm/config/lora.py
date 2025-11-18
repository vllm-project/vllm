# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.config.cache import CacheConfig
else:
    ModelConfig = Any
    CacheConfig = Any

logger = init_logger(__name__)

LoRADType = Literal["auto", "float16", "bfloat16"]
MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
LoRAExtraVocabSize = Literal[256, 512]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    """Configuration for LoRA."""

    max_lora_rank: MaxLoRARanks = 16
    """Max LoRA rank."""
    max_loras: int = Field(default=1, ge=1)
    """Max number of LoRAs in a single batch."""
    fully_sharded_loras: bool = False
    """By default, only half of the LoRA computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    max_cpu_loras: int | None = None
    """Maximum number of LoRAs to store in CPU memory. Must be >= than
    `max_loras`."""
    lora_dtype: torch.dtype | LoRADType = "auto"
    """Data type for LoRA. If auto, will default to base model dtype."""
    lora_extra_vocab_size: LoRAExtraVocabSize = Field(
        default=256,
        deprecated=(
            "`lora_extra_vocab_size` is deprecated and will be removed "
            "in v0.12.0. Additional vocabulary support for "
            "LoRA adapters is being phased out."
        ),
    )
    """(Deprecated) Maximum size of extra vocabulary that can be present in a 
    LoRA adapter. Will be removed in v0.12.0."""
    lora_vocab_padding_size: ClassVar[int] = (
        current_platform.get_lora_vocab_padding_size()
    )
    default_mm_loras: dict[str, str] | None = None
    """Dictionary mapping specific modalities to LoRA model paths; this field
    is only applicable to multimodal models and should be leveraged when a
    model always expects a LoRA to be active when a given modality is present.
    Note that currently, if a request provides multiple additional
    modalities, each of which have their own LoRA, we do NOT apply
    default_mm_loras because we currently only support one lora adapter
    per prompt. When run in offline mode, the lora IDs for n modalities
    will be automatically assigned to 1-n with the names of the modalities
    in alphabetic order."""

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

        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @model_validator(mode="after")
    def _validate_lora_config(self) -> Self:
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})"
            )

        return self

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
