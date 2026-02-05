# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from vllm.config.utils import CompileFactors, config, get_compile_factors
from vllm.logger import init_logger

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


@config(config=ConfigDict(arbitrary_types_allowed=True))
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
    enable_tower_connector_lora: bool = False
    """If `True`, LoRA support for the tower (vision encoder) and connector 
    of multimodal models will be enabled. This is an experimental feature and 
    currently only supports some MM models such as the Qwen VL series. The default 
    is False."""
    specialize_active_lora: bool = False
    """Whether to construct lora kernel grid by the number of active LoRA adapters.
    When set to True, separate cuda graphs will be captured for different counts
    of active LoRAs (powers of 2 up to max_loras), which can improve performance
    for variable LoRA usage patterns at the cost of increased startup time and
    memory usage. Only takes effect when cudagraph_specialize_lora is True.
    """

    def compile_factors(self) -> CompileFactors:
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
        ignored_factors = {
            # Runtime/placement only; does not affect compiled graph
            "max_cpu_loras",
            "default_mm_loras",
        }
        return get_compile_factors(self, ignored_factors)

    @model_validator(mode="after")
    def _validate_lora_config(self) -> Self:
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})."
            )

        return self

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
