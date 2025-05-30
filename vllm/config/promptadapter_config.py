# SPDX-License-Identifier: Apache-2.0

import hashlib
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.config.utils import config

if TYPE_CHECKING:
    from vllm.config.model_config import ModelConfig


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PromptAdapterConfig:
    """Configuration for PromptAdapters."""

    max_prompt_adapters: int = 1
    """Max number of PromptAdapters in a batch."""
    max_prompt_adapter_token: int = 0
    """Max number of PromptAdapters tokens."""
    max_cpu_prompt_adapters: Optional[int] = None
    """Maximum number of PromptAdapters to store in CPU memory. Must be >= than
    `max_prompt_adapters`."""
    prompt_adapter_dtype: Union[torch.dtype, str] = "auto"
    """Data type for PromptAdapter. If auto, will default to base model dtype.
    """

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
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):

        if self.max_prompt_adapters < 1:
            raise ValueError(f"max_prompt_adapters "
                             f"({self.max_prompt_adapters}) must be >= 1.")
        if self.max_prompt_adapter_token == 0:
            raise ValueError("max_prompt_adapter_token must be set.")
        if self.max_cpu_prompt_adapters is None:
            self.max_cpu_prompt_adapters = self.max_prompt_adapters

    def verify_with_model_config(self, model_config: "ModelConfig"):
        if self.prompt_adapter_dtype == "auto":
            self.prompt_adapter_dtype = model_config.dtype
        elif isinstance(self.prompt_adapter_dtype, str):
            self.prompt_adapter_dtype = getattr(torch,
                                                self.prompt_adapter_dtype)
