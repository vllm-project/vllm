# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from typing import Any, Optional, cast

from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config.model_config import ModelConfig
from vllm.config.utils import config, get_field


@config
@dataclass
class MultiModalConfig:
    """Controls the behavior of multimodal models."""

    limit_per_prompt: dict[str, int] = \
        cast(dict[str, int], get_field(ModelConfig, "limit_mm_per_prompt"))
    """
    The maximum number of input items allowed per prompt for each modality.
    Defaults to 1 (V0) or 999 (V1) for each modality.

    For example, to allow up to 16 images and 2 videos per prompt:
    `{"images": 16, "videos": 2}`
    """

    mm_processor_kwargs: Optional[dict[str, object]] = None
    """
    Overrides for the multi-modal processor obtained from
    `transformers.AutoProcessor.from_pretrained`.

    The available overrides depend on the model that is being run.

    For example, for Phi-3-Vision:
    `{"num_crops": 4}`.
    """

    disable_mm_preprocessor_cache: bool = False
    """
    If `True`, disable caching of the processed multi-modal inputs.
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

    def get_limit_per_prompt(self, modality: str) -> int:
        """
        Get the maximum number of input items allowed per prompt
        for the given modality.
        """
        return self.limit_per_prompt.get(
            modality,
            999 if envs.VLLM_USE_V1 else 1,
        )

    # TODO: Add configs to init vision tower or not.
