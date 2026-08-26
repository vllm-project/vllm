# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform

from .base import BaseLayerWithLoRA


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.org_vocab_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                self.base_layer.embedding_dim,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked_2d = self.lora_a_stacked.view(
            self.lora_a_stacked.shape[0] * self.lora_a_stacked.shape[1],
            self.lora_a_stacked.shape[2],
        )

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        assert isinstance(lora_a, torch.Tensor)
        assert isinstance(lora_b, torch.Tensor)
        self.reset_lora(index)
        # NOTE self.lora_a_stacked is row-major, and lora_a is col-major,
        # so we need transpose here

        self.lora_a_stacked[index, : lora_a.shape[1], : lora_a.shape[0]].copy_(
            lora_a.T, non_blocking=True
        )
        self.lora_b_stacked[index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(
            lora_b, non_blocking=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NB: Don't use torch.narrow here. torch.narrow triggers some
        # Dynamic Shape specialization in torch.compile
        num_tokens = x.shape[0]
        embeddings_indices = self.punica_wrapper._embeddings_indices[:num_tokens]

        full_lora_a_embeddings = F.embedding(
            x + embeddings_indices,
            self.lora_a_stacked_2d,
        )
        full_output = self.base_layer.forward(x)

        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(
                full_output.shape[0] * full_output.shape[1], -1
            )
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(
                full_lora_a_embeddings.shape[0] * full_lora_a_embeddings.shape[1],
                -1,
            )

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_embedding(
            full_output, full_lora_a_embeddings, self.lora_b_stacked, add_input=True
        )

        if not current_platform.can_update_inplace():
            full_output = lora_output

        return full_output.view_as(full_output_org)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # Accept both the in-tree and any OOT class, because layers built on top of
        # `VocabParallelEmbedding` (e.g. by the Transformers modelling backend) use the
        # former while `PluggableLayer.__new__` instantiates the latter
        embedding_classes = (
            VocabParallelEmbedding,
            maybe_get_oot_by_class(VocabParallelEmbedding),
        )
        lm_head_classes = (ParallelLMHead, maybe_get_oot_by_class(ParallelLMHead))
        return isinstance(source_layer, embedding_classes) and not isinstance(
            source_layer, lm_head_classes
        )

    @property
    def weight(self):
        return self.base_layer.weight
