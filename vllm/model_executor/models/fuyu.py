# coding=utf-8
# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/fuyu/modeling_fuyu.py
# Copyright 2023 The vLLM team.
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Fuyu model."""
from typing import Iterable, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import FuyuConfig

from vllm.attention import AttentionMetadata
from vllm.config import VisionLanguageConfig
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               RowParallelLinear)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.persimmon import PersimmonForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


class FuyuInputProcessor:
    """Simple Processor for padding and patch PIXEL_VALUES image input"""

    def __init__(self, patch_size=30, bos_token_id=1) -> None:
        self.patch_size = patch_size
        self.bos_token_id = bos_token_id

    def _calculate_patch_nums(self, x: int) -> int:
        x = x // self.patch_size
        if x % self.patch_size != 0:
            x += 1
        return x

    def _patch_image(self, image):
        patches = list(image.split(self.patch_size, dim=0))
        patches = [
            torch.stack(patch.split(self.patch_size, dim=1))
            for patch in patches
        ]
        patches = torch.concat(patches, dim=0)
        return patches

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, H, W = image.shape
        assert batch_size == 1
        assert num_channels == 3, "Input image should have 3 channels"

        image = image.squeeze(0).permute(1, 2, 0)
        patch_y = self._calculate_patch_nums(H)
        patch_x = self._calculate_patch_nums(W)

        image_padded = torch.ones(
            patch_y * self.patch_size,
            patch_x * self.patch_size,
            num_channels,
            dtype=image.dtype,
            device=image.device,
        )
        image_padded[:H, :W, :] = image
        image_patches = self._patch_image(image_padded).view(
            patch_y * patch_x, self.patch_size**2 * num_channels)
        return image_patches


class FuyuForCausalLM(nn.Module):

    def __init__(
        self,
        config: FuyuConfig,
        vision_language_config: VisionLanguageConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vision_language_config = vision_language_config
        self.image_token_id = self.vision_language_config.image_token_id

        self.processor = FuyuInputProcessor()
        self.vision_embed_tokens = RowParallelLinear(
            config.patch_size * config.patch_size * config.num_channels,
            config.hidden_size,
            linear_method=linear_method,
        )
        self.language_model = PersimmonForCausalLM(config,
                                                   linear_method=linear_method)

    def merge_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:
        mask = input_ids == image_token_id
        inputs_embeds[mask] = vision_embeddings.view(
            -1, vision_embeddings.shape[-1])
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        image_input: Optional[torch.Tensor] = None,
    ):
        if image_input is not None:
            # input_ids, image_patches = self.processor(input_ids, image_input)
            image_patches = self.processor(image_input)
            vision_embeddings, _ = self.vision_embed_tokens(
                image_patches.to(self.vision_embed_tokens.weight.dtype))
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            inputs_embeds = self.merge_embeddings(
                input_ids,
                inputs_embeds,
                vision_embeddings,
                image_token_id=self.image_token_id,
            )

        else:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.language_model.logits_processor(
            self.language_model.lm_head.weight, hidden_states,
            sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            param = params_dict[name]

            if "query_key_value" in name:
                # copy from vllm/model_executor/models/bloom.py
                # NOTE: Fuyu's fused QKV's output_dim has the shape of
                # (num_heads * 3 * head_size), while the
                # required shape is (3 * num_heads * head_size).
                # Thus, we need weight conversion.
                output_dim = getattr(param, "output_dim", None)
                num_heads = self.config.num_attention_heads
                if output_dim is not None:
                    loaded_weight_shape = loaded_weight.shape
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim] + (num_heads, 3, -1) +
                        loaded_weight_shape[output_dim + 1:])
                    loaded_weight = loaded_weight.transpose(
                        output_dim, output_dim + 1)
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
