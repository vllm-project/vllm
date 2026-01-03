# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
# https://huggingface.co/microsoft/kosmos-2.5/resolve/main/config.json
"""PyTorch KOSMOS-2.5 model."""

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn
from transformers import BatchFeature
from transformers.models.kosmos2_5 import Kosmos2_5Config, Kosmos2_5Processor
from transformers.models.kosmos2_5.modeling_kosmos2_5 import (
    Kosmos2_5ImageToTextProjection,
    Kosmos2_5TextConfig,
    Kosmos2_5VisionModel,
)

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors


# Copied from transformers.models.kosmos2.modeling_kosmos2
# .Kosmos2TextSinusoidalPositionalEmbedding with Kosmos2->Kosmos2_5
class Kosmos2_5TextSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    # Copied from transformers.models.m2m_100.modeling_m2m_100
    # .M2M100SinusoidalPositionalEmbedding.__init__
    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: int | None = None
    ):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # Copied from transformers.models.m2m_100.modeling_m2m_100
    # .M2M100SinusoidalPositionalEmbedding.make_weights
    def make_weights(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
    ):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(
                dtype=self.weights.dtype, device=self.weights.device
            )

        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    # Copied from transformers.models.m2m_100.modeling_m2m_100
    # .M2M100SinusoidalPositionalEmbedding.get_embedding
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
    ):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs
        slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        if input_ids is not None:
            total_tokens_length = input_ids.size(0)
            if position_ids is None:
                position_ids = torch.arange(
                    self.padding_idx + 1,
                    total_tokens_length + self.padding_idx + 1,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            else:
                position_ids = position_ids + self.padding_idx + 1
        else:
            total_tokens_length = inputs_embeds.size(0)
            if position_ids is None:
                position_ids = torch.arange(
                    self.padding_idx + 1,
                    total_tokens_length + self.padding_idx + 1,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            else:
                position_ids = position_ids + self.padding_idx + 1

        return self.weights.index_select(0, position_ids).detach()


# Copied from transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextFFN
# with Kosmos2->Kosmos2_5
class Kosmos2_5TextFFN(nn.Module):
    def __init__(
        self,
        config: Kosmos2_5TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.activation_fn = get_act_fn(config.activation_function)

        self.fc1 = ColumnParallelLinear(
            input_size=config.embed_dim,
            output_size=config.ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )

        self.fc2 = RowParallelLinear(
            input_size=config.ffn_dim,
            output_size=config.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

        self.ffn_layernorm = nn.LayerNorm(config.ffn_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class Kosmos2_5TextAttention(nn.Module):
    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        layer_idx: int | None = None,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = num_heads // self.tp_size
        self.num_kv_heads_per_partition = num_heads // self.tp_size  # MHA

        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.attn = Attention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads_per_partition,
            cache_config=vllm_config.cache_config if vllm_config else None,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        q_size = self.num_heads_per_partition * self.head_dim
        query, key, value = qkv.split([q_size, q_size, q_size], dim=-1)

        attn_output = self.attn(
            query=query,
            key=key,
            value=value,
        )

        attn_output, _ = self.out_proj(attn_output)

        return attn_output


class Kosmos2_5TextBlock(nn.Module):
    def __init__(
        self,
        config: Kosmos2_5TextConfig,
        layer_idx: int,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.layer_idx = layer_idx
        self.self_attn = Kosmos2_5TextAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            layer_idx=layer_idx,
            vllm_config=vllm_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps
        )
        self.ffn = Kosmos2_5TextFFN(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Kosmos2_5TextTransformer(nn.Module):
    def __init__(
        self,
        config: Kosmos2_5TextConfig,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layerdrop = config.layerdrop

        self.embed_scale = (
            math.sqrt(config.embed_dim) if config.scale_embedding else 1.0
        )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )

        self.embed_positions = Kosmos2_5TextSinusoidalPositionalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
        )

        self.segment_emb = nn.Embedding(2, config.embed_dim)
        self.layers = nn.ModuleList(
            [
                Kosmos2_5TextBlock(
                    config,
                    layer_idx,
                    vllm_config=vllm_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        image_embeds_position_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at "
                "the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embed_scale

        positions = self.embed_positions(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        positions = positions.to(inputs_embeds.device)

        if image_embeds_position_mask is not None:
            image_embeds_position_mask = image_embeds_position_mask.ne(0).long()
            segment_embeds = self.segment_emb(image_embeds_position_mask).to(
                positions.device
            )
            positions += segment_embeds
        else:
            total_tokens_length = positions.size(0)
            zero_emb = self.segment_emb(
                torch.zeros(
                    (total_tokens_length,),
                    dtype=torch.long,
                    device=self.segment_emb.weight.device,
                )
            ).to(positions.device)
            positions += zero_emb

        hidden_states = inputs_embeds + positions

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class Kosmos2_5ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Kosmos2_5Config)

    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(Kosmos2_5Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.ctx.get_hf_config(Kosmos2_5Config)
        return hf_config.latent_query_num


class Kosmos2_5MultiModalProcessor(BaseMultiModalProcessor[Kosmos2_5ProcessingInfo]):
    def validate_num_items(
        self,
        modality: str,
        num_items: int,
    ) -> None:
        super().validate_num_items(modality, num_items)

        if modality == "image" and num_items < 1:
            raise ValueError(
                f"Kosmos2.5 requires at least 1 image per prompt, "
                f"but {num_items} image(s) were provided."
            )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            flattened_patches=MultiModalFieldConfig.batched("image"),
            image_embeds_position_mask=MultiModalFieldConfig.batched("image"),
        )

    def _compress_to_image_tags(self, processed_outputs: BatchFeature) -> BatchFeature:
        input_ids = processed_outputs["input_ids"]
        mask = processed_outputs.get("image_embeds_position_mask")

        if mask is None:
            return processed_outputs

        keep_mask = mask[0] != 1
        compressed_ids = input_ids[0][keep_mask]

        processed_outputs["input_ids"] = compressed_ids.unsqueeze(0)
        processed_outputs.pop("image_embeds_position_mask", None)

        return processed_outputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        hf_processor = self.info.get_hf_processor()
        tokenizer = hf_processor.tokenizer

        image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
        image_start_token_id = tokenizer.convert_tokens_to_ids(
            hf_processor.image_start_token
        )
        image_end_token_id = tokenizer.convert_tokens_to_ids(
            hf_processor.image_end_token
        )

        return [
            PromptReplacement(
                modality="image",
                target=[image_start_token_id, image_end_token_id],
                replacement=PromptUpdateDetails.select_token_id(
                    [image_start_token_id]
                    + [image_token_id] * hf_config.latent_query_num
                    + [image_end_token_id],
                    embed_token_id=image_token_id,
                ),
            ),
        ]

    def _apply_hf_processor_text_only(
        self,
        prompt_text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        dummy_image = self.dummy_inputs._get_dummy_images(
            width=224, height=224, num_images=1
        )

        hf_processor = self.info.get_hf_processor()
        processed_data = self.info.ctx.call_hf_processor(
            hf_processor=hf_processor,
            data={"text": prompt_text, "images": dummy_image},
            kwargs=tokenization_kwargs,
        )

        # Kosmos2_5 need at least one image for the tokenizer to work
        # so we put the dummy image to the tokenizer to make it work
        # we also compress the image tokens to behave like a text only prompt
        processed_data = self._compress_to_image_tags(processed_data)
        (prompt_ids,) = processed_data.pop("input_ids").tolist()

        return prompt_ids

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ):
        mm_counts = mm_items.get_all_counts()
        dummy_text = self.dummy_inputs.get_dummy_text(mm_counts)

        if "image" in mm_items and len(mm_items["image"]) > 0:
            image_items = mm_items["image"]
            hf_mm_data = image_items.get_processor_data()
        else:
            dummy_image = self.dummy_inputs._get_dummy_images(
                width=224, height=224, num_images=1
            )
            hf_mm_data = {"images": dummy_image}

        hf_processor = self.info.get_hf_processor()
        mm_processed_data = self.info.ctx.call_hf_processor(
            hf_processor=hf_processor,
            data={"text": dummy_text, **hf_mm_data},
            kwargs={**hf_processor_mm_kwargs, **tokenization_kwargs},
        )

        mm_processed_data.pop("input_ids", None)

        return mm_processed_data


class Kosmos2_5DummyInputsBuilder(BaseDummyInputsBuilder[Kosmos2_5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<md>"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        return {
            "image": self._get_dummy_images(
                width=224,
                height=224,
                num_images=mm_counts.get("image", 0),
            )
        }


@MULTIMODAL_REGISTRY.register_processor(
    Kosmos2_5MultiModalProcessor,
    info=Kosmos2_5ProcessingInfo,
    dummy_inputs=Kosmos2_5DummyInputsBuilder,
)
class Kosmos2_5ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "text_model.model.": "text_model.",
            "text_model.lm_head.": "lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        # TODO: vision encoder and image to text projection should be moved to vllm
        # so that attention implementation can be controlled by vllm config
        if (
            not hasattr(config.vision_config, "_attn_implementation")
            or config.vision_config._attn_implementation is None
        ):
            config.vision_config._attn_implementation = "sdpa"
        if (
            not hasattr(config, "_attn_implementation")
            or config._attn_implementation is None
        ):
            config._attn_implementation = "sdpa"

        self.vision_model = Kosmos2_5VisionModel(config.vision_config)
        self.image_to_text_projection = Kosmos2_5ImageToTextProjection(config)
        self.text_model = Kosmos2_5TextTransformer(
            config.text_config,
            vllm_config=vllm_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "text_model"),
        )

        self.lm_head = nn.Linear(
            config.text_config.embed_dim, config.text_config.vocab_size, bias=False
        )
        self.lm_head.weight = self.text_model.embed_tokens.weight

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {"hidden_states": torch.tensor([], dtype=dtype, device=device)}
        )

    def get_language_model(self) -> nn.Module:
        return self.text_model

    def get_multimodal_embeddings(self, **kwargs) -> torch.Tensor | None:
        flattened_patches = kwargs.pop("flattened_patches", None)
        if flattened_patches is None:
            return None

        flattened_patches = flattened_patches.view(
            flattened_patches.shape[0], -1, flattened_patches.shape[-1]
        )

        vision_outputs = self.vision_model(flattened_patches=flattened_patches)
        image_features = vision_outputs.last_hidden_state
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        image_embeds, _ = self.image_to_text_projection(image_features)

        return image_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        embed_fn = self.text_model.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        if is_multimodal is None:
            raise ValueError(
                "`get_input_embeddings` now requires `is_multimodal` arg, "
                "please update your model runner according to "
                "https://github.com/vllm-project/vllm/pull/16229."
            )

        # Merge text and image embeddings
        merged_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return merged_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Generate image_embeds_position_mask dynamically based on positions
        # TODO: this is a hack and should be removed in the future
        # it is ideal to be passed in as a kwarg or something
        image_embeds_position_mask = None
        if inputs_embeds is not None and len(positions) > 2048:
            image_embeds_position_mask = torch.zeros_like(positions, dtype=torch.long)
            image_mask = (positions >= 2) & (positions <= 2049)
            image_embeds_position_mask[image_mask] = 1

        hidden_states = self.text_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_embeds_position_mask=image_embeds_position_mask,
            position_ids=positions,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Define stacked params mapping for QKVParallelLinear
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Apply HF to vLLM mapping first
        if self.hf_to_vllm_mapper:
            weights = self.hf_to_vllm_mapper.apply(weights)

        for name, loaded_weight in weights:
            # Handle stacked QKV projections
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Regular parameter loading
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
