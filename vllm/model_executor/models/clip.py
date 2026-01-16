# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cached_property
from typing import Annotated, Literal

import torch
import torch.nn as nn
from transformers import (
    BatchFeature,
    CLIPConfig,
    CLIPProcessor,
    CLIPTextConfig,
    CLIPVisionConfig,
)

from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .interfaces_base import default_pooling_type
from .utils import AutoWeightsLoader, maybe_prefix
from .vision import (
    VisionEncoderInfo,
    VisionFeatureSelectStrategy,
    VisionFeatureSelectStrategyStr,
    get_num_selected_vision_tokens,
    resolve_visual_encoder_outputs,
)


class CLIPImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each image
        - w: Width of each image
    """

    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("bn", 3, "h", "w")]


class CLIPEncoderInfo(VisionEncoderInfo[CLIPVisionConfig]):
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return self.get_patch_grid_length() ** 2 + 1

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()
        assert image_size % patch_size == 0
        return image_size // patch_size


_POOLING_TYPE_TO_STRATEGY: dict[str, VisionFeatureSelectStrategyStr] = {
    "MEAN": "full",
    "ALL": "full",
    "CLS": "class",
    # This lets us use the same pooling type for both text and image
    "LAST": "class",
}


def _get_vision_feature_select_strategy(pooling_type: str):
    try:
        return _POOLING_TYPE_TO_STRATEGY[pooling_type]
    except KeyError:
        raise ValueError(
            f"No feature selection strategy is defined for "
            f"pooling_type: {pooling_type!r}"
        ) from None


class CLIPProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(CLIPConfig)

    def get_vision_encoder_info(self):
        return CLIPEncoderInfo(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(CLIPProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        vision_encoder_info = self.get_vision_encoder_info()

        pooler_config = self.ctx.model_config.pooler_config
        assert pooler_config is not None

        return get_num_selected_vision_tokens(
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
            _get_vision_feature_select_strategy(pooler_config.seq_pooling_type),
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )


class CLIPDummyInputsBuilder(BaseDummyInputsBuilder[CLIPProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class CLIPMultiModalProcessor(BaseMultiModalProcessor[CLIPProcessingInfo]):
    @cached_property
    def image_token_id(self) -> int:
        tokenizer = self.info.get_tokenizer()
        dummy_token_id = 0

        assert dummy_token_id not in tokenizer.all_special_ids

        return dummy_token_id

    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        if prompt and mm_data:
            raise ValueError(
                "CLIP accepts text-only or image-only inputs, not both! "
                "Image-only inputs means passing an image with an empty text "
                "prompt."
            )

        if mm_data:
            # For multi-modal data, the prompt after processing should
            # only contain the dummy image tokens
            tokenization_kwargs = {
                **(tokenization_kwargs or {}),
                "add_special_tokens": False,
            }

        return super().apply(
            prompt=prompt,
            mm_data=mm_data,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.image_token_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            num_image_tokens = self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
            )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=PromptIndexTargets.start(),
                replacement=get_replacement,
            ),
        ]


# Adapted from: https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clip/modeling_clip.py
class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()

        embed_dim = config.hidden_size

        self.token_embedding = VocabParallelEmbedding(config.vocab_size, embed_dim)
        self.position_embedding = VocabParallelEmbedding(
            config.max_position_embeddings, embed_dim
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "Either `input_ids` or `input_embeds` must be provided"
                )

            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        assert self.image_size % self.patch_size == 0

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = Conv2dLayer(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPAttention(nn.Module):
    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        *,
        prefix: str = "",
        attn_cls: type[Attention] | type[MMEncoderAttention],
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )

        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        if attn_cls == MMEncoderAttention:
            self.attn = attn_cls(
                self.num_heads_per_partition,
                self.head_dim,
                self.scale,
                prefix=f"{prefix}.attn",
                multimodal_config=multimodal_config,
            )
        else:
            self.attn = attn_cls(
                self.num_heads_per_partition,
                self.head_dim,
                self.scale,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        out = self.attn(query_states, key_states, value_states)
        attn_output, _ = self.out_proj(out)

        return attn_output, None


class CLIPMLP(nn.Module):
    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.activation_fn = get_act_fn(config.hidden_act)

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        *,
        prefix: str = "",
        attn_cls: type[Attention] | type[MMEncoderAttention],
    ) -> None:
        super().__init__()

        self.self_attn = CLIPAttention(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.self_attn",
            attn_cls=attn_cls,
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self
    attention layers. Each layer is a [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        config: CLIPTextConfig | CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        num_hidden_layers_override: int | None = None,
        *,
        prefix: str = "",
        attn_cls: type[Attention] | type[MMEncoderAttention],
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    attn_cls=attn_cls,
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> torch.Tensor | list[torch.Tensor]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPTextEmbeddings(config)

        self.encoder = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
            attn_cls=Attention,
        )

        self.final_layer_norm = nn.LayerNorm(
            embed_dim,
            eps=config.layer_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.token_embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=False,
        )
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        return last_hidden_state

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)

        # NOTE: This typo of "layrnorm" is not fixed on purpose to match
        # the original transformers code and name of the model weights.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.encoder = CLIPEncoder(
            config=config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
            attn_cls=MMEncoderAttention,
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have select_layers or not
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=select_layers is not None,
        )

        # Handle post-norm (if applicable) and stacks feature layers if needed
        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs,
            self.post_layernorm,
            select_layers=select_layers,
            max_possible_layers=self.config.num_hidden_layers,
            feature_select_strategy=feature_select_strategy,
        )

        return encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is not needed in CLIPVisionModel
            if name.startswith("post_layernorm") and self.post_layernorm is None:
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("encoder.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class CLIPVisionModel(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.vision_model = CLIPVisionTransformer(
            config=config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values,
            select_layers=select_layers,
            feature_select_strategy=feature_select_strategy,
        )

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def device(self):
        return self.vision_model.device


# Assume EOS token corresponds to LAST token in text model
@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    CLIPMultiModalProcessor,
    info=CLIPProcessingInfo,
    dummy_inputs=CLIPDummyInputsBuilder,
)
class CLIPEmbeddingModel(nn.Module, SupportsMultiModal, SupportsQuant):
    is_pooling_model = True

    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: CLIPConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(
            text_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "text_model"),
        )
        self.vision_model = CLIPVisionTransformer(
            vision_config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler_config = pooler_config

        self.pooler = DispatchPooler.for_embedding(pooler_config)

        # Assumes that self.forward is called after self.embed_input_ids
        self._is_text_input = True

    def get_text_features(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_output = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor:
        if feature_select_strategy is None:
            feature_select_strategy = _get_vision_feature_select_strategy(
                self.pooler_config.seq_pooling_type
            )

        pooled_output = self.vision_model(
            pixel_values=pixel_values,
            select_layers=None,
            feature_select_strategy=feature_select_strategy,
        )

        image_features = self.visual_projection(pooled_output)

        return image_features

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> CLIPImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        expected_h = expected_w = self.config.vision_config.image_size
        return CLIPImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            resolve_bindings={"h": expected_h, "w": expected_w},
        )

    def _process_image_inputs(self, inputs: CLIPImagePixelInputs) -> torch.Tensor:
        pixel_values = inputs["data"]

        return self.get_image_features(pixel_values)

    def get_language_model(self) -> torch.nn.Module:
        return self.text_model

    def _embed_text_input_ids(
        self,
        input_ids: torch.Tensor,
        embed_input_ids: Callable[[torch.Tensor], torch.Tensor],
        *,
        is_multimodal: torch.Tensor | None,
        handle_oov_mm_token: bool,
    ) -> torch.Tensor:
        inputs_embeds = super()._embed_text_input_ids(
            input_ids,
            embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        # NOTE: inputs_embeds in model runner has size text_config.projection_dim
        # (instead of text_config.hidden_size) to accommodate image embeddings
        inputs_embeds_size = self.projection_dim
        if inputs_embeds.shape[1] < inputs_embeds_size:
            inputs_embeds = torch.cat(
                [
                    inputs_embeds,
                    inputs_embeds.new_empty(
                        inputs_embeds.shape[0],
                        inputs_embeds_size - inputs_embeds.shape[1],
                    ),
                ],
                dim=1,
            )
        elif inputs_embeds.shape[1] > inputs_embeds_size:
            # No need to handle this case for now
            raise NotImplementedError

        return inputs_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        self._is_text_input = (
            multimodal_embeddings is None or len(multimodal_embeddings) == 0
        )

        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        vision_embeddings = self._process_image_inputs(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise RuntimeError("PP is not supported for this model")

        # Multimodal inputs
        if not self._is_text_input:
            return inputs_embeds

        # NOTE: inputs_embeds in model runner has size text_config.projection_dim
        # (instead of text_config.hidden_size) to accommodate image embeddings
        hidden_size = self.text_embed_dim
        if inputs_embeds.shape[1] > hidden_size:
            inputs_embeds = inputs_embeds[:, :hidden_size]
        elif inputs_embeds.shape[1] < hidden_size:
            # No need to handle this case for now
            raise NotImplementedError

        return self.get_text_features(input_ids, positions, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_substrs=[".position_ids"],
            ignore_unexpected_prefixes=["logit_scale."],
        )

        return loader.load_weights(weights)
