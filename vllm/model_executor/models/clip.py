"""Minimal implementation of CLIPVisionModel intended to be only used
within a vision language model."""
from array import array
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionConfig
from xformers import ops as xops

from vllm.config import ModelConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData


def get_clip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0
    return image_size // patch_size


def get_clip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_clip_patch_grid_length(image_size=image_size,
                                             patch_size=patch_size)
    return grid_length * grid_length


def get_clip_image_feature_size(hf_config: CLIPVisionConfig) -> int:
    return get_clip_num_patches(image_size=hf_config.image_size,
                                patch_size=hf_config.patch_size) + 1


def get_max_clip_image_tokens(hf_config: CLIPVisionConfig) -> int:
    return get_clip_image_feature_size(hf_config)


def dummy_seq_data_for_clip(
    hf_config: CLIPVisionConfig,
    seq_len: int,
    num_images: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_clip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                      [image_token_id]) * image_feature_size * num_images
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                       [0]) * (seq_len - image_feature_size * num_images)
    return SequenceData(token_ids)


def dummy_image_for_clip(
    hf_config: CLIPVisionConfig,
    num_images: int,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}


def input_processor_for_clip(
    model_config: ModelConfig,
    hf_config: CLIPVisionConfig,
    llm_inputs: LLMInputs,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[Union[int, List[int]]] = None,
):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    if image_feature_size_override is None:
        image_data = multi_modal_data["image"]
        if isinstance(image_data, Image.Image):
            image_feature_size = get_clip_image_feature_size(hf_config)
        elif isinstance(image_data, torch.Tensor):
            image_feature_size = image_data.shape[0]
        else:
            raise TypeError(f"Invalid image type: {type(image_data)}")
    else:
        image_feature_size = image_feature_size_override

    new_prompt, new_token_ids = repeat_and_pad_placeholder_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        placeholder_token_id=image_token_id,
        repeat_count=image_feature_size,
    )

    # NOTE: Create a defensive copy of the original inputs
    return LLMInputs(prompt_token_ids=new_token_ids,
                     prompt=new_prompt,
                     multi_modal_data=multi_modal_data)


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py#L164 # noqa
class CLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = get_clip_num_patches(image_size=self.image_size,
                                                patch_size=self.patch_size)
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions,
                                               self.embed_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)

        query_states = query_states.view(bsz, tgt_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)
        key_states = key_states.view(bsz, tgt_len,
                                     self.num_heads_per_partition,
                                     self.head_dim)
        value_states = value_states.view(bsz, tgt_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)

        out = xops.memory_efficient_attention_forward(query_states,
                                                      key_states,
                                                      value_states,
                                                      p=self.dropout,
                                                      scale=self.scale)
        out = out.view(bsz, tgt_len, -1)
        attn_output, _ = self.out_proj(out)

        return attn_output


class CLIPMLP(nn.Module):

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config)
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()

        self.self_attn = CLIPAttention(config, quant_config=quant_config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config, quant_config=quant_config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
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

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 num_hidden_layers_override: Optional[int] = None):
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config=config, quant_config=quant_config)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor):

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class CLIPVisionTransformer(nn.Module):

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 num_hidden_layers_override: Optional[int] = None):
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
            num_hidden_layers_override=num_hidden_layers_override)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = self.encoder(inputs_embeds=hidden_states)

        return hidden_states


class CLIPVisionModel(nn.Module):

    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self,
                 config: CLIPVisionConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 num_hidden_layers_override: Optional[int] = None):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override)

    def forward(self, pixel_values: Optional[torch.Tensor] = None):

        return self.vision_model(pixel_values=pixel_values)

    @property
    def device(self):
        return next(self.parameters()).device

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is not needed in CLIPVisionModel
            if "vision_model.post_layernorm" in name:
                continue
            # omit layers when num_hidden_layers_override is set
            if "vision_model.encoder.layers." in name:
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                param = params_dict[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
