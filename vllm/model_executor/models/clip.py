"""Minimal implementation of CLIPVisionModel intended to be only used
within a vision language model."""
from typing import Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionConfig

from vllm.attention.layer import MultiHeadAttention
from vllm.config import ModelConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import DecoderOnlyInputs, token_inputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   consecutive_placeholder_ranges,
                                   repeat_and_pad_placeholder_tokens,
                                   resolve_visual_encoder_outputs)
from vllm.sequence import SequenceData


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


def dummy_seq_data_for_clip(hf_config: CLIPVisionConfig,
                            seq_len: int,
                            num_images: int,
                            *,
                            image_token_id: int,
                            image_feature_size_override: Optional[int] = None,
                            mm_key: str = "image"):
    if image_feature_size_override is None:
        image_feature_size = get_clip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    return SequenceData.from_prompt_token_counts(
        (image_token_id, image_feature_size * num_images),
        (0, seq_len - image_feature_size * num_images),
    ), {
        mm_key:
        consecutive_placeholder_ranges(num_items=num_images,
                                       item_size=image_feature_size)
    }


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


def dummy_video_for_clip(
    hf_config: CLIPVisionConfig,
    num_frames: int,
    num_videos: int = 1,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    pil_frame = dummy_image_for_clip(
        hf_config,
        num_images=1,
        image_width_override=image_width_override,
        image_height_override=image_height_override)
    np_frame = np.array(pil_frame["image"])
    mm_data_per_video = np.repeat([np_frame], num_frames, axis=0)
    video_data = [mm_data_per_video] * num_videos
    mm_data = {"video": video_data}
    return mm_data


def input_processor_for_clip(
    model_config: ModelConfig,
    hf_config: CLIPVisionConfig,
    inputs: DecoderOnlyInputs,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[Union[int, List[int]]] = None,
):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    if "multi_modal_placeholders" in inputs and "image" in inputs[
            "multi_modal_placeholders"]:
        # The inputs already have placeholders.
        return inputs

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    if image_feature_size_override is None:
        image_data = multi_modal_data["image"]
        if isinstance(image_data, Image.Image):
            image_feature_size = get_clip_image_feature_size(hf_config)
        elif isinstance(image_data, torch.Tensor):
            num_images, image_feature_size, hidden_size = image_data.shape
        else:
            raise TypeError(f"Invalid image type: {type(image_data)}")
    else:
        image_feature_size = image_feature_size_override

    new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
        tokenizer,
        inputs.get("prompt"),
        inputs["prompt_token_ids"],
        placeholder_token_id=image_token_id,
        repeat_count=image_feature_size,
    )

    # NOTE: Create a defensive copy of the original inputs
    return token_inputs(prompt_token_ids=new_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data,
                        multi_modal_placeholders={"image": ranges})


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
        prefix: str = "",
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
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

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
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = CLIPAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config,
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")
        self.layer_norm2 = nn.LayerNorm(config.hidden_size,
                                        eps=config.layer_norm_eps)

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
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config=config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self, inputs_embeds: torch.Tensor, return_all_hidden_states: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = []
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


class CLIPVisionTransformer(nn.Module):

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
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
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
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
            self.post_layernorm = nn.LayerNorm(embed_dim,
                                               eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        return_all_hidden_states = feature_sample_layers is not None

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have feature_sample_layers or not
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states)

        # Handle post-norm (if applicable) and stacks feature layers if needed
        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs, feature_sample_layers, self.post_layernorm,
            self.config.num_hidden_layers)

        return encoder_outputs


class CLIPVisionModel(nn.Module):

    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: CLIPVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model")

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        return self.vision_model(pixel_values, feature_sample_layers)

    @property
    def device(self):
        return next(self.parameters()).device

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is not needed in CLIPVisionModel
            if (name.startswith("vision_model.post_layernorm")
                    and self.vision_model.post_layernorm is None):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
