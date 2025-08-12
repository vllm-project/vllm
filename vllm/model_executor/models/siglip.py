"""Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

import math
from array import array
from typing import Iterable, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch import nn
from transformers import SiglipVisionConfig
from xformers import ops as xops

from vllm.config import ModelConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal.utils import (cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData


def get_siglip_patch_grid_length(*, image_size: int, patch_size: int) -> int:
    # Since interpolation is applied, the image size need not be divisible
    # assert image_size % patch_size == 0
    return image_size // patch_size


def get_siglip_num_patches(*, image_size: int, patch_size: int) -> int:
    grid_length = get_siglip_patch_grid_length(image_size=image_size,
                                               patch_size=patch_size)
    return grid_length * grid_length


def get_siglip_image_feature_size(hf_config: SiglipVisionConfig) -> int:
    return get_siglip_num_patches(image_size=hf_config.image_size,
                                  patch_size=hf_config.patch_size)


def get_max_siglip_image_tokens(hf_config: SiglipVisionConfig) -> int:
    return get_siglip_image_feature_size(hf_config)


def dummy_seq_data_for_siglip(
    hf_config: SiglipVisionConfig,
    seq_len: int,
    num_images: int,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_siglip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                      [image_token_id]) * image_feature_size
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                       [0]) * (seq_len - image_feature_size)
    return SequenceData(token_ids)


def dummy_image_for_siglip(
    hf_config: SiglipVisionConfig,
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


def input_processor_for_siglip(
    model_config: ModelConfig,
    hf_config: SiglipVisionConfig,
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
            image_feature_size = get_siglip_image_feature_size(hf_config)
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
    return LLMInputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data=multi_modal_data,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/siglip/modeling_siglip.py#L249 # noqa
class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = VocabParallelEmbedding(
            self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions, dtype=torch.int64).expand(
                (1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int,
                                 width: int) -> torch.Tensor:
        """
        This method is an adapted method for SigLIP (due to SigLIP not having
        class embedding unlike other ViTs) that allows the model to interpolate
        the pre-trained position encodings such that it can be usable on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        position_embeddings = self.position_embedding.weight.unsqueeze(0)
        num_patches = embeddings.shape[1]
        num_positions = position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return position_embeddings

        dim = embeddings.shape[-1]
        height = height // self.patch_size
        width = width // self.patch_size
        # we add a small number to avoid floating point error
        # in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1

        patch_pos_embed = position_embeddings.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)),
            dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                height / math.sqrt(num_positions),
                width / math.sqrt(num_positions),
            ),
            mode="bicubic",
            align_corners=False,
        )
        if (int(height) != patch_pos_embed.shape[-2]
                or int(width) != patch_pos_embed.shape[-1]):
            raise ValueError("Width or height does not match with "
                             "the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self,
                pixel_values: torch.Tensor,
                interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(
                self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got "
                             "`embed_dim`: {self.embed_dim} and `num_heads`:"
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

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        batch_size, q_len, _ = hidden_states.size()

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)

        query_states = query_states.view(batch_size, q_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)
        key_states = key_states.view(batch_size, q_len,
                                     self.num_heads_per_partition,
                                     self.head_dim)
        value_states = value_states.view(batch_size, q_len,
                                         self.num_heads_per_partition,
                                         self.head_dim)

        out = xops.memory_efficient_attention_forward(query_states,
                                                      key_states,
                                                      value_states,
                                                      p=self.dropout,
                                                      scale=self.scale)
        out = out.view(batch_size, q_len, -1)
        attn_output, _ = self.out_proj(out)

        return attn_output


class SiglipMLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)

        # For quantization, we require the hidden size to be a multiple of 64
        quantizable = (config.hidden_size % 64 == 0
                       and config.intermediate_size % 64 == 0)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config if quantizable else None,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config if quantizable else None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = SiglipAttention(config, quant_config=quant_config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            quant_config=quant_config,
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class SiglipEncoder(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
    ):
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            SiglipEncoderLayer(config, quant_config=quant_config)
            for _ in range(num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states, _ = encoder_layer(hidden_states)

        return hidden_states


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # TODO(ChristopherCho): Implement vLLM version of MultiheadAttention
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config=config, quant_config=quant_config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
        )
        self.post_layernorm = nn.LayerNorm(embed_dim,
                                           eps=config.layer_norm_eps)
        self.use_head = (True if not hasattr(config, "vision_use_head") else
                         config.vision_use_head)
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(
                config=config, quant_config=quant_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(encoder_outputs)

        # TODO: add this back when pooled_output is used in inference
        # if self.use_head:
        # pooled_output = self.head(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
    ):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(
            config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # omit layers when num_hidden_layers_override is set
            if "vision_model.encoder.layers." in name:
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
