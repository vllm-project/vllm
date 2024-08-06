"""Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

import math
from typing import Iterable, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipAttention
from vllm_flash_attn import flash_attn_func
from xformers.ops import memory_efficient_attention

from vllm.config import ModelConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal.image import (cached_get_tokenizer,
                                   repeat_and_pad_image_tokens)
from vllm.sequence import SequenceData


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
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    if image_feature_size_override is None:
        image_feature_size = get_siglip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    token_ids = [image_token_id] * image_feature_size
    token_ids += [0] * (seq_len - image_feature_size)
    return SequenceData(token_ids)


def dummy_image_for_siglip(
    hf_config: SiglipVisionConfig,
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
    return {"image": image}


def input_processor_for_siglip(
    model_config: ModelConfig,
    hf_config: SiglipVisionConfig,
    llm_inputs: LLMInputs,
    *,
    image_token_id: int,
    image_feature_size_override: Optional[int] = None,
):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    if image_feature_size_override is None:
        image_feature_size = get_siglip_image_feature_size(hf_config)
    else:
        image_feature_size = image_feature_size_override

    new_prompt, new_token_ids = repeat_and_pad_image_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        image_token_id=image_token_id,
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


# NOTE: Not used - kept for later when we TP the ViT
# TODO(ChristopherCho): Implement TP version of Attention
class SiglipTPAttention(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"Number of attention heads ({self.total_num_heads}) "
                "must be divisible by the tensor model parallel size"
                f" ({tp_size}).")

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.embed_dim // self.total_num_heads
        if self.head_dim * self.total_num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got "
                             "`embed_dim`: {self.embed_dim} and `num_heads`:"
                             f" {self.num_heads}).")
        self.qkv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
        )

        self.attn_fn = self._basic_attention_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        batch_size, q_len, _ = hidden_states.size()

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.split(
            [self.qkv_size] * 3, dim=-1)

        attn_output = self.attn_fn(
            q=query_states,
            k=key_states,
            v=value_states,
            batch_size=batch_size,
            q_len=q_len,
        )

        attn_output, _ = self.out_proj(attn_output)
        return attn_output

    def _basic_attention_forward(self, q, k, v, batch_size, q_len):
        q = q.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)

        k_v_seq_len = k.shape[-2]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale

        if attn_weights.size() != (
                batch_size,
                self.num_heads,
                q_len,
                k_v_seq_len,
        ):
            raise ValueError(
                "Attention weights should be of size "
                f"{(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}")

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (
                batch_size,
                self.num_heads,
                q_len,
                self.head_dim,
        ):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        return attn_output


# NOTE: Not used - kept for later when we TP the ViT
# TODO(ChristopherCho): flash_attn_func is not working properly.
#                       It constantly throws a CUDA error.
class SiglipFlashAttention2(SiglipTPAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_fn = self._flash_attention_forward

    # Ported from https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/siglip/modeling_siglip.py#L449
    # and https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/modeling_flash_attention_utils.py#L133
    def _flash_attention_forward(self, q, k, v, batch_size, q_len, *args,
                                 **kwargs):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the
                     query, key, and value. (B, S, H, D)
        """

        q = q.view(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, q_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, q_len, self.num_heads, self.head_dim)

        attn_output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout,
            causal=False,
        )

        attn_output = attn_output.reshape(batch_size, q_len,
                                          self.embed_dim).contiguous()

        return attn_output


# NOTE: Not used - kept for later when we TP the ViT
class SiglipSdpaAttention(SiglipTPAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False
        self.attn_fn = self._sdpa_attention_forward

    def _sdpa_attention_forward(self, q, k, v, batch_size, q_len):
        q = q.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, q_len, self.num_heads,
                   self.head_dim).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=False, scale=self.scale)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        return attn_output


# NOTE: Not used - kept for later when we TP the ViT
class SiglipxFormersAttention(SiglipTPAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_fn = self._xformers_attention_forward

    def _xformers_attention_forward(self, q, k, v, batch_size, q_len):
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, q_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, q_len, self.num_heads, self.head_dim)

        attn_output = memory_efficient_attention(q,
                                                 k,
                                                 v,
                                                 p=0.0,
                                                 scale=self.scale)
        attn_output = attn_output.reshape(batch_size, q_len,
                                          self.embed_dim).contiguous()

        return attn_output


# NOTE: Not used - kept for later when we TP the ViT
SIGLIP_ATTENTION_CLASSES = {
    "eager": SiglipTPAttention,
    "flash_attention_2": SiglipFlashAttention2,
    "sdpa": SiglipSdpaAttention,
    "xformers": SiglipxFormersAttention,
}


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

        # TODO(ChristopherCho): use TP'ed Attention block
        self.self_attn = SiglipAttention(config)
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
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
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
