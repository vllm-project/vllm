# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implementation of Siglip2VisionModel intended to be only used
within a vision language model."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Siglip2VisionConfig

from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import MultiModalConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .vision import should_torch_compile_mm_vit


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional
                embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized
                positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for
        # bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) ->
            # (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(
                embed_dim, height * width
            ).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(
        self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches,
                num_channels * patch_size * patch_size)
            spatial_shapes (`list[tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional
                embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        use_data_parallel = (
            multimodal_config is not None
            and multimodal_config.mm_encoder_tp_mode == "data"
        )
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_heads_per_partition = self.num_heads // tp_size

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
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.scale,
            prefix=f"{prefix}.attn",
            multimodal_config=multimodal_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(
            hidden_states
        )  # batch_size, q_len, 3 * num_heads_per_partition * head_dim
        bsz, q_len, _ = qkv.shape
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        query_states = query_states.view(
            bsz, q_len, self.num_heads_per_partition, self.head_dim
        )
        key_states = key_states.view(
            bsz, q_len, self.num_heads_per_partition, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_heads_per_partition, self.head_dim
        )

        # Use unified MultiHeadAttention implementation
        out = self.attn(
            query=query_states,
            key=key_states,
            value=value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        out = out.reshape(bsz, q_len, -1)
        attn_output, _ = self.out_proj(out)
        return attn_output


class Siglip2MLP(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        use_data_parallel = (
            multimodal_config is not None
            and multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={"hidden_states": [0, 1], "cu_seqlens": 0},
    enable_if=should_torch_compile_mm_vit,
)
class Siglip2EncoderLayer(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, embed_dim).
            cu_seqlens: Cumulative sequence lengths tensor.
            max_seqlen: Maximum sequence length.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers`
    self attention layers. Each layer is a [`Siglip2EncoderLayer`].

    Args:
        config: PretrainedConfig
    """

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            hidden_states = layer_outputs
        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_dim = config.hidden_size
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        # Keep the import local to avoid circular dependencies during model init.
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("Siglip2Encoder", is_encoder=True):
            self.encoder = Siglip2Encoder(
                config,
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                prefix=f"{prefix}.encoder",
            )
        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        packed_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width)
            of the input images.
        """
        hidden_states = self.embeddings(pixel_values, spatial_shapes)
        flat_mask = packed_mask.view(-1)
        packed_indices = flat_mask.nonzero(as_tuple=True)[0]
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = flat_hidden_states.index_select(0, packed_indices).unsqueeze(0)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        unpacked = encoder_outputs.new_zeros(
            packed_mask.numel(), encoder_outputs.shape[-1]
        )
        unpacked.index_copy_(0, packed_indices, encoder_outputs.squeeze(0))
        encoder_outputs = unpacked.view(
            packed_mask.shape + (encoder_outputs.shape[-1],)
        )
        last_hidden_state = self.post_layernorm(encoder_outputs)
        return last_hidden_state


class Siglip2Model(torch.nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        packed_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            packed_mask=packed_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

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
