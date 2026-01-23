# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Implementation of Siglip2VisionModel intended to be only used
within a vision language model."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Siglip2VisionConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .vision import (
    is_vit_use_data_parallel,
    resolve_visual_encoder_outputs,
    should_torch_compile_mm_vit,
)


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

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """Embed patchified pixel values in packed (unpadded) form.

        Args:
            pixel_values_packed: (1, total_tokens, patch_dim) or
                (total_tokens, patch_dim), packed in tile order.
            spatial_shapes: (num_tiles, 2) on CPU (height, width) per tile.

        Returns:
            (1, total_tokens, embed_dim) packed embeddings.
        """
        assert spatial_shapes.device.type == "cpu", (
            "Expected `spatial_shapes` on CPU to avoid device-to-host sync in "
            "variable-length packing."
        )

        if pixel_values_packed.dim() == 3:
            assert pixel_values_packed.shape[0] == 1
            pixel_values_flat = pixel_values_packed[0]
        else:
            pixel_values_flat = pixel_values_packed

        lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(dtype=torch.int64)
        lengths_list = lengths.tolist()
        total_tokens = int(sum(lengths_list))
        if total_tokens != pixel_values_flat.shape[0]:
            raise ValueError(
                "Packed pixel_values token count does not match spatial_shapes: "
                f"{pixel_values_flat.shape[0]} vs {total_tokens}."
            )

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values_flat.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        packed_pos_embeds = self.resize_positional_embeddings_packed(
            positional_embeddings,
            spatial_shapes,
            lengths_list=lengths_list,
        )

        embeddings = patch_embeds + packed_pos_embeds
        return embeddings.unsqueeze(0)

    @staticmethod
    def resize_positional_embeddings_packed(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        lengths_list: list[int],
    ) -> torch.Tensor:
        """Resize positional embeddings per image and return a packed tensor.

        Args:
            positional_embeddings: (height, width, embed_dim) base grid.
            spatial_shapes: (batch_size, 2) on CPU, (height, width) per image.
            lengths_list: flattened token length per image (height * width).

        Returns:
            (total_tokens, embed_dim) packed positional embeddings, concatenated
            in the same order as `lengths_list`.
        """
        assert spatial_shapes.device.type == "cpu"

        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        total_tokens = int(sum(lengths_list))
        packed_pos_embeds = torch.empty(
            (total_tokens, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width)
        pos_4d = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for
        # bfloat16/float16 on CPU.
        if pos_4d.device.type == "cpu":
            pos_4d = pos_4d.to(torch.float32)

        offset = 0
        for i, length in enumerate(lengths_list):
            if length <= 0:
                continue
            height, width = spatial_shapes[i].tolist()
            resized = F.interpolate(
                pos_4d,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape(embed_dim, height * width).transpose(0, 1)
            resized = resized.to(source_dtype)
            packed_pos_embeds[offset : offset + length] = resized
            offset += length

        return packed_pos_embeds


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
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

        use_data_parallel = is_vit_use_data_parallel()
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
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        use_data_parallel = is_vit_use_data_parallel()
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
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
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
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
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
                num_hidden_layers_override=num_hidden_layers_override,
                prefix=f"{prefix}.encoder",
            )
        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width)
        of the input images.
        """
        hidden_states = self.embeddings(pixel_values_packed, spatial_shapes)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            return_all_hidden_states=select_layers is not None,
        )
        return self.post_layernorm(encoder_outputs)


class Siglip2Model(torch.nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the vision model.

        Args:
            select_layers: Layer indices to select hidden states from.
                Supports negative indices (e.g., [-2] for second-to-last).
                If None, returns the last layer output with post_layernorm.
                Multiple layers can be selected and will be concatenated.
        """
        return self.vision_model(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            select_layers=select_layers,
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
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is optional in Siglip2Model
            if (
                name.startswith("vision_model.post_layernorm")
                and self.vision_model.post_layernorm is None
            ):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
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
