# Adapted from
# https://github.com/THUDM/GLM-4
"""Inference-only GLM-4v model visual encoder compatible with THUDM weights."""
from argparse import Namespace
from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


class PatchEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels,
                              config.hidden_size,
                              kernel_size=config.patch_size,
                              stride=config.patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions,
                                               config.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        images : torch.Tensor
            Input image tensor with shape (B, C, H, W)

        Returns:
        torch.Tensor
            Transformed tensor with shape (B, L, D)
        """
        images = images.to(self.proj.weight.device)
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x

class GlmSelfAttention(nn.Module):
    """Multi-headed attention without any cache, used for ViT."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: batch_size x seq_len x hidden_size"""
        # TODO(Isotr0py): Use existing backend implementations and support FA2
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        query, key, value = (x.transpose(1, 2)
                                for x in (query, key, value))
        from ipex_llm.transformers.models.utils import use_sdp_causal
        from vllm.attention.backends.ipex_attn import use_sdp_causal
        import xe_addons, math
        mask = None
        scale = 1 / math.sqrt(self.head_size) if self.scale is None else self.scale
        from ipex_llm.transformers.models.common import padding_qkv_hd

        query, key, value, = padding_qkv_hd(
            query, key, value,
            self.head_size, 128
        )
        if use_sdp_causal(query.shape[-1], query, 0):
            out = xe_addons.sdp_causal(query.contiguous(), key.contiguous(), value.contiguous(), mask, scale)[:, :, :, :self.head_size].transpose(1, 2)
        # import torch.nn.functional as F
        # out = F.scaled_dot_product_attention(query,
        #                                      key,
        #                                      value,
        #                                      scale=self.scale)
        # out = out.transpose(1, 2)
        #return out.view(bsz, q_len, -1)
        return out.reshape(bsz, q_len, -1)

class Attention(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_rank = config.num_heads // self.tp_size
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_heads,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            quant_config=quant_config,
        )

        # self.attn = MultiHeadAttention(self.num_heads_per_rank, self.head_dim,
        #                                self.scale)
        self.attn = GlmSelfAttention(self.num_heads_per_rank, self.head_dim,
                                self.scale)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.query_key_value(x)  # B, L, 3 * H * D
        q, k, v = qkv.chunk(3, dim=-1)

        out = self.attn(q, k, v)
        output, _ = self.dense(out)
        output = self.output_dropout(output)
        return output


class MLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x, _ = self.fc2(x)
        return x


class TransformerLayer(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size,
                                         eps=config.layer_norm_eps)
        self.attention = Attention(config,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.attention")
        self.mlp = MLP(config, quant_config=quant_config)
        self.post_attention_layernorm = LayerNorm(config.hidden_size,
                                                  eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(
            self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.layer.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Module):

    def __init__(
        self,
        config,
        in_features,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        """
        The original implementation is the same as:
        ```python
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            quant_config=quant_config
        )

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            quant_config=quant_config
        )
        ```
        ```
        gate_proj_output, _ = self.gate_proj(x)
        dense_h_to_4h_output, _ = self.dense_h_to_4h(x)
        x = torch.cat([gate_proj_output, dense_h_to_4h_output], dim=-1)
        ```

        We merge two ColumnParallelLinear into one MergedColumnParallelLinear:
        ```
        self.merged_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=False,
            quant_config=quant_config
        )
        ```
        ```
        x, _ = self.merged_proj(x)
        ```
        """
        super().__init__()
        self.linear_proj = ReplicatedLinear(in_features,
                                            config.hidden_size,
                                            bias=False,
                                            quant_config=quant_config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = SiluAndMul()

        self.merged_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.ffn_hidden_size] * 2,
            bias=False,
            quant_config=quant_config)

        self.dense_4h_to_h = RowParallelLinear(config.ffn_hidden_size,
                                               config.hidden_size,
                                               bias=False,
                                               quant_config=quant_config)

    def forward(self, x):
        x, _ = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x, _ = self.merged_proj(x)
        x = self.act2(x)
        x, _ = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = '',
    ):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.transformer")
        self.linear_proj = GLU(config,
                               in_features=config.hidden_size,
                               quant_config=quant_config)
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size,
                              out_channels=config.hidden_size,
                              kernel_size=2,
                              stride=2)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        images : torch.Tensor
            Input image tensor with shape (B, C, H, W)

        Returns:
        torch.Tensor
            Transformed tensor with shape (B, L, D)
        """
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x
