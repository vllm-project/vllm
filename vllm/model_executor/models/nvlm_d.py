# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py
# --------------------------------------------------------
# NVLM-D
# Copyright (c) 2024 NVIDIA
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .intern_vit import (InternVisionEmbeddings, InternVisionEncoder,
                         InternVisionEncoderLayer, InternVisionModel)
from .internvl import (InternVLChatModel, dummy_data_for_internvl,
                       get_max_internvl_image_tokens,
                       input_mapper_for_internvl, input_processor_for_internvl)

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False


class NVLMVisionEmbeddings(InternVisionEmbeddings):

    def _get_position_embedding(self, H: int, W: int) -> torch.Tensor:
        return self.position_embedding


class NVLMParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        # We added additional dummy heads to the original num of heads to make
        # the number of heads divisible by 8.
        self.num_dummy_heads = 7
        self.dummy_dim = (self.num_dummy_heads +
                          self.num_heads) * self.head_dim

        self.scale = self.head_dim**-0.5
        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.dummy_dim,
            self.num_dummy_heads + self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
        )

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)
            self.k_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)

        self.proj = RowParallelLinear(
            self.dummy_dim,
            self.embed_dim,
            quant_config=quant_config,
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(
            self.num_dummy_heads + self.num_heads, self.tp_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads_per_partition, self.head_dim)
        k = k.view(B, N, self.num_heads_per_partition, self.head_dim)
        v = v.view(B, N, self.num_heads_per_partition, self.head_dim)

        if self.qk_normalization:
            B_, N_, H_, D_ = q.shape
            q = self.q_norm.forward_native(q.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)
            k = self.k_norm.forward_native(k.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)

        x = xops.memory_efficient_attention_forward(q, k, v, scale=self.scale)
        x = x.view(B, N, -1)

        x, _ = self.proj(x)
        return x


class NVLMSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        # We added additional dummy heads to the original num of heads to make
        # the number of heads divisible by 8.
        self.num_dummy_heads = 7
        self.dummy_dim = (self.num_dummy_heads +
                          self.num_heads) * self.head_dim

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim,
                             3 * self.dummy_dim,
                             bias=config.qkv_bias)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)
            self.k_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)

        self.proj = nn.Linear(self.dummy_dim, self.embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_dummy_heads + self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_dummy_heads + self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_dummy_heads + self.num_heads, self.head_dim)

        if self.qk_normalization:
            B_, N_, H_, D_ = q.shape
            q = self.q_norm.forward_native(q.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)
            k = self.k_norm.forward_native(k.flatten(-2,
                                                     -1)).view(B_, N_, H_, D_)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).view(B, N, -1)

        x = self.proj(x)
        return x


class NVLMVisionEncoderLayer(InternVisionEncoderLayer):

    def _init_attn(self, config: PretrainedConfig,
                   quant_config: Optional[QuantizationConfig]):
        # fallback to sdpa attention if tp unavailable
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads

        if USE_XFORMERS_OPS and num_heads % tp_size == 0:
            return NVLMParallelAttention(config, quant_config=quant_config)

        return NVLMSdpaAttention(config)


class NVLMVisionEncoder(InternVisionEncoder):

    def _init_encoder_layer(self, config: PretrainedConfig,
                            quant_config: Optional[QuantizationConfig]):
        return NVLMVisionEncoderLayer(config=config, quant_config=quant_config)


class NVLMVisionModel(InternVisionModel):

    def _init_encoder(self, config: PretrainedConfig,
                      quant_config: Optional[QuantizationConfig],
                      num_hidden_layers_override: Optional[int]):
        return NVLMVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_internvl)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_internvl_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_internvl)
@INPUT_REGISTRY.register_input_processor(input_processor_for_internvl)
class NVLM_D_Model(InternVLChatModel):

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_intermediate_size = config.text_config.intermediate_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_intermediate_size,
                      bias=False),
            nn.GELU(),
            nn.Linear(llm_intermediate_size, llm_hidden_size, bias=False),
        )

    def _init_vision_model(self, config: PretrainedConfig,
                           num_hidden_layers: int):
        return NVLMVisionModel(config.vision_config,
                               num_hidden_layers_override=num_hidden_layers)
