# coding=utf-8
# Adapted from
# https://github.com/THUDM/ChatGLM2-6B
"""Inference-only ChatGLM model compatible with THUDM weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import LayerNorm
from argparse import Namespace
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math
from torch.nn import LayerNorm
from accelerate import init_empty_weights

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.transformers_utils.configs import ChatGLMConfig
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensors
from .interfaces import SupportsVision

from .interfaces import SupportsLoRA


class GLMAttention(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = (config.multi_query_group_num
                                   if config.multi_query_attention else
                                   config.num_attention_heads)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        # https://huggingface.co/THUDM/chatglm3-6b-32k/blob/e210410255278dd9d74463cf396ba559c0ef801c/modeling_chatglm.py#L141
        rope_ratio = getattr(config, "rope_ratio", 1.0)
        max_positions = getattr(config, "seq_length", 8192)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim // 2,
            max_position=max_positions,
            base=10000 * rope_ratio,
            is_neox_style=False,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h.
        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        self.activation_func = SiluAndMul()

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)

        self.fp32_residual_connection = config.fp32_residual_connection

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size,
                                               eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = GLMAttention(config, cache_config, quant_config)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = GLMMLP(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.mlp(layernorm_output) + residual

        return output


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        self.layers = nn.ModuleList([
            GLMBlock(config, cache_config, quant_config)
            for i in range(self.num_layers)
        ])

        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
            )
        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states



def get_eva2clip_model(config, quant_config):

    class PatchEmbedding(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size,
                                stride=config.patch_size)
            self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
            self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

        def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
            x = self.proj(images)
            x = x.flatten(2).transpose(1, 2)
            cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x += self.position_embedding.weight.unsqueeze(0)
            return x


    class Attention(nn.Module):

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
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

            self.output_dropout = torch.nn.Dropout(config.dropout_prob)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, _ = x.shape # B, L, 3 * H * D

            # Special case for bitsandbytes, need reshape
            qkv, _ = self.query_key_value(x)

            q, k, v = qkv.chunk(3, dim=-1)
    
            q = q.reshape(B, L, self.num_heads_per_rank,
                        self.head_dim).permute(0, 2, 1, 3)  # B, H, L, D
            k = k.reshape(B, L, self.num_heads_per_rank,
                        self.head_dim).permute(0, 2, 1, 3)  # B, H, L, D
            v = v.reshape(B, L, self.num_heads_per_rank,
                        self.head_dim).permute(0, 2, 1, 3)  # B, H, L, D

            out = torch.nn.functional.scaled_dot_product_attention(q,
                                                                k,
                                                                v,
                                                                attn_mask=None,
                                                                dropout_p=0.,
                                                                is_causal=False)
            out = out.transpose(1, 2).view(B, L, -1)

            # Special case for bitsandbytes, need reshape
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
        ):
            super().__init__()
            self.input_layernorm = LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
            self.attention = Attention(config, quant_config=quant_config)
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
        ):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerLayer(config, quant_config=quant_config)
                for _ in range(config.num_hidden_layers)
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

            # self.merged_proj = MergedColumnParallelLinear(
            #     config.hidden_size,
            #     [config.ffn_hidden_size] * 2,
            #     bias=False,
            #     quant_config=quant_config
            # )

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

            self.dense_4h_to_h = RowParallelLinear(config.ffn_hidden_size,
                                                config.hidden_size,
                                                bias=False,
                                                quant_config=quant_config)

        def forward(self, x):
            x, _ = self.linear_proj(x)

            x = self.act1(self.norm1(x))

            # x, _ = self.merged_proj(x)
            
            gate_proj_output, _ = self.gate_proj(x)

            dense_h_to_4h_output, _ = self.dense_h_to_4h(x)

            x = torch.cat([gate_proj_output, dense_h_to_4h_output], dim=-1)

            x = self.act2(x)

            x, _ = self.dense_4h_to_h(x)
            return x


    class EVA2CLIPModel(nn.Module):

        def __init__(
            self,
            config,
            quant_config: Optional[QuantizationConfig] = None,
        ):
            super().__init__()
            vision_config = Namespace(**config.vision_config)
            self.patch_embedding = PatchEmbedding(vision_config)
            self.transformer = Transformer(vision_config,
                                        quant_config=quant_config)
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
            B, L, D = x.shape

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
            
    return EVA2CLIPModel(config, quant_config)


def merge_glm_vision_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    vision_embeddings: BatchedTensors,
    boi_token_id: int,
    eoi_token_id: int,
) -> torch.Tensor:
    boi_positions = (input_ids == boi_token_id).nonzero(as_tuple=True)[0]
    eoi_positions = (input_ids == eoi_token_id).nonzero(as_tuple=True)[0]

    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for boi_pos, eoi_pos in zip(boi_positions, eoi_positions):
        assert boi_pos < eoi_pos
        mask[boi_pos:eoi_pos + 1] = True

    inputs_embeds[mask] = vision_embeddings.view(-1,
                                                 vision_embeddings.shape[-1])
    return inputs_embeds


class ChatGLMModel(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.embedding = VocabParallelEmbedding(config.padded_vocab_size,
                                                config.hidden_size)

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config, cache_config, quant_config)

        self.output_layer = ParallelLMHead(config.padded_vocab_size,
                                           config.hidden_size,
                                           quant_config=quant_config)
        if config.vision_config:
            # glm-4v vision encoder
            self.vision = get_eva2clip_model(config, quant_config)
            self.multimodal = True
            self.image_size = config.vision_config['image_size']
            self.patch_size = config.vision_config['patch_size']
            self.boi_token_id = config.boi_token_id
            self.eoi_token_id = config.eoi_token_id
        else:
            self.multimodal = False
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        inputs_embeds = self.embedding(input_ids)

        image_features = None
        if images is not None and self.multimodal == True:
            image_size: int = self.image_size
            patch_size: int = self.patch_size

            images = images.to(dtype=inputs_embeds.dtype)
            image_features = self.vision(images)
        
        if image_features is not None:
            boi_token_id = self.boi_token_id
            eoi_token_id = self.eoi_token_id
            inputs_embeds = merge_glm_vision_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                vision_embeddings=image_features,
                boi_token_id=boi_token_id,
                eoi_token_id=eoi_token_id)

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return hidden_states



def get_max_glm4v_image_tokens(ctx: InputContext):
    vision_config = ctx.get_hf_config(ChatGLMConfig).vision_config
    if vision_config is None:
        return 1
    elif isinstance(vision_config, dict):

        return (vision_config["image_size"] // vision_config["patch_size"] //
                2)**2

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)

def dummy_data_for_glm4v(ctx: InputContext, seq_len: int):
    hf_config = ctx.get_hf_config(ChatGLMConfig)
    vision_config = hf_config.vision_config

    if vision_config is None:
        token_ids = [0] * seq_len
        seq_data = SequenceData(token_ids)
        return seq_data, None
    elif isinstance(vision_config, dict):
        image_placeholder_length = (vision_config["image_size"] //
                                    vision_config["patch_size"] // 2)**2
        token_ids = [
            hf_config.boi_token_id
            ] + [0] * image_placeholder_length + [hf_config.eoi_token_id]
        
        token_ids += [0] * (seq_len - image_placeholder_length - 2)
        seq_data = SequenceData(token_ids)

        mm_data = {
            "image":
            torch.zeros(1, 3, vision_config["image_size"],
                             vision_config["image_size"])
        }
        return seq_data, mm_data
    
    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def find_all_positions(input_ids: List[int], target: int) -> List[int]:
    return [index for index, value in enumerate(input_ids) if value == target]


def input_processor_for_glm4v(ctx: InputContext, llm_inputs: LLMInputs):
    hf_config = ctx.get_hf_config(ChatGLMConfig)
    vision_config = hf_config.vision_config

    if vision_config is None:
        return llm_inputs
    elif isinstance(vision_config, dict):
        image_placeholder_length = (vision_config["image_size"] //
                                    vision_config["patch_size"] //
                                    2)**2  # 1600
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    input_ids = llm_inputs.get("prompt_token_ids")
    position_ids = llm_inputs.get("position_ids")
    if position_ids is None:
        position_ids = list(range(len(input_ids)))
    boi_token_id = hf_config.boi_token_id
    eoi_token_id = hf_config.eoi_token_id
    boi_positions = find_all_positions(input_ids, boi_token_id)
    eoi_positions = find_all_positions(input_ids, eoi_token_id)

    assert len(boi_positions) == len(eoi_positions)

    new_input_ids = []
    new_position_ids = []
    final_processed_position = 0
    final_processed_position = 0

    for boi_position, eoi_position in zip(boi_positions, eoi_positions):
        assert boi_position < eoi_position
        new_input_ids.extend(input_ids[final_processed_position:boi_position +
                                       1])
        new_position_ids.extend(
            list(range(final_processed_position, boi_position + 1)))
        new_input_ids.extend([input_ids[boi_position + 1]] *
                             image_placeholder_length)
        new_position_ids.extend([boi_position + 1] * image_placeholder_length)
        final_processed_position = eoi_position

    new_input_ids.extend(input_ids[final_processed_position:])
    new_position_ids.extend(
        list(range(final_processed_position, len(input_ids))))

    assert len(new_input_ids) == len(new_position_ids)

    llm_inputs["prompt_token_ids"] = new_input_ids
    llm_inputs["position_ids"] = new_position_ids
    return llm_inputs

# @MULTIMODAL_REGISTRY.register_image_input_mapper()
# @MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_glm4v_image_tokens)
# @INPUT_REGISTRY.register_dummy_data(dummy_data_for_glm4v)
# @INPUT_REGISTRY.register_input_processor(input_processor_for_glm4v)
# class ChatGLMForCausalLM(nn.Module, SupportsLoRA, SupportsVision):
class ChatGLMForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"]
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    bitsandbytes_stacked_params_mapping = {}

    bitsandbytes_quant_target_modules = ['dense.weight', 
                                         'dense_h_to_4h.weight',
                                         'query_key_value.weight',
                                         'dense_4h_to_h.weight']

    def __init__(
        self,
        config: ChatGLMConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        **kwargs
    ):
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.max_position_embeddings = getattr(config, "max_sequence_length",
                                               8192)
        self.transformer = ChatGLMModel(config, cache_config, quant_config)
        self.lm_head = self.transformer.output_layer
        self.logits_processor = LogitsProcessor(config.padded_vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        image_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, image_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_pos_emb.inv_freq" in name:
                continue
            if "word_embeddings" in name:
                name = name.replace(".word_embeddings", "")
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
