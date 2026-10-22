# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.beam import BeamConfig
from vllm.v1.attention.backend import AttentionType

from ..common.layernorm import RMSNorm
from .attention_ops import beam_output_gate, beam_qk_norm_rope


def use_merged_qkv_gate(config: BeamConfig, tp_size: int) -> bool:
    return config.num_key_value_heads >= tp_size


class BeamAttention(nn.Module):
    """Beam GQA using standard vLLM projections and attention dispatch."""

    def __init__(
        self,
        config: BeamConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = extract_layer_index(prefix)
        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.merged_qkv_gate = use_merged_qkv_gate(config, tp_size)
        if self.merged_qkv_gate:
            self.qkv_gate_proj = MergedColumnParallelLinear(
                hidden_size,
                [
                    self.total_num_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    self.total_num_heads * self.head_dim,
                ],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_gate_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=reduce_results,
        )

        self.rms_norm_eps = config.rms_norm_eps
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.disable_rope = config.layer_types[self.layer_id] == "full_attention"
        if not self.merged_qkv_gate:
            self.output_gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.output_gate_proj",
            )
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads))
        set_weight_attrs(self.gate_bias, {"weight_loader": sharded_weight_loader(0)})

        if self.disable_rope:
            self.rotary_emb = None
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                is_neox_style=True,
                rope_parameters=config.rope_parameters,
                dtype=torch.float32,
            )

        is_sliding = config.layer_types[self.layer_id] == "sliding_attention"
        per_layer_sw = config.sliding_window if is_sliding else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sw,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        gate = None
        if self.merged_qkv_gate:
            packed, _ = self.qkv_gate_proj(hidden_states)
            qkv, gate = packed.split(
                [self.q_size + 2 * self.kv_size, self.q_size], dim=-1
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            gate, _ = self.output_gate_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        rope = self.rotary_emb
        q, k = beam_qk_norm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rms_norm_eps,
            positions,
            rope.cos_sin_cache if rope is not None else None,
            self.head_dim,
            rope.rotary_dim if rope is not None else 0,
            rope.is_neox_style if rope is not None else True,
            "root10" if self.disable_rope else "none",
        )

        attn_output = self.attn(q, k, v)
        attn_output = beam_output_gate(
            attn_output,
            gate,
            self.gate_bias,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            floor=0.0,
            temperature=1.0,
        )
        output, _ = self.o_proj(attn_output)
        return output
