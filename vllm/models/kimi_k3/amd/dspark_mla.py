# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm binding for the Kimi-K3 DSpark MLA draft.

The factory builds ``KimiK3MultiHeadLatentAttentionWrapper`` so absorb BMM
goes through generic ``MLAAttention``. Do not reuse ``KimiMLAAttention``:
that class is NoPE-only.
"""

import math

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.kimi_k3.common.dspark_mla import (
    K3DSparkDecoderLayer as _K3DSparkDecoderLayer,
)
from vllm.models.kimi_k3.common.dspark_mla import (
    K3DSparkForCausalLM as _K3DSparkForCausalLM,
)
from vllm.models.kimi_k3.common.dspark_mla import (
    K3DSparkModel as _K3DSparkModel,
)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from .linear import KimiMLP
from .mla import KimiK3MultiHeadLatentAttentionWrapper

__all__ = [
    "K3DSparkDecoderLayer",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
]


def _make_dspark_mla_attention(
    *,
    config: KimiLinearConfig,
    cache_config,
    quant_config,
    prefix: str,
) -> KimiK3MultiHeadLatentAttentionWrapper:
    """Build DSpark MLA with RoPE and a non-causal decode KV-cache spec.

    The projection and YaRN-mscale setup below is duplicated from
    ``MultiHeadLatentAttention.__init__`` in ``nvidia/mla.py``. Keep it
    aligned with that constructor: rope-type remapping
    (``attention_factor == 1.0`` becomes ``deepseek_llama_scaling``, otherwise
    ``deepseek_yarn``), ``dtype=torch.float32``, ``is_neox_style=False``, and
    ``disable_tp=True`` on the fused QKV down-proj.
    """
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    qk_nope_head_dim = config.qk_nope_head_dim
    qk_rope_head_dim = config.qk_rope_head_dim
    v_head_dim = config.v_head_dim
    q_lora_rank = config.q_lora_rank
    kv_lora_rank = config.kv_lora_rank
    assert qk_nope_head_dim is not None
    assert qk_rope_head_dim is not None
    assert v_head_dim is not None
    assert kv_lora_rank is not None
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    tp_size = get_tensor_model_parallel_world_size()
    assert num_heads % tp_size == 0
    num_local_heads = num_heads // tp_size
    scaling = qk_head_dim**-0.5

    rope_parameters = dict(config.rope_parameters)
    if rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_llama_scaling"
            if rope_parameters.get("attention_factor") == 1.0
            else "deepseek_yarn"
        )
    rotary_emb = get_rope(
        qk_rope_head_dim,
        max_position=config.max_position_embeddings,
        rope_parameters=rope_parameters,
        is_neox_style=False,
        dtype=torch.float32,
    )
    if rope_parameters["rope_type"] == "deepseek_yarn":
        mscale_all_dim = rope_parameters.get("mscale_all_dim", False)
        scaling_factor = rope_parameters["factor"]
        mscale = (
            1.0
            if scaling_factor <= 1
            else 0.1 * float(mscale_all_dim) * math.log(scaling_factor) + 1.0
        )
        scaling *= mscale * mscale

    fused_qkv_a_proj = None
    kv_a_proj_with_mqa = None
    q_a_layernorm = None
    q_b_proj = None
    q_proj = None
    if q_lora_rank is not None:
        fused_qkv_a_proj = MergedColumnParallelLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
    else:
        kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        q_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

    kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
    kv_b_proj = ColumnParallelLinear(
        kv_lora_rank,
        num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.kv_b_proj",
    )
    o_proj = RowParallelLinear(
        num_heads * v_head_dim,
        hidden_size,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.o_proj",
    )

    mla_modules = MLAModules(
        kv_a_layernorm=kv_a_layernorm,
        kv_b_proj=kv_b_proj,
        rotary_emb=rotary_emb,
        o_proj=o_proj,
        fused_qkv_a_proj=fused_qkv_a_proj,
        kv_a_proj_with_mqa=kv_a_proj_with_mqa,
        q_a_layernorm=q_a_layernorm,
        q_b_proj=q_b_proj,
        q_proj=q_proj,
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )
    return KimiK3MultiHeadLatentAttentionWrapper(
        hidden_size,
        num_local_heads,
        scaling,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
        mla_modules,
        cache_config,
        quant_config,
        prefix,
        non_causal_multi_token_decode=True,
    )


class K3DSparkDecoderLayer(_K3DSparkDecoderLayer):
    def build_self_attn(
        self,
        *,
        vllm_config: VllmConfig,
        config: KimiLinearConfig,
        quant_config,
        prefix: str,
    ):
        return _make_dspark_mla_attention(
            config=config,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def build_mlp(self, *, config, quant_config, prefix: str):
        return KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=prefix,
        )


class K3DSparkModel(_K3DSparkModel):
    decoder_layer_cls = K3DSparkDecoderLayer

    def kv_cache_layer(self, attn: nn.Module) -> nn.Module:
        return attn.mla_attn  # type: ignore[attr-defined]


class K3DSparkForCausalLM(_K3DSparkForCausalLM):
    model_cls = K3DSparkModel
