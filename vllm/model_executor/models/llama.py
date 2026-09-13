# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""

# [CN] 文件总览：Llama 及其同构模型的 vLLM 实现 —— DecoderOnly 模型的标准范本
# [CN] 职责：把 HuggingFace 的 LlamaConfig 装配成一个可推理的 nn.Module。
# [CN] 层级：LlamaForCausalLM（含 lm_head）
# [CN]         └─ LlamaModel（embed + N 层 + final norm）
# [CN]              └─ LlamaDecoderLayer × N（pre-norm + residual）
# [CN]                   ├─ LlamaAttention（QKV 投影 + RoPE + attention 后端）
# [CN]                   └─ LlamaMLP（gate_up 融合 + down）
# [CN] 为什么先读它：Qwen / Mistral / Yi / InternLM 等大量模型都是这个骨架的变体，
# [CN]   读懂本文件后，读其他 model 只需看它们的差异点（是不是 MoE、有无 sliding window）。
# [CN] 与 HF 的关键差别（也是 vLLM 能快的原因）：
# [CN]   1. 融合 QKV / gate_up 成一次 GEMM，减少 kernel launch 与中间显存
# [CN]   2. 所有线性层换成 TP 感知的 xxxParallelLinear，天然支持张量并行
# [CN]   3. attention 走统一后端接口，KV 落在分页的 KV cache 里而非连续缓存
# [CN]   4. 加载二手库认 this->vllm 的名字映射（hf_to_vllm_mapper），不改权重文件

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .adapters import as_embedding_model, as_seq_cls_model
from .interfaces import (
    EagleModelMixin,
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)


# [CN] FFN 层。HF 里是独立的 gate_proj / up_proj / down_proj，这里把前两个融合。

class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        # [CN] gate 与 up 拼在一次 GEMM 里：它们的输入相同，合起来能让输出一次写满连续内存，
        # [CN] 随后 SiluAndMul 直接在前一半/后一半之间做 SiLU(x)*y。省一次 launch 与一块中间显存。

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        # [CN] RowParallel：输入按 intermediate 维切、输出需要跨卡求和，因此带 all-reduce
        # [CN] （reduce_results 控制是否在这里 reduce，某些 PP 场景要延迟聚合）。

        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        # [CN] 融合实现绑定了 SiLU：换激活就必须改 MLP 结构，这里直接拒绝而不是走慢路径。

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # [CN] 丢弃的第二个返回值是 bias（parallel linear 统一返回 (out, bias)）。

        x, _ = self.gate_up_proj(x)
        # [CN] 把「激活 + 下一层的量化」尝试融合成一个算子；不支持时自动退回分开执行。

        x = maybe_fused_act_quant(self.act_fn, x, self.down_proj)
        x, _ = self.down_proj(x)
        return x


# [CN] 注意力层。核心是三件事：TP 下怎么切头、GQA 怎么处理、qkv 怎么一次性算出。

class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        # [CN] 层号从 prefix 字符串里解析出来 —— 因为 внимание sliding window 的判定需要「第几层」。

        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        # [CN] 注意力头必须能被 TP size 整除：每个 rank 拿一段完整的头，
        # [CN] 输出再拼起来就是完整的 num_heads。

        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        # [CN] 两个分支的区别是 GQA：KV 头数多于 TP 时可以真正切分；
        # [CN] 少于 TP 时只能复制（每份 KV 被多张卡各存一份，显存上是浪费但是必须的）。

        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        # [CN] max(1, ...) 保底：KV 头数少于 TP 时每份至少 1 个头（即上面的复制路径）。

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # [CN] 优先用 config 显式给的 head_dim；没有就从 hidden_size/num_heads 推 ——
        # [CN] 老版 config 常不写这个字段，而某些小模型 MQA 时的整除关系并不平凡。

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # [CN] Q/K/V 合并投影。它对 GQA 有专门处理：K/V 只按 num_kv_heads 切，
        # [CN] 因此每张卡上持有的张量长度不同（q_size vs kv_size），后面要按各自长度 split。

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config, quant_config=quant_config)

        # [CN] 逐层 sliding window：混合 attention 架构（如部分层滑窗）靠它传给后端。

        sliding_window = None
        # [CN] layer_types 是「每层一个字符串」的数组，指明该层是 full 还是 sliding_attention。

        if layer_types := getattr(config, "layer_types", None):
            # Fix for Eagle3 compatibility:
            # for draft models, subtract target layer count
            # to get draft-relative layer index starting from 0
            # [CN] Eagle3 的 draft 模型：它的 prefix 层号是接续 target 之后编号的，
            # [CN] 因此要减掉 target 层数才能落在 draft 自己的 layer_types 数组范围内。

            if hasattr(config, "target_layer_count"):
                # This is a draft model,
                # adjust layer_idx to be relative to draft layers
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                # This is a target model, use layer_idx directly
                effective_layer_idx = layer_idx
            assert effective_layer_idx < len(layer_types), (
                f"effective_layer_idx: {effective_layer_idx} "
                f"is out of bounds for layer_types: {layer_types}"
            )

            is_sliding = layer_types[effective_layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        # [CN] 双向（embedding 模型）场景用 EncoderOnlyAttention —— 它不做因果掩码、也不写 KV cache。

        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    # [CN] 标准四步：QKV 投影 -> 拆 q/k/v -> 给 q,k 加 RoPE -> 交给 attention 后端。

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # [CN] 注意这里的 qkv 是「融合输出」，其列排布是 [q..., k..., v...] 的顺序。

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # [CN] 只给 q 和 k 加位置编码，v 从不加 —— 注意力内积只发生在 q·k 上。

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        # [CN] Llama 系固定使用 neox 风格的旋转编码（相邻两维配对旋转），写成常量便于子类覆盖。

        is_neox_style = True

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


# [CN] 一个 Transformer 块。注意 pre-norm 结构：norm 在 attention/mlp **之前**。

class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        attn_layer_type: type[nn.Module] = LlamaAttention,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # [CN] 两个别名都要看：有些 checkpoint 用 attention_bias，有些用 bias。

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        # [CN] qkv_bias 优先级最高：internlm3 这类模型只给 QKV 加 bias 而 o_proj 不带。

        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. nvidia/llama-nemotron-embed-1b-v2)
        # [CN] 默认因果（decoder-only）；某些 embedding 模型通过 is_causal=False 打开双向注意力。

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = attn_layer_type(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        # [CN] RMSNorm 而非 LayerNorm：少了减均值与 bias，计算更省，Llama 系标配。

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    # [CN] residual 贯穿整层：进来的是上一层未归一化的输出，出去的也是，
    # [CN] 归一化只作用在要走 attention/mlp 的那一路分支上。

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # [CN] 首层的特殊处理：没有上一层的 residual，就用 hidden_states 自己当 residual
        # [CN] —— 这样第一层与后续层可以共用同一段 Zelメラlogic。

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # [CN] 这里进去的是已经归一化过的一路，另一路 residual 原样透传到下一层。

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    # [CN] 钩子：子类（如不同 MoE 变体）可以在这里换成自己的量化配置。

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Get quantization config for this layer. Override in subclasses."""
        return vllm_config.quant_config


# [CN] 编译配置：把 batch 维度标成动态维(dim='b')，让不同 batch size 共用一份图，
# [CN] 否则每次 batch 变化都会触发重新编译（首 Token 延迟暴增的直接原因之一）。

@support_torch_compile(
    # TODO[#32068]: Investigate recompilation
    # mark_unbacked_dims={"input_ids": 0},
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
# [CN] 骨干。EagleModelMixin 让它能被 Eagle/Eagle3 投机解码复用 ——
# [CN] 复用的方式是收集中间层的 hidden states（aux_hidden_states）。

class LlamaModel(nn.Module, EagleModelMixin):
    # [CN] HF ->vLLM 的名字/结构映射表：把 HF 的 q_proj/k_proj/v_proj 三个独立权重，
    # [CN] 叠成本文件里融合后的 qkv_proj 的一个切片（第 2 列是 shard id）。
    # [CN] 这是「不用改 checkpoint 就能加载」的关键。

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )
    # [CN] 声明「支持跨 PP rank 收集中间隐藏状态」，Eagle 的 draft 模型依赖这个能力。

    supports_aux_hidden_states_over_pp = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        # [CN] embed_tokens 只在该装的时候装：PP 中间 rank 不需要它，
        # [CN] 但**最后一个 rank 若开了 tie_word_embeddings 仍需要**（要复用同一份权重）。

        if (
            get_pp_group().is_first_rank
            or (config.tie_word_embeddings and get_pp_group().is_last_rank)
            or spec_decode_needs_target_embed(vllm_config)
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # [CN] final norm 只在最后一段 PP 上；其他 rank 放一个 PPMissingLayer 占位，
        # [CN] 保证 forward 里不用每处都写「如果我是中间 rank」。

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    # [CN] 单独拆出来是为了让多模态模型能覆盖它（先算好自己的 embeddings 再进来）。

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # [CN] PP 首段负责 embedding；其余 rank 从上一段传来的 intermediate_tensors 接手。

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                # [CN] inputs_embeds 优先：多模态模型会预先把图像/音频embedding拼好传进来。

                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # [CN] 跨 PP 拉取其他 rank 产生的中间隐藏状态，后面与本地的拼成一串。

        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)

        aux_hidden_states: list[torch.Tensor] = []
        if get_pp_group().is_first_rank:
            self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, hidden_states, residual
            )
        # [CN] islice 而非切片 (list)：self.layers 在 PP 下可能已经是 view，
        # [CN] 切片会额外拷贝；且 start_layer/end_layer 语义上是本 rank 的管辖范围。

        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions, hidden_states, residual, **extra_layer_kwargs
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        # [CN] 非最后一段：把 [hidden_states, residual] 一起交給下一个 PP rank。
        # [CN] 为什么连 residual 也要传 —— 因为下一个 rank 的 norms 需要它做加法。

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        # [CN] remote（前序 rank）在前、本地在后，顺序对应流水线的真实层级顺序。

        aux_hidden_states = remote_aux + aux_hidden_states
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    # [CN] 把加载交给 AutoWeightsLoader，返回被实际消费过的权重名集合（用于严格性校验）。

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


# [CN] 顶层封装：骨干 + lm_head + logits 处理。混入的多个 Supports* 是「能力声明」，
# [CN] 让引擎知道它支持 LoRA / PP / 量化 / Eagle 等特性。

class LlamaForCausalLM(
    LocalArgmaxMixin,
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
    SupportsQuant,
):
    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper
    # LoRA specific attributes
    # [CN] LoRA 需要知道「哪些 HF 子模块被打包进了同一个并联线性层」，以便正确重排 B 矩阵。

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # [CN] 告诉 LoRA 这两个东西是 Embedding（不是 Linear），走另一套 add_lora 实现。

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            # [CN] 词表维并行：lm_head 按 vocab 切给各 TP rank，因此后面拿 logits 需要聚合。

            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            # [CN] 权重共享：lm_head 直接指向 embed_tokens 的参数，省下一份 vocab×hidden 的显存。
            # [CN] 代价是 embedding 的梯度/更新会影响输出头 —— 推理无妨，训推一体时要注意。

            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            # [CN] 部分模型（如 Cohere 系）在 logits 上乘一个固定系数，必须在采样前生效。

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    # [CN] 完整路径：含 TP 下的 logits 聚合，得到全词表概率（用于 sampler）。

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # [CN] 只有最后一个 PP rank 会走到这里 —— 中间 rank 根本没有 lm_head。

        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    # [CN] skip_gather=True：只算本卡那一段词表，用于 LocalArgmaxMixin 的贪心路径，
    # [CN] 省掉一次全词表通信（见 logits_processor.get_top_tokens）。

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


if TYPE_CHECKING:
    _LlamaBidirectionalForSequenceClassificationBase = LlamaForCausalLM
    _LlamaBidirectionalModelBase = LlamaForCausalLM
else:
    _LlamaBidirectionalForSequenceClassificationBase = as_seq_cls_model(
        LlamaForCausalLM
    )
    _LlamaBidirectionalModelBase = as_embedding_model(LlamaForCausalLM)


# [CN] 双向变体全部继承自 CausalLM：差异只在 LlamaBidirectionalConfig 里改
# [CN] attention 类型与 pooling 方式，因此这里没有一行代码。

class LlamaBidirectionalForSequenceClassification(
    _LlamaBidirectionalForSequenceClassificationBase
):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass


# [CN] embedding 变体，同上：实现全靠 as_embedding_model 的运行时改写。

class LlamaBidirectionalModel(_LlamaBidirectionalModelBase):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass
