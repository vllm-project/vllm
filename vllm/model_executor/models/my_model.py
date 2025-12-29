# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from LLaMA implementation with DeepSeek MTP (Multi-Token Prediction) support
# MTP核心逻辑参考DeepSeek官方实现：https://github.com/deepseek-ai/DeepSeek-MoE

from collections.abc import Iterable
from itertools import islice
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
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
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .adapters import as_embedding_model, as_seq_cls_model
from .interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


# ======================== DeepSeek MTP 核心模块 ========================
class MTPDecoderLayer(nn.Module):
    """DeepSeek MTP 轻量级解码层（用于多token预测）"""
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias,
            cache_config=None,  # MTP层不使用KV缓存
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
        )
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size // 2,  # MTP层MLP维度减半（轻量化）
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=bias,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MTPModule(nn.Module):
    """DeepSeek MTP 多token预测模块（核心）"""
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.mtp_num_layers = getattr(config, "mtp_num_layers", 2)  # MTP层数量（建议2-4层）
        self.mtp_prediction_length = getattr(config, "mtp_prediction_length", 4)  # 预测未来token数（D）
        self.mtp_loss_weight = getattr(config, "mtp_loss_weight", 0.5)  # MTP损失权重λ

        # MTP投影层（将主模型hidden state映射到MTP空间）
        self.mtp_proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mtp_proj",
        )

        # MTP解码层（轻量级）
        self.mtp_layers = nn.ModuleList([
            MTPDecoderLayer(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads // 2,  # MTP头数减半
                num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads) // 2,
                quant_config=quant_config,
                bias=False,
                prefix=f"{prefix}.mtp_layers.{i}",
            ) for i in range(self.mtp_num_layers)
        ])

        # MTP输出投影（共享主模型lm_head，无需额外参数）
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        main_hidden_states: torch.Tensor,  # 主模型最后一层输出
        positions: torch.Tensor,
        lm_head: nn.Module,  # 共享主模型的lm_head
        labels: torch.Tensor | None = None,  # 训练时的标签
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            main_hidden_states: 主模型输出 [batch, seq_len, hidden_size]
            positions: 位置编码 [batch, seq_len]
            lm_head: 主模型的lm_head（共享权重）
            labels: 训练标签 [batch, seq_len]
        Returns:
            mtp_logits: MTP预测的多token logits [batch, seq_len-D, D, vocab_size]
            mtp_loss: MTP损失（训练时）
        """
        batch_size, seq_len, hidden_size = main_hidden_states.shape
        mtp_loss = None

        # 1. 投影到MTP空间
        mtp_hidden = self.mtp_proj(main_hidden_states)[0]  # [batch, seq_len, hidden_size]

        # 2. MTP层前向（逐层预测未来token）
        mtp_logits_list = []
        current_hidden = mtp_hidden[:, :-self.mtp_prediction_length, :]  # 截断最后D个token（无未来标签）
        current_positions = positions[:, :-self.mtp_prediction_length]

        for d in range(self.mtp_prediction_length):
            # MTP解码层前向
            for layer in self.mtp_layers:
                current_hidden = layer(current_hidden, current_positions)
            
            # 归一化 + 计算当前步logits
            current_hidden_norm = self.norm(current_hidden)
            step_logits = lm_head(current_hidden_norm)  # [batch, seq_len-D, vocab_size]
            mtp_logits_list.append(step_logits)

            # 下一步输入：用当前步logits的argmax作为伪输入（因果链）
            if d < self.mtp_prediction_length - 1:
                pseudo_tokens = step_logits.argmax(dim=-1)  # [batch, seq_len-D]
                current_hidden = lm_head.embed_tokens(pseudo_tokens)  # 复用embedding层

        # 3. 拼接多token logits
        mtp_logits = torch.stack(mtp_logits_list, dim=2)  # [batch, seq_len-D, D, vocab_size]

        # 4. 计算MTP损失（训练时）
        if labels is not None and self.training:
            # 标签截断：取第1~D个未来token作为MTP标签
            mtp_labels = labels.unfold(1, self.mtp_prediction_length + 1, 1)[:, :, 1:]
            mtp_labels = mtp_labels.reshape(-1)
            mtp_logits_flat = mtp_logits.reshape(-1, self.config.vocab_size)  # [batch*(seq_len-D)*D, vocab_size]
            
            # 计算交叉熵损失（忽略padding token）
            ignore_index = -100
            valid_mask = mtp_labels != ignore_index
            mtp_loss = F.cross_entropy(
                mtp_logits_flat[valid_mask],
                mtp_labels[valid_mask],
                reduction="mean"
            ) * self.mtp_loss_weight

        return mtp_logits, mtp_loss


# ======================== 原有LLaMA模块（兼容MTP） ========================
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
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


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
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        
        # MTP兼容：KV头数适配
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None) or (self.hidden_size // self.total_num_heads)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Llama4 scaling兼容
        llama_4_scaling_config = getattr(config, "llama_4_scaling", None)
        self.do_llama_4_scaling = llama_4_scaling_config is not None
        if self.do_llama_4_scaling:
            self.llama_4_scaling_original_max_position_embeddings = llama_4_scaling_config["original_max_position_embeddings"]
            self.llama_4_scaling_beta = llama_4_scaling_config["beta"]

        # QKV投影
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

        # 滑动窗口兼容
        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            effective_layer_idx = layer_idx - getattr(config, "target_layer_count", 0)
            assert effective_layer_idx < len(layer_types)
            if layer_types[effective_layer_idx] == "sliding_attention":
                sliding_window = config.sliding_window

        # 注意力层
        attn_cls = EncoderOnlyAttention if attn_type == AttentionType.ENCODER_ONLY else Attention
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

    def _get_llama_4_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        scaling = 1 + self.llama_4_scaling_beta * torch.log(
            1 + torch.floor(positions / self.llama_4_scaling_original_max_position_embeddings)
        )
        return scaling.unsqueeze(-1)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        
        if self.do_llama_4_scaling:
            attn_scale = self._get_llama_4_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(self, config: LlamaConfig, quant_config: QuantizationConfig | None):
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        bias_o_proj = attention_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        attn_type = AttentionType.DECODER if getattr(config, "is_causal", True) else AttentionType.ENCODER_ONLY

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
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
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        return vllm_config.quant_config


def llama_model_invariants(input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    if input_ids is not None:
        torch._check(positions.size()[0] == input_ids.size()[0])


@support_torch_compile(shape_invariants=llama_model_invariants)
class LlamaModel(nn.Module):
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

        # 嵌入层
        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # 主模型层
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        # 归一化层
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # MTP模块（仅在启用时初始化）
        self.enable_mtp = getattr(config, "enable_mtp", False)
        if self.enable_mtp and get_pp_group().is_last_rank:
            self.mtp_module = MTPModule(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mtp_module",
            )
        else:
            self.mtp_module = None

        self.aux_hidden_state_layers = tuple[int, ...]()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # MTP训练时的标签
    ) -> tuple[torch.Tensor | IntermediateTensors, torch.Tensor | None]:
        """
        扩展forward，返回主模型输出 + MTP损失（训练时）
        """
        mtp_loss = None

        # 主模型前向
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual}), mtp_loss

        # 主模型输出归一化
        hidden_states, _ = self.norm(hidden_states, residual)

        # MTP模块前向（仅在最后PP rank且启用MTP时）
        if self.enable_mtp and self.mtp_module is not None and self.training:
            # 共享lm_head（需从LlamaForCausalLM传入，这里先占位，实际在LlamaForCausalLM中调用）
            # 注：实际使用时需修改LlamaForCausalLM的forward，传入lm_head
            pass

        if len(aux_hidden_states) > 0:
            return (hidden_states, aux_hidden_states), mtp_loss
        return hidden_states, mtp_loss

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # MTP模块权重映射
        if self.enable_mtp and self.mtp_module is not None:
            stacked_params_mapping.extend([
                (".mtp_proj", ".mtp_proj.weight", 0),
                (".mtp_layers", ".mtp_layers", None),
            ])
        
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # 跳过rotary_emb缓存
            if "rotary_emb.inv_freq" in name or "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # 量化scale加载
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            # FP8 kv-scale重映射
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # 堆叠参数加载
            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if shard_id is not None:
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight)
                loaded = True
                break

            if not loaded:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "q_fake_quantizer.qscale_act": "attn.q_scale",
        "k_fake_quantizer.qscale_act": "k_scale",
        "v_fake_quantizer.qscale_act": "v_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
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

        # 主模型初始化
        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        # LM Head
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _init_model(self, vllm_config: VllmConfig, prefix: str = "", layer_type: type[nn.Module] = LlamaDecoderLayer):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # MTP训练标签
    ) -> tuple[torch.Tensor | IntermediateTensors, torch.Tensor | None]:
        """
        扩展forward，支持MTP训练：
        - 训练时返回 (主模型输出, MTP损失)
        - 推理时返回主模型输出（MTP模块自动关闭）
        """
        # 主模型前向
        model_output, mtp_loss = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, labels
        )

        # MTP模块前向（仅在训练时，且启用MTP）
        if (
            self.model.enable_mtp 
            and self.model.mtp_module is not None 
            and self.training 
            and isinstance(model_output, torch.Tensor)
        ):
            _, mtp_loss = self.model.mtp_module(
                main_hidden_states=model_output,
                positions=positions,
                lm_head=self.lm_head,
                labels=labels,
            )

        # 推理时直接返回主模型输出
        if not self.training:
            return model_output

        # 训练时返回 (主模型输出, MTP损失)
        return model_output, mtp_loss

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """主模型logits计算（兼容MTP）"""
        if self.model.enable_mtp and self.training:
            # 训练时：主模型logits + MTP logits（但仅主模型logits用于标准损失）
            main_logits = self.logits_processor(self.lm_head, hidden_states)
            return main_logits
        else:
            # 推理时：仅主模型logits
            return self.logits_processor(self.lm_head, hidden_states)

    def compute_mtp_logits(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """推理时MTP草稿生成（投机解码）"""
        if not self.model.enable_mtp or self.model.mtp_module is None:
            raise ValueError("MTP is not enabled for this model")
        
        mtp_logits, _ = self.model.mtp_module(
            main_hidden_states=hidden_states,
            positions=positions,
            lm_head=self.lm_head,
            labels=None,
        )
        return mtp_logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights
        )

    def maybe_remap_mistral(self, name: str, loaded_weight: torch.Tensor) -> tuple[str, torch.Tensor]:
        def permute(w: torch.Tensor, n_heads: int, attn_out: int):
            attn_in = self.config.head_dim * n_heads
            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        # MTP兼容的权重置换
        if getattr(self.config, "enable_mtp", False):
            mtp_group_size = getattr(self.config, "mtp_group_size", 1)
            def permute_mtp(w: torch.Tensor, n_heads: int, attn_out: int):
                num_kv_heads = (n_heads + mtp_group_size - 1) // mtp_group_size
                attn_in = self.config.head_dim * n_heads
                return w.view(n_heads, self.config.head_dim, attn_out) \
                       .reshape(num_kv_heads, mtp_group_size, self.config.head_dim, attn_out) \
                       .transpose(1, 2) \
                       .reshape(attn_in, attn_out)
            permute = permute_mtp

        mapping = self.mistral_mapping
        modules = name.split(".")

        # 权重置换（适配Mistral/LLaMA格式）
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_key_value_heads, self.config.hidden_size
            )
        elif "wk" in modules and modules[-1] == "qscale_weight" and loaded_weight.numel() > 1:
            loaded_weight = permute(loaded_weight, self.config.num_key_value_heads, 1)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_attention_heads, self.config.hidden_size
            )
        elif "wq" in modules and modules[-1] == "qscale_weight" and loaded_weight.numel() > 1:
            loaded_weight = permute(loaded_weight, self.config.num_attention_heads, 1)

        # 名称映射
        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None
            combined_item = f"{item}.{next_item}" if next_item is not None else None

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight


class LlamaBidirectionalForSequenceClassification(as_seq_cls_model(LlamaForCausalLM)):
    pass


class LlamaBidirectionalModel(as_embedding_model(LlamaForCausalLM)):
    pass