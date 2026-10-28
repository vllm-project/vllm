# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated DSpark KV producer with no query, MLP, or vocabulary weights."""

import regex as re
import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.transformers_utils.config import set_default_rope_theta

from .qwen3_dflash import (
    DFlashQwen3Model,
    _get_dflash_fc_input_size,
    _resolve_layer_attention,
)

logger = init_logger(__name__)


def context_kv_only(vllm_config):
    spec = vllm_config.speculative_config
    enabled = spec is not None and spec.prefill_context_kv_only
    if enabled:
        transfer = vllm_config.kv_transfer_config
        if spec is None or spec.method != "dspark":
            raise ValueError("Context-KV-only requires DSpark")
        if transfer is None or transfer.kv_role != "kv_producer":
            raise ValueError("Context-KV-only is restricted to kv_producer")
        if spec.draft_model_config.architectures != ["Qwen3DSparkModel"]:
            raise ValueError("Context-KV-only supports Qwen3 GQA DSpark")
        if spec.draft_parallel_config.tensor_parallel_size != 1:
            raise ValueError("Context-KV-only requires draft TP1")
        parallel = spec.target_parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("Context-KV-only requires target TP1, PCP1 and DCP1")
        if not vllm_config.use_v2_model_runner:
            raise ValueError("Context-KV-only requires Model Runner V2")
        if spec.draft_model_config.dtype != torch.bfloat16:
            raise ValueError("Context-KV-only currently requires BF16 draft weights")
    return enabled


class ContextKVAttention(nn.Module):
    def __init__(self, config, layer_idx, cache_config, prefix):
        super().__init__()
        assert get_tensor_model_parallel_world_size() == 1
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=getattr(config, "is_neox_style", True),
            rope_parameters=config.rope_parameters,
        )
        self.sliding_window, self.causal = _resolve_layer_attention(config, layer_idx)
        self.v_scale = (getattr(config, "dflash_config", None) or {}).get(
            "attention_value_scale"
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
        )


class DSparkContextKVModel(DFlashQwen3Model):
    def __init__(self, vllm_config, start_layer_id, prefix):
        nn.Module.__init__(self)
        spec = vllm_config.speculative_config
        self.config = spec.draft_model_config.hf_config
        config = self.config
        self.use_aux_hidden_state = True
        self.confidence_head = None
        self.quant_config = None
        if getattr(config, "attention_bias", False):
            raise ValueError("Context-KV-only requires bias-free draft weights")
        if getattr(config, "quantization_config", None):
            raise ValueError("Quantized draft weights are not supported")
        set_default_rope_theta(config, default_theta=1000000)
        self.fc = ReplicatedLinear(
            _get_dflash_fc_input_size(vllm_config),
            config.hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=None,
            prefix=f"{prefix}.fc",
            return_bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList()
        cache = get_current_vllm_config().cache_config
        for i in range(config.num_hidden_layers):
            layer = nn.Module()
            layer.self_attn = ContextKVAttention(
                config, i, cache, f"{prefix}.layers.{start_layer_id + i}.self_attn"
            )
            self.layers.append(layer)
        kv_size = config.num_key_value_heads * config.head_dim
        self._fused_kv_weight = nn.Parameter(
            torch.empty(
                config.num_hidden_layers * 2 * kv_size,
                config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )

    def _build_fused_kv_buffers(self):
        attns = [layer.self_attn for layer in self.layers]
        first = attns[0]
        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._fused_kv_bias = None
        self._k_norm_weights = torch.stack([a.k_norm.weight.data for a in attns])
        self._rope_head_size = first.rotary_emb.head_size
        self._rope_cos_sin_cache = first.rotary_emb.cos_sin_cache
        self._rope_is_neox = first.rotary_emb.is_neox_style
        self._num_attn_layers = len(attns)
        self._kv_size = first.kv_size
        self._head_dim = first.head_dim
        self._num_kv_heads = first.num_kv_heads
        self._rms_norm_eps = self.config.rms_norm_eps
        self._attn_layers = [a.attn for a in attns]

    def load_context_weights(self, weights):
        params = dict(self.named_parameters())
        kv = self.config.num_key_value_heads * self.config.head_dim
        required = {"fc.weight", "hidden_norm.weight"}
        for i in range(self.config.num_hidden_layers):
            suffixes = ("k_proj.weight", "v_proj.weight", "k_norm.weight")
            required.update(f"layers.{i}.self_attn.{suffix}" for suffix in suffixes)
        seen = set()
        for name, tensor in weights:
            name = name.removeprefix("model.")
            if name not in required:
                continue
            if name in seen:
                raise ValueError(f"Duplicate context weight: {name}")
            match = re.fullmatch(r"layers\.(\d+)\.self_attn\.([kv])_proj\.weight", name)
            if match:
                layer, kind = int(match[1]), match[2]
                row = (2 * layer + int(kind == "v")) * kv
                assert self._fused_kv_weight is not None
                destination = self._fused_kv_weight[row : row + kv]
            else:
                destination = params[name]
            if destination.shape != tensor.shape:
                raise ValueError(f"Context weight shape mismatch: {name}")
            if tensor.dtype != destination.dtype:
                raise ValueError(f"Context weight dtype mismatch: {name}")
            destination.data.copy_(tensor)
            seen.add(name)
        if seen != required:
            raise ValueError(f"Missing context weights: {sorted(required - seen)}")
        self._build_fused_kv_buffers()
        count = sum(p.numel() * p.element_size() for p in self.parameters())
        logger.info(
            "DSPARK_CONTEXT_KV_ONLY loaded %d parameter bytes; keys=%s",
            count,
            sorted(seen),
        )
        return {"model." + name for name in params}
