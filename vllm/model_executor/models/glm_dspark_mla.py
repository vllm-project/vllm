# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM DSpark with compressed MLA caches and per-layer block attention."""

import torch
from torch import nn
from torch.nn import functional as F

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.qwen3 import Qwen3MLP
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.v1.attention.backends.mla.dspark import DSparkMLABackend, attention
from vllm.v1.kv_cache_interface import MLAAttentionSpec

logger = init_logger(__name__)


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    )
    return normalized.to(x.dtype) * weight.to(x.dtype)


class MLARMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms(x, self.weight, self.eps)


class DSparkMLACache(Attention):
    """Register the latent cache with the standard vLLM cache manager."""

    def get_kv_cache_spec(self, vllm_config):
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        )

    def bind_kv_cache(self, cache):
        if cache.ndim == 4 and cache.shape[1] == 1:
            cache = cache.squeeze(1)
        if cache.ndim != 3 or cache.shape[-1] != 576:
            raise ValueError(f"Expected latent KV cache [..., 576], got {cache.shape}")
        self.kv_cache = cache

    def get_attn_backend(self):
        return DSparkMLABackend


class GlmDSparkMLAAttention(nn.Module):
    def __init__(self, config, layer_idx, cache_config, prefix):
        super().__init__()
        dimensions = (
            config.q_lora_rank,
            config.kv_lora_rank,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.v_head_dim,
        )
        if dimensions != (2048, 512, 192, 64, 256):
            raise ValueError(f"Unsupported GLM DSpark MLA dimensions: {dimensions}")
        self.layer_name = prefix
        self.num_heads = (
            config.num_attention_heads // get_tensor_model_parallel_world_size()
        )
        self.sliding_window, self.causal = _resolve_layer_attention(config, layer_idx)
        self.eps = config.rms_norm_eps
        self.rope_theta = config.rope_parameters["rope_theta"]
        self.is_neox_style = getattr(config, "is_neox_style", True)
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = MLARMSNorm(config.q_lora_rank, self.eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * 256,
            bias=False,
            prefix=prefix + ".q_b_proj",
        )
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, 576, bias=False)
        self.kv_a_layernorm = MLARMSNorm(512, self.eps)
        self.kv_b_proj = ColumnParallelLinear(
            512,
            config.num_attention_heads * 448,
            bias=False,
            prefix=prefix + ".kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * 256,
            config.hidden_size,
            bias=False,
            prefix=prefix + ".o_proj",
        )
        self.attn = DSparkMLACache(
            self.num_heads,
            128,
            256**-0.5,
            num_kv_heads=1,
            cache_config=cache_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=prefix + ".attn",
        )

    def rope(self, x, positions):
        inv = self.rope_theta ** (
            -torch.arange(0, 64, 2, device=x.device, dtype=torch.float32) / 64
        )
        phase = positions.float().unsqueeze(-1) * inv
        if self.is_neox_style:
            cosine = torch.cat([phase.cos(), phase.cos()], -1).to(x.dtype).unsqueeze(1)
            sine = torch.cat([phase.sin(), phase.sin()], -1).to(x.dtype).unsqueeze(1)
            a, b = x.chunk(2, -1)
            rotated = torch.cat([-b, a], -1)
        else:
            cosine = phase.cos().repeat_interleave(2, -1).to(x.dtype).unsqueeze(1)
            sine = phase.sin().repeat_interleave(2, -1).to(x.dtype).unsqueeze(1)
            rotated = torch.stack([-x[..., 1::2], x[..., ::2]], -1).flatten(-2)
        return x * cosine + rotated * sine

    def store(self, hidden_states, positions, slots):
        z = self.kv_a_proj_with_mqa(hidden_states)
        latent = self.kv_a_layernorm(z[:, :512])
        rotary = self.rope(z[:, 512:].unsqueeze(1), positions).squeeze(1)
        z = torch.cat([latent, rotary], -1)
        n = min(len(z), slots.numel())
        slots = slots[:n].long()
        valid = slots >= 0
        block_size = self.attn.kv_cache.shape[1]
        self.attn.kv_cache[slots[valid] // block_size, slots[valid] % block_size] = z[
            :n
        ][valid]

    def forward(self, positions, hidden_states):
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            return torch.zeros_like(hidden_states)
        metadata = (
            metadata[self.attn.layer_name] if isinstance(metadata, dict) else metadata
        )
        if metadata is None:
            return torch.zeros_like(hidden_states)
        self.store(hidden_states, positions, context.slot_mapping[self.attn.layer_name])
        q = F.linear(
            self.q_a_layernorm(self.q_a_proj(hidden_states)), self.q_b_proj.weight
        )
        q = q.view(-1, self.num_heads, 256)
        nope, rotary = q.split([192, 64], -1)
        rotary = self.rope(rotary, positions)
        up = self.kv_b_proj.weight.view(self.num_heads, 448, 512)
        latent_q = torch.bmm(nope.transpose(0, 1), up[:, :192]).transpose(0, 1)
        query = torch.cat([latent_q, rotary], -1).contiguous()
        latent = attention(
            query,
            self.attn.kv_cache,
            metadata.block_table_tensor,
            metadata.query_start_loc,
            metadata.seq_lens,
            self.causal,
            self.sliding_window or 0,
        )
        values = torch.bmm(latent.transpose(0, 1), up[:, 192:].transpose(1, 2))
        values = values.transpose(0, 1).reshape(-1, self.num_heads * 256)
        return self.o_proj(values)[0]


class GlmDSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config,
        *,
        config,
        layer_idx,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        nn.Module.__init__(self)
        if quant_config is not None:
            raise ValueError("GLM MLA DSpark currently requires BF16 draft weights")
        self.hidden_size = config.hidden_size
        self.self_attn = GlmDSparkMLAAttention(
            config,
            layer_idx,
            cache_config,
            prefix + ".self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=None,
            prefix=prefix + ".mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class GlmDSparkMLAModel(Qwen3DSparkModel):
    decoder_layer_cls = GlmDSparkDecoderLayer

    def precompute_and_store_context_kv(
        self, context_states, context_positions, context_slot_mapping=None
    ):
        context_states = rms(
            context_states, self.hidden_norm.weight, self.config.rms_norm_eps
        )
        for i, layer in enumerate(self.layers):
            slots = (
                context_slot_mapping[i]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slots is not None:
                layer.self_attn.store(context_states, context_positions, slots)

    def _build_fused_kv_buffers(self):
        # GQA's fused QKV buffers are replaced by each layer's latent projection.
        pass


class GlmDSparkMLAForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = GlmDSparkMLAModel

    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("GLM MLA DSpark currently supports PP1 and DCP1")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("GLM MLA DSpark currently requires --enforce-eager")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        seen = set()

        def tracked():
            for name, value in weights:
                seen.add(name)
                yield name, value

        result = super().load_weights(tracked())
        for i in range(self.config.num_hidden_layers):
            for suffix in (
                "q_a_proj.weight",
                "q_a_layernorm.weight",
                "q_b_proj.weight",
                "kv_a_proj_with_mqa.weight",
                "kv_a_layernorm.weight",
                "kv_b_proj.weight",
                "o_proj.weight",
            ):
                if f"layers.{i}.self_attn.{suffix}" not in seen:
                    raise ValueError(f"Missing MLA weight: layer {i}, {suffix}")
        logger.info(
            "GLM_DSPARK_MLA_NATIVE_LOADED weights=%d latent_cache=512+64", len(seen)
        )
        return result
