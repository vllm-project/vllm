# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dory: normalized latent recurrent transformer."""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

EPS = 1e-6


def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NGPT normalization: x * rsqrt(sum(x^2) + eps), fp32 accumulation."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)
    return y.to(x.dtype)


def slerp_norm(a: torch.Tensor, b: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
    return slerp_norm(a, b, (alpha.float() * alpha_scale).abs())


class DoryAttention(nn.Module):
    """S / * layer: GQA attention with nGPT qk-norm, output gate, fngpt justnorm."""

    def __init__(
        self, config, layer_idx, cache_config, quant_config, prefix, num_loops=1
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        std = config.init_method_std

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.q_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.sqk = nn.Parameter(torch.empty(self.q_size))
        self.s_gate = nn.Parameter(torch.empty(self.q_size))
        self._sqk_factor = self.head_dim**0.5 / std
        g0 = config.attention_output_gate_logit_scale_init
        if g0 is None:
            g0 = config.hidden_size**0.5
        self._s_gate_factor = g0**0.5 / std

        self.is_swa = config.hybrid_layer_pattern[layer_idx] == "S"
        profile = config.rope_profile_layers[layer_idx]
        rope_theta = config.rope_theta_2 if profile == 2 else config.rope_theta
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=rope_theta,
                partial_rotary_factor=(
                    config.partial_rotary_factor_2
                    if profile == 2
                    else config.partial_rotary_factor
                ),
            ),
            is_neox_style=True,
        )

        self.per_loop_kv = config.recurrent_kv_cache_mode == "per_loop"
        # Match LoopCoder's virtual layer indices for independent loop caches.
        # Projection prefixes keep the physical layer index to share weights.
        self.attn = nn.ModuleList()
        for loop_idx in range(num_loops if self.per_loop_kv else 1):
            unique_layer_idx = loop_idx * config.num_hidden_layers + layer_idx
            unique_prefix = prefix.replace(
                f".layers.{layer_idx}.", f".layers.{unique_layer_idx}."
            )
            self.attn.append(
                Attention(
                    self.num_heads,
                    self.head_dim,
                    scale=1.0 / math.sqrt(self.head_dim),
                    num_kv_heads=self.num_kv_heads,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    per_layer_sliding_window=(
                        config.swa_window_size if self.is_swa else None
                    ),
                    prefix=f"{unique_prefix}.attn",
                )
            )

    def forward(self, positions, hidden_states, loop_idx=0):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate, _ = self.g_proj(hidden_states)

        q, k = self.rotary_emb(positions, q, k)

        # nGPT qk-norm AFTER rope: q scaled by (sqk*f)^2 post-justnorm, k plain justnorm
        t = q.shape[0]
        q = q.view(t, self.num_heads, self.head_dim)
        sqk = (self.sqk.float() * self._sqk_factor).view(
            1, self.num_heads, self.head_dim
        )
        q = (sqk.square() * justnorm(q).float()).to(q.dtype).view(t, self.q_size)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        k = justnorm(k).view(t, self.kv_size)

        # In last-loop mode every loop writes this step's KV to the same cache,
        # overwriting the previous loop's. Loop i thus attends to loop-i KV for
        # this step's tokens and last-loop KV for earlier tokens.
        out = self.attn[loop_idx if self.per_loop_kv else 0](q, k, v)

        # output gate (sigmoid, pre-o_proj), then fngpt justnorm over np*hn
        sg = (self.s_gate.float() * self._s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        out = justnorm(out)
        out, _ = self.o_proj(out)
        return out


class DoryMLP(nn.Module):
    """'-' layer: swiglu with nGPT suv scaling and fngpt justnorm."""

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.ffn = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.ffn] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.ffn,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.suv_gate = nn.Parameter(torch.empty(self.ffn))
        self.suv_up = nn.Parameter(torch.empty(self.ffn))
        self._suv_factor = config.hidden_size**0.5

    def forward(self, hidden_states):
        m, _ = self.gate_up_proj(hidden_states)
        g, u = m.chunk(2, dim=-1)
        g = (g.float() * (self.suv_gate.float() * self._suv_factor)).to(g.dtype)
        u = (u.float() * (self.suv_up.float() * self._suv_factor)).to(u.dtype)
        m = (F.silu(g.float()) * u.float()).to(m.dtype)
        m = justnorm(m)
        out, _ = self.down_proj(m)
        return out


class DoryDecoderLayer(nn.Module):
    """One physical layer: a single sublayer (attention OR mlp) + slerp residual."""

    def __init__(
        self, config, layer_idx, cache_config, quant_config, prefix, num_loops=1
    ):
        super().__init__()
        char = config.hybrid_layer_pattern[layer_idx]
        self.char = char
        self._alpha_scale = config.ngpt_alpha_init / config.init_method_std
        if char in ("S", "*"):
            self.mixer = DoryAttention(
                config,
                layer_idx,
                cache_config,
                quant_config,
                f"{prefix}.mixer",
                num_loops=num_loops,
            )
            self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
        elif char == "-":
            self.mixer = DoryMLP(config, quant_config, f"{prefix}.mixer")
            self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))
        else:
            raise NotImplementedError(f"pattern char {char!r} not supported by Dory")

    def forward(self, positions, hidden_states, loop_idx=0):
        residual = hidden_states
        if self.char in ("S", "*"):
            out = self.mixer(positions, hidden_states, loop_idx=loop_idx)
            return slerp_alpha(residual, out, self.attn_alpha, self._alpha_scale)
        out = self.mixer(hidden_states)
        return slerp_alpha(residual, out, self.mlp_alpha, self._alpha_scale)


@support_torch_compile
class DoryBackbone(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        self.config = config
        self.num_loops = config.n_recurrent_loops
        n_in = config.n_input_layers
        n_rec = config.n_recurrent_layers
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DoryDecoderLayer(
                    config,
                    i,
                    cache_config,
                    quant_config,
                    f"{prefix}.layers.{i}",
                    num_loops=self.num_loops if n_in <= i < n_in + n_rec else 1,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embeddings(input_ids)
        c = self.config
        n_in, n_rec = c.n_input_layers, c.n_recurrent_layers

        for layer in self.layers[:n_in]:
            h = layer(positions, h)
        input_h = h

        state = input_h
        for i in range(self.num_loops):
            combined = justnorm(state + input_h)
            h = combined
            for layer in self.layers[n_in : n_in + n_rec]:
                h = layer(positions, h, loop_idx=i)
            state = justnorm(h)

        h = state
        for layer in self.layers[n_in + n_rec :]:
            h = layer(positions, h)
        # no final norm (ngpt_enable_final_norm=False)
        return h


class DoryForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        config.validate_architecture()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Dory currently supports tensor_parallel_size=1 only")
        if vllm_config.quant_config is not None:
            raise ValueError("Dory weight quantization is not supported")
        self.backbone = DoryBackbone(
            config,
            vllm_config.cache_config,
            vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.backbone.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        return self.backbone(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            logits = logits.float() * (
                self.lm_head.logit_scale.float() / self.config.init_method_std
            )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [
            # (target, artifact shard name, shard id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            mapped = False
            for target, shard_name, shard_id in stacked:
                if f".{shard_name}." not in name:
                    continue
                pname = name.replace(f".{shard_name}.", f".{target}.")
                param = params[pname]
                if target == "qkv_proj":
                    heads = (
                        self.config.num_attention_heads
                        if shard_id == "q"
                        else self.config.num_key_value_heads
                    )
                    rows = heads * self.config.attention_head_dim
                else:
                    rows = self.config.intermediate_size
                self._check_weight_shape(name, w, (rows, self.config.hidden_size))
                param.weight_loader(param, w, shard_id)
                loaded.add(pname)  # report the MODEL param name, not the artifact name
                mapped = True
                break
            if mapped:
                continue

            param = params[name]
            shape = (
                (self.config.vocab_size, self.config.hidden_size)
                if name in ("backbone.embeddings.weight", "lm_head.weight")
                else tuple(param.shape)
            )
            self._check_weight_shape(name, w, shape)
            getattr(param, "weight_loader", default_weight_loader)(param, w)
            loaded.add(name)
        return loaded

    @staticmethod
    def _check_weight_shape(name, weight, expected):
        # TP loaders may narrow oversized tensors; reject incompatible exports.
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"Dory weight {name}: expected shape {expected}, "
                f"got {tuple(weight.shape)}"
            )
