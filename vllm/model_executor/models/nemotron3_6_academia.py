# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Inference-only Academia: recurrent gated delta product and attention layers."""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.gated_delta_product import (
    GatedDeltaProductAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron3_6_academia import (
    Nemotron3_6AcademiaConfig,
)

from .interfaces import HasInnerState, IsHybrid
from .nemotron_h import NemotronHMLP
from .utils import extract_layer_index, maybe_prefix


class AcademiaAttention(nn.Module):
    """Grouped-query attention without positional embeddings."""

    def __init__(self, config, vllm_config, prefix, cache_prefixes):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )
        self.per_loop_kv = config.recurrent_kv_cache_mode == "per_loop"
        self.attn = nn.ModuleList(
            [
                Attention(
                    self.num_heads,
                    self.head_dim,
                    scale=self.head_dim**-0.5,
                    num_kv_heads=self.num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    prefix=f"{p}.attn",
                )
                for p in (cache_prefixes if self.per_loop_kv else cache_prefixes[:1])
            ]
        )

    def forward(self, hidden_states, loop_idx=0):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Each loop overwrites this step's KV; earlier steps retain last-loop KV.
        output, _ = self.o_proj(self.attn[loop_idx if self.per_loop_kv else 0](q, k, v))
        return output


def _sinusoidal_embedding(value: float, dim: int, like: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(100.0)
        * torch.arange(half, device=like.device, dtype=torch.float32)
        / half
    )
    args = value * freqs * 100.0
    return torch.cat([torch.cos(args), torch.sin(args)]).to(like.dtype)


class AcademiaLoopConditioner(nn.Module):
    """`(progress, budget) -> c`: sinusoidal embeddings, Linear, SiLU, Linear."""

    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, progress: float, budget: float) -> torch.Tensor:
        like = self.mlp[0].bias
        return self.mlp(
            torch.cat(
                [
                    _sinusoidal_embedding(progress, self.cond_dim, like),
                    _sinusoidal_embedding(budget, self.cond_dim, like),
                ]
            )
        )


class AcademiaLoopConditioning(nn.Module):
    """Adaptive normalization and a gated low-rank adapter."""

    def __init__(self, config: Nemotron3_6AcademiaConfig, prefix: str):
        super().__init__()
        hidden, cond, rank = (
            config.hidden_size,
            config.loop_conditioning_hidden_size,
            config.loop_adapter_rank,
        )
        self.hidden_size = hidden
        self.adapter_rank = rank
        self.gamma_head = nn.Linear(cond, hidden)
        self.beta_head = nn.Linear(cond, hidden)
        self.gate_head = nn.Linear(cond, hidden)
        self.adapter_gate = nn.Linear(cond, rank) if rank else None
        self.up_proj = (
            ReplicatedLinear(rank, hidden, bias=False, prefix=f"{prefix}.up_proj")
            if rank
            else None
        )
        self.down_proj = (
            ReplicatedLinear(hidden, rank, bias=False, prefix=f"{prefix}.down_proj")
            if rank
            else None
        )
        self.register_buffer(
            "table",
            torch.zeros(config.n_recurrent_loops + 1, 3 * hidden + rank),
            persistent=False,
        )

    def update(self, conds):
        rows = []
        for c in conds:
            parts = [1 + self.gamma_head(c), self.beta_head(c), 1 + self.gate_head(c)]
            if self.adapter_gate is not None:
                parts.append(self.adapter_gate(c))
            rows.append(torch.cat(parts))
        self.table.copy_(torch.stack(rows))

    def forward(self, hidden_states, step):
        scale, shift, gate, adapter_gate = self.table[step].split(
            [self.hidden_size] * 3 + [self.adapter_rank]
        )
        hidden_states = scale * hidden_states + shift
        if self.down_proj is not None and self.up_proj is not None:
            adapter, _ = self.down_proj(hidden_states)
            adapter, _ = self.up_proj(adapter_gate * adapter)
            hidden_states = hidden_states + adapter
        return hidden_states, gate


class AcademiaLoopSeam(nn.Module):
    """Conditioned recurrent entry and exit."""

    def __init__(self, config: Nemotron3_6AcademiaConfig):
        super().__init__()
        hidden, cond = config.hidden_size, config.loop_conditioning_hidden_size
        self.hidden_size = hidden
        self.entry_norm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.entry_gamma = nn.Linear(cond, hidden)
        self.entry_beta = nn.Linear(cond, hidden)
        self.exit_gate = nn.Linear(cond, hidden)
        self.input_mix = nn.Linear(cond, hidden) if config.loop_input_mixing else None
        self.register_buffer(
            "table",
            torch.zeros(
                config.n_recurrent_loops,
                hidden * (4 if self.input_mix is not None else 3),
            ),
            persistent=False,
        )

    def update(self, conds):
        rows = []
        for step, c in enumerate(conds):
            parts = [
                1 + self.entry_gamma(c),
                self.entry_beta(c),
                (1.0 if step == 0 else 0.0) + self.exit_gate(c),
            ]
            if self.input_mix is not None:
                parts.append(self.input_mix(c))
            rows.append(torch.cat(parts))
        self.table.copy_(torch.stack(rows))

    def entry(self, loop_input, input_h, step):
        parts = self.table[step].split(self.hidden_size)
        hidden_states = parts[0] * self.entry_norm(loop_input) + parts[1]
        if self.input_mix is not None:
            hidden_states = hidden_states + parts[3] * input_h
        return hidden_states

    def exit(self, state, new_state, step):
        gate = self.table[step, 2 * self.hidden_size : 3 * self.hidden_size]
        return state + gate * (new_state - state)


class AcademiaStateUpdate(nn.Module):
    """`state + sigmoid(W x + b) * softplus(base_alpha_raw) * (new - state)`, with
    `x` the loop input."""

    def __init__(self, config: Nemotron3_6AcademiaConfig, prefix: str) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=True, prefix=f"{prefix}.proj"
        )
        self.base_alpha_raw = nn.Parameter(torch.empty(config.hidden_size))

    def forward(
        self, state: torch.Tensor, new_state: torch.Tensor, loop_input: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.proj(loop_input)
        beta = torch.sigmoid(logits) * F.softplus(self.base_alpha_raw)
        return state + beta * (new_state - state)


class AcademiaDecoderLayer(nn.Module):
    def __init__(self, config, vllm_config, prefix, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        n_in, n_rec = config.n_input_layers, config.n_recurrent_layers
        loops = config.n_recurrent_loops if n_in <= layer_idx < n_in + n_rec else 1
        # Virtual indices distinguish caches; projections keep physical indices.
        indices = [layer_idx + loop * config.num_hidden_layers for loop in range(loops)]
        cache_prefixes = [
            f"{prefix}.mixer".replace(f"layers.{layer_idx}.", f"layers.{i}.", 1)
            for i in indices
        ]
        if self.layer_type == "linear_attention":
            self.mixer = GatedDeltaProductAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.mixer",
                cache_prefixes=cache_prefixes,
            )
        elif self.layer_type == "full_attention":
            self.mixer = AcademiaAttention(
                config, vllm_config, f"{prefix}.mixer", cache_prefixes
            )
        else:
            self.mixer = NemotronHMLP(
                config=None,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                prefix=f"{prefix}.mixer",
            )
        self.loop_conditioning = (
            AcademiaLoopConditioning(config, f"{prefix}.loop_conditioning")
            if layer_idx >= n_in
            else None
        )

    def forward(self, hidden_states, loop_idx=0, cond_idx=0):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.loop_conditioning is not None:
            hidden_states, gate = self.loop_conditioning(hidden_states, cond_idx)
        if self.layer_type == "mlp":
            hidden_states = self.mixer(hidden_states)
        else:
            hidden_states = self.mixer(hidden_states, loop_idx)
        if self.loop_conditioning is not None:
            hidden_states = hidden_states * gate
        return residual + hidden_states


@support_torch_compile
class Nemotron3_6AcademiaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                AcademiaDecoderLayer(config, vllm_config, f"{prefix}.layers.{i}", i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.loop_conditioner = AcademiaLoopConditioner(
            config.loop_conditioning_hidden_size
        )
        self.loop_seam = AcademiaLoopSeam(config)
        self.state_update = AcademiaStateUpdate(config, f"{prefix}.state_update")

    @torch.no_grad()
    def build_loop_tables(self):
        config = self.config
        count = config.n_recurrent_loops
        budget = (
            min(count, config.loop_conditioning_max_loops)
            / config.loop_conditioning_max_loops
        )
        conds = [
            self.loop_conditioner((step + 1) / count, budget) for step in range(count)
        ]
        conds.append(self.loop_conditioner(1.0, budget))
        for layer in self.layers:
            if layer.loop_conditioning is not None:
                layer.loop_conditioning.update(conds)
        self.loop_seam.update(conds[:-1])

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        config = self.config
        n_in, n_rec = config.n_input_layers, config.n_recurrent_layers
        for layer in self.layers[:n_in]:
            h = layer(h)
        input_h = state = h
        for loop in range(config.n_recurrent_loops):
            loop_input = state + input_h
            h = self.loop_seam.entry(loop_input, input_h, loop) if loop else loop_input
            for layer in self.layers[n_in : n_in + n_rec]:
                h = layer(h, loop, loop)
            new_state = self.state_update(state, h, loop_input)
            state = self.loop_seam.exit(state, new_state, loop)
        h = state
        for layer in self.layers[n_in + n_rec :]:
            h = layer(h, cond_idx=config.n_recurrent_loops)
        return self.norm_f(h)


class Nemotron3_6AcademiaForCausalLM(nn.Module, HasInnerState, IsHybrid):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = self.config = vllm_config.model_config.hf_config
        config.validate()
        if vllm_config.quant_config is not None:
            raise ValueError("Academia does not support quantized checkpoints")
        if vllm_config.speculative_config is not None:
            raise ValueError("Academia currently supports plain decoding only")
        self.model = Nemotron3_6AcademiaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        # Also called by loaders that restore parameters without load_weights.
        self.model.build_loop_tables()

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        c = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.gated_delta_product_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
            c.linear_num_householder,
            c.linear_conv_kernel_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if self.config.tie_word_embeddings and name.startswith("lm_head."):
                continue
            if name.startswith("model.layers."):
                layer_idx = extract_layer_index(name)
                stem, sep, leaf = name.rpartition(".mixer.")
                role = leaf.split(".")[0]
                layer_type = self.config.layer_types[layer_idx]
                if (
                    sep
                    and layer_type == "linear_attention"
                    and role
                    in ("z_proj", "v_proj", "k_proj", "q_proj", "b_proj", "a_proj")
                ):
                    self.model.layers[layer_idx].mixer.load_in_proj_weight(
                        role[0], weight
                    )
                    loaded.add(f"{stem}.mixer.in_proj.weight")
                    continue
                if (
                    sep
                    and layer_type == "full_attention"
                    and role in ("q_proj", "k_proj", "v_proj")
                ):
                    name = f"{stem}.mixer.qkv_proj.weight"
                    params[name].weight_loader(params[name], weight, role[0])
                    loaded.add(name)
                    continue
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        self.model.build_loop_tables()
        return loaded
