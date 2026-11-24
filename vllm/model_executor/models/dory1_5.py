# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Inference-only Dory1.5 model."""

import math
import types
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
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
from vllm.transformers_utils.configs.dory1_5 import (
    Dory1_5Config,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

from .interfaces import HasInnerState, IsHybrid
from .nemotron_h import NemotronHMLP
from .utils import extract_layer_index, maybe_prefix

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel

GDP_PROJECTION_ROLES = ("z_proj", "v_proj", "k_proj", "q_proj", "b_proj", "a_proj")
QKV_PROJECTION_ROLES = ("q_proj", "k_proj", "v_proj")


class Dory1_5PackedMethod(LinearMethodBase):
    kernel: "MarlinMxfp8LinearKernel | MarlinNvFp4LinearKernel"

    def __init__(self, shape):
        self.shape = shape

    def create_weights(self, *args, **kwargs):
        raise RuntimeError(
            "Dory1.5 packed weights are installed before checkpoint load"
        )

    def process_weights_after_loading(self, layer):
        layer.output_size_per_partition, layer.input_size_per_partition = self.shape
        layer.params_dtype = torch.bfloat16
        self.kernel.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.kernel.apply_weights(layer, x, bias)


class Dory1_5MxFp8Method(Dory1_5PackedMethod):
    def __init__(self, shape):
        super().__init__(shape)
        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp8.marlin import (
            MarlinMxfp8LinearKernel,
        )

        self.kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(shape))


class Dory1_5NvFp4Method(Dory1_5PackedMethod):
    def __init__(self, shape):
        super().__init__(shape)
        from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
        from vllm.model_executor.kernels.linear.nvfp4.marlin import (
            MarlinNvFp4LinearKernel,
        )

        self.kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())

    def process_weights_after_loading(self, layer):
        layer.register_parameter(
            "weight_global_scale", layer._parameters.pop("weight_scale_2")
        )
        super().process_weights_after_loading(layer)


class Dory1_5PackedLinear(nn.Linear):
    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


@dataclass(frozen=True)
class PackedFormat:
    block_size: int
    elements_per_byte: int
    weight_dtype: torch.dtype
    scale_dtype: torch.dtype
    method: type[Dory1_5PackedMethod]
    tensor_scale: bool = False


PACKED_FORMATS = {
    "MXFP8": PackedFormat(32, 1, torch.float8_e4m3fn, torch.uint8, Dory1_5MxFp8Method),
    "NVFP4": PackedFormat(
        16, 2, torch.uint8, torch.float8_e4m3fn, Dory1_5NvFp4Method, True
    ),
}


def install_packed_linear(module, fmt):
    """Allocate source-format storage while retaining vLLM parameter metadata."""
    spec = PACKED_FORMATS[fmt]
    w = module.weight
    n, k = w.shape
    if k % spec.block_size:
        raise ValueError(f"{fmt} block size does not divide input size {k}")
    w.requires_grad_(False)
    w.data = torch.empty(
        n, k // spec.elements_per_byte, dtype=spec.weight_dtype, device=w.device
    )
    module.register_parameter(
        "weight_scale",
        nn.Parameter(
            torch.empty(
                n, k // spec.block_size, dtype=spec.scale_dtype, device=w.device
            ),
            requires_grad=False,
        ),
    )
    if spec.tensor_scale:
        module.register_parameter(
            "weight_scale_2",
            nn.Parameter(
                torch.empty((), dtype=torch.float32, device=w.device),
                requires_grad=False,
            ),
        )
    module.quant_method = spec.method((n, k))
    if isinstance(module, nn.Linear):
        module.__class__ = Dory1_5PackedLinear


class SplitProjection(nn.Module):
    def __init__(self, roles, sizes, k):
        super().__init__()
        self.parts = nn.ModuleDict(
            {role: nn.Linear(k, n, bias=False) for role, n in zip(roles, sizes)}
        )

    def forward(self, x):
        return torch.cat([part(x) for part in self.parts.values()], dim=-1), None


def mapped_name(model, name):
    if name.startswith("model.layers.") and ".mixer." in name:
        stem, leaf = name.rsplit(".mixer.", 1)
        role, _, suffix = leaf.partition(".")
        kind = model.config.layer_types[int(stem.split(".")[-1])]
        if kind == "linear_attention" and role in GDP_PROJECTION_ROLES:
            return f"{stem}.mixer.in_proj.parts.{role}.{suffix}"
        if kind == "full_attention" and role in QKV_PROJECTION_ROLES:
            return f"{stem}.mixer.qkv_proj.parts.{role}.{suffix}"
    return name


def _install_per_matrix_weights(model, vllm_config):
    marker = model.config.per_tensor_quantization
    if (
        vllm_config.parallel_config.tensor_parallel_size != 1
        or vllm_config.parallel_config.pipeline_parallel_size != 1
    ):
        raise ValueError("Dory1.5 packed weights require TP=PP=1")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Dory1.5 packed weights require --dtype bfloat16")
    if model.config.tie_word_embeddings:
        raise ValueError("Per-matrix Dory1.5 checkpoints require untied embeddings")
    allocation = get_hf_file_to_dict(
        marker["allocation"],
        vllm_config.model_config.model,
        revision=vllm_config.model_config.revision,
    )
    if allocation is None:
        raise ValueError("Dory1.5 quantization allocation file was not found")
    if not isinstance(allocation, dict) or not isinstance(
        allocation.get("formats"), dict
    ):
        raise ValueError("Dory1.5 quantization allocation requires a formats mapping")
    for i, layer in enumerate(model.model.layers):
        mixer = layer.mixer
        kind = model.config.layer_types[i]
        if kind == "linear_attention":
            m = mixer.num_householder
            sizes = [
                mixer.value_dim,
                m * mixer.value_dim,
                m * mixer.key_dim,
                mixer.key_dim,
                m * mixer.num_v_heads,
                mixer.num_v_heads,
            ]
            mixer.in_proj = SplitProjection(
                GDP_PROJECTION_ROLES, sizes, model.config.hidden_size
            )
        elif kind == "full_attention":
            mixer.qkv_proj = SplitProjection(
                QKV_PROJECTION_ROLES,
                [mixer.q_size, mixer.kv_size, mixer.kv_size],
                model.config.hidden_size,
            )
    modules = dict(model.named_modules())
    for name, fmt in allocation["formats"].items():
        if fmt == "BF16":
            continue
        if fmt not in PACKED_FORMATS:
            raise ValueError(f"Unsupported Dory1.5 matrix format {fmt!r}: {name}")
        module = modules.get(mapped_name(model, name).rsplit(".", 1)[0])
        if (
            not name.endswith(".weight")
            or not isinstance(module, (nn.Linear, LinearBase, ParallelLMHead))
            or module.weight.ndim != 2
        ):
            raise ValueError(f"Dory1.5 can only pack linear matrix weights: {name}")
        install_packed_linear(module, fmt)


def _load_per_matrix_weights(model, weights):
    params = dict(model.named_parameters())
    loaded = set()
    for name, value in weights:
        target = mapped_name(model, name)
        if target in loaded:
            raise ValueError(f"Duplicate source tensor {name}")
        if target not in params:
            raise ValueError(
                f"Unexpected checkpoint tensor or missing allocation: {name}"
            )
        p = params[target]
        if value.shape != p.shape or (
            value.dtype != p.dtype and not name.endswith(".A_log")
        ):
            raise ValueError(f"Packed shape/dtype mismatch: {name}")
        p.data.copy_(value)
        loaded.add(target)
    if loaded != set(params):
        raise ValueError(f"Incomplete checkpoint: missing {set(params) - loaded}")
    return loaded


class Dory1_5Attention(nn.Module):
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
        stash = None
        if not self.per_loop_kv and len(cache_prefixes) > 1:
            stash = torch.zeros(
                vllm_config.scheduler_config.max_num_batched_tokens,
                2 * self.kv_size,
                dtype=vllm_config.model_config.dtype,
                device=self.qkv_proj.weight.device,
            )
        self.register_buffer("stash", stash, persistent=False)
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

    def forward(self, hidden_states, loop_idx=0, active=None):
        qkv, _ = self.qkv_proj(hidden_states)
        q, kv = qkv.split([self.q_size, 2 * self.kv_size], dim=-1)
        if self.stash is not None:
            stash = self.stash[: kv.shape[0]]
            kv = torch.where(active.unsqueeze(-1), kv, stash)
            stash.copy_(kv)
        k, v = kv.split(self.kv_size, dim=-1)
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


class Dory1_5LoopConditioner(nn.Module):
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


class Dory1_5LoopConditioning(nn.Module):
    """Adaptive normalization and a gated low-rank adapter."""

    def __init__(self, config: Dory1_5Config, prefix: str):
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
            torch.zeros(
                config.n_recurrent_loops * (config.n_recurrent_loops + 3) // 2,
                3 * hidden + rank,
            ),
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
            [self.hidden_size] * 3 + [self.adapter_rank], dim=-1
        )
        hidden_states = scale * hidden_states + shift
        if self.down_proj is not None and self.up_proj is not None:
            adapter, _ = self.down_proj(hidden_states)
            adapter, _ = self.up_proj(adapter_gate * adapter)
            hidden_states = hidden_states + adapter
        return hidden_states, gate


class Dory1_5LoopSeam(nn.Module):
    """Conditioned recurrent entry and exit."""

    def __init__(self, config: Dory1_5Config):
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
                config.n_recurrent_loops * (config.n_recurrent_loops + 3) // 2,
                hidden * (4 if self.input_mix is not None else 3),
            ),
            persistent=False,
        )

    def update(self, conds, steps):
        rows = []
        for step, c in zip(steps, conds):
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
        parts = self.table[step].split(self.hidden_size, dim=-1)
        hidden_states = parts[0] * self.entry_norm(loop_input) + parts[1]
        if self.input_mix is not None:
            hidden_states = hidden_states + parts[3] * input_h
        return hidden_states

    def exit(self, state, new_state, step):
        gate = self.table[step, 2 * self.hidden_size : 3 * self.hidden_size]
        return state + gate * (new_state - state)


class Dory1_5StateUpdate(nn.Module):
    """`state + sigmoid(W x + b) * softplus(base_alpha_raw) * (new - state)`, with
    `x` the loop input."""

    def __init__(self, config: Dory1_5Config, prefix: str) -> None:
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


class Dory1_5DecoderLayer(nn.Module):
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
            self.mixer = Dory1_5Attention(
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
            Dory1_5LoopConditioning(config, f"{prefix}.loop_conditioning")
            if layer_idx >= n_in
            else None
        )

    def forward(self, hidden_states, loop_idx=0, cond_idx=0, active=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.loop_conditioning is not None:
            hidden_states, gate = self.loop_conditioning(hidden_states, cond_idx)
        if self.layer_type == "mlp":
            hidden_states = self.mixer(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.mixer(hidden_states, loop_idx, active)
        else:
            hidden_states = self.mixer(hidden_states, loop_idx)
        if self.loop_conditioning is not None:
            hidden_states = hidden_states * gate
        return residual + hidden_states


def _compiled_stage_class(cls: type[nn.Module], name: str) -> type[nn.Module]:
    """A subclass of a `support_torch_compile` stage class with its own copy of
    `forward`'s code object and qualified name. vLLM keys compiled code by both
    (guards are not evaluated), so stage instances sharing them would all run the
    first instance's graph, which is bound to that instance's caches."""
    base = cls.forward
    forward = types.FunctionType(
        base.__code__.replace(co_qualname=f"{name}.forward"),
        base.__globals__,
        "forward",
        base.__defaults__,
        base.__closure__,
    )
    forward.__annotations__ = dict(base.__annotations__)
    return type(name, (cls,), {"forward": forward})


def loop_table_row(num_loops, step):
    """Packed row for a budget and step; step == budget selects the output block."""
    return (num_loops - 1) * (num_loops + 2) // 2 + step


@support_torch_compile
class Dory1_5InputBlock(nn.Module):
    def __init__(self, *, vllm_config, prefix, model):
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers[: model.config.n_input_layers]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


@support_torch_compile
class Dory1_5Loop(nn.Module):
    def __init__(self, *, vllm_config, prefix, model, loop):
        super().__init__()
        config = model.config
        self.loop = loop
        self.layers = model.layers[
            config.n_input_layers : config.n_input_layers + config.n_recurrent_layers
        ]
        self.loop_seam = model.loop_seam
        self.state_update = model.state_update

    def forward(
        self, state: torch.Tensor, input_h: torch.Tensor, loop_counts: torch.Tensor
    ) -> torch.Tensor:
        rows = loop_table_row(
            loop_counts.long(), loop_counts.clamp(max=self.loop + 1) - 1
        )
        active = loop_counts > self.loop
        loop_input = state + input_h
        h = self.loop_seam.entry(loop_input, input_h, rows) if self.loop else loop_input
        for layer in self.layers:
            h = layer(h, self.loop, rows, active)
        new_state = self.state_update(state, h, loop_input)
        new_state = self.loop_seam.exit(state, new_state, rows)
        return torch.where(active.unsqueeze(-1), new_state, state)


@support_torch_compile
class Dory1_5OutputBlock(nn.Module):
    def __init__(self, *, vllm_config, prefix, model):
        super().__init__()
        config = model.config
        self.layers = model.layers[config.n_input_layers + config.n_recurrent_layers :]
        self.norm = model.norm_f

    def forward(self, state: torch.Tensor, loop_counts: torch.Tensor) -> torch.Tensor:
        rows = loop_table_row(loop_counts.long(), loop_counts)
        for layer in self.layers:
            state = layer(state, cond_idx=rows)
        return self.norm(state)


class Dory1_5Stages(nn.Module):
    """Separately compiled stages sharing physical weights and distinct loop caches."""

    def __init__(self, model, vllm_config, prefix):
        super().__init__()
        from vllm.compilation.backends import set_model_tag

        def stage(cls, name, **kwargs):
            with set_model_tag(name):
                return _compiled_stage_class(cls, name)(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.{name}",
                    model=model,
                    **kwargs,
                )

        self.input_block = stage(Dory1_5InputBlock, "dory1_5_input")
        self.loops = nn.ModuleList(
            [
                stage(Dory1_5Loop, f"dory1_5_loop_{i}", loop=i)
                for i in range(model.config.n_recurrent_loops)
            ]
        )
        self.output_block = stage(Dory1_5OutputBlock, "dory1_5_output")


class Dory1_5Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Dory1_5DecoderLayer(config, vllm_config, f"{prefix}.layers.{i}", i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.loop_conditioner = Dory1_5LoopConditioner(
            config.loop_conditioning_hidden_size
        )
        self.loop_seam = Dory1_5LoopSeam(config)
        self.state_update = Dory1_5StateUpdate(config, f"{prefix}.state_update")
        self.stages = Dory1_5Stages(self, vllm_config, prefix)
        tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "input_h",
            self.embed_tokens.weight.new_zeros(tokens, config.hidden_size),
            persistent=False,
        )
        self.register_buffer("state", torch.zeros_like(self.input_h), persistent=False)

    @torch.no_grad()
    def build_loop_tables(self):
        config = self.config
        conds, steps = [], []
        for count in range(1, config.n_recurrent_loops + 1):
            budget = (
                min(count, config.loop_conditioning_max_loops)
                / config.loop_conditioning_max_loops
            )
            for step in range(count + 1):
                conds.append(
                    self.loop_conditioner(min((step + 1) / count, 1.0), budget)
                )
                steps.append(step)
        for layer in self.layers:
            if layer.loop_conditioning is not None:
                layer.loop_conditioning.update(conds)
        self.loop_seam.update(conds, steps)

    def forward(self, input_ids, positions, inputs_embeds, loop_counts, num_loops):
        num_tokens = positions.shape[-1]
        input_h, state = self.input_h[:num_tokens], self.state[:num_tokens]
        h = self.stages.input_block(input_ids, positions, inputs_embeds)
        input_h.copy_(h)
        state.copy_(h)
        for loop in range(num_loops):
            state.copy_(self.stages.loops[loop](state, input_h, loop_counts))
        return self.stages.output_block(state, loop_counts)


class Dory1_5ModelState(MambaHybridModelState):
    """Hybrid model state plus each request's loop count, delivered to the forward as
    a per-token `loop_counts` tensor (persistent, so CUDA graphs see each step's
    values)."""

    def __init__(self, vllm_config: VllmConfig, model: nn.Module, *args, **kwargs):
        super().__init__(vllm_config, model, *args, **kwargs)
        config = vllm_config.model_config.hf_config
        self.default_num_loops = config.n_recurrent_loops
        self.max_loops = config.n_recurrent_loops
        # Host copies, to pick each batch's loop count (the largest of its requests).
        self.num_loops_by_slot_np = np.full(
            self.max_num_reqs, self.default_num_loops, dtype=np.int32
        )
        self.num_loops_by_slot = torch.full(
            (self.max_num_reqs,),
            self.default_num_loops,
            dtype=torch.int32,
            device=self.device,
        )
        self.loop_counts = torch.full(
            (self.max_num_tokens,),
            self.default_num_loops,
            dtype=torch.int32,
            device=self.device,
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Validated by the config hook at engine admission.
        params = new_req_data.sampling_params
        num_loops = self.model_config.hf_config.get_num_loops(
            params.extra_args if params is not None else None
        )
        self.num_loops_by_slot[req_index].fill_(num_loops)
        self.num_loops_by_slot_np[req_index] = num_loops

    def cudagraph_variants(self) -> tuple[int, ...]:
        # FULL graphs per loop count: a batch runs as many loops as its largest
        # loop count needs.
        return tuple(range(1, self.max_loops + 1))

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        inputs = super().prepare_inputs(input_batch, req_states)
        num_reqs, num_tokens = input_batch.num_reqs, input_batch.num_tokens
        slots = input_batch.idx_mapping_np[:num_reqs]
        loops = self.num_loops_by_slot_np[slots]
        if num_reqs and loops.min() == loops.max():
            # Every request runs the same loop count (the usual case): one fill
            # instead of an eager repeat_interleave (cumsum + gather) per step.
            self.loop_counts[:num_tokens].fill_(int(loops[0]))
        else:
            query_start_loc = input_batch.query_start_loc[: num_reqs + 1]
            self.loop_counts[:num_tokens] = torch.repeat_interleave(
                self.num_loops_by_slot[input_batch.idx_mapping[:num_reqs]],
                query_start_loc[1:] - query_start_loc[:-1],
                output_size=num_tokens,
            )
        # Padding tokens keep earlier (valid) loop counts.
        inputs["loop_counts"] = self.loop_counts[: input_batch.num_tokens_after_padding]
        self.cudagraph_variant = (
            int(loops.max()) if num_reqs else self.default_num_loops
        )
        inputs["num_loops"] = self.cudagraph_variant
        return inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict:
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        inputs["loop_counts"] = self.loop_counts[:num_tokens]
        num_loops = self.cudagraph_variant or self.max_loops
        self.cudagraph_variant = num_loops
        self.loop_counts[:num_tokens].fill_(num_loops)
        inputs["num_loops"] = num_loops
        return inputs


class Dory1_5ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    @staticmethod
    def get_model_state_cls():
        return Dory1_5ModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = self.config = vllm_config.model_config.hf_config
        config.validate()
        if vllm_config.quant_config is not None:
            raise ValueError(
                "Dory1.5 quantization uses per_tensor_quantization; "
                "do not pass --quantization"
            )
        if vllm_config.speculative_config is not None:
            raise ValueError("Dory1.5 currently supports plain decoding only")
        self.model = Dory1_5Model(
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
        if config.per_tensor_quantization:
            _install_per_matrix_weights(self, vllm_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        loop_counts: torch.Tensor,
        num_loops: int,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds, loop_counts, num_loops)

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
        if self.config.per_tensor_quantization:
            return _load_per_matrix_weights(self, weights)
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
