# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dory: normalized latent recurrent transformer."""

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
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
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.looped_self_draft.activation_cache import (
        LoopedSelfDraftActivationCache,
    )

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
    """GQA attention with nGPT qk-norm, output gate, and fngpt justnorm."""

    def __init__(
        self, config, layer_idx, cache_config, quant_config, prefix, num_loops=1
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        std = config.initializer_range

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

        layer_type = config.layer_types[layer_idx]
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters[layer_type],
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
                        config.sliding_window
                        if layer_type == "sliding_attention"
                        else None
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
    """SwiGLU with nGPT suv scaling and fngpt justnorm."""

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
        self.is_attention = config.layer_types[layer_idx] != "mlp"
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.is_attention:
            self.mixer = DoryAttention(
                config,
                layer_idx,
                cache_config,
                quant_config,
                f"{prefix}.mixer",
                num_loops=num_loops,
            )
            self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
        else:
            self.mixer = DoryMLP(config, quant_config, f"{prefix}.mixer")
            self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, positions, hidden_states, loop_idx=0):
        residual = hidden_states
        if self.is_attention:
            out = self.mixer(positions, hidden_states, loop_idx=loop_idx)
            return slerp_alpha(residual, out, self.attn_alpha, self._alpha_scale)
        out = self.mixer(hidden_states)
        return slerp_alpha(residual, out, self.mlp_alpha, self._alpha_scale)


class DoryStage(nn.Module):
    """A tensor-only stage, sharing the backbone without registering its weights.

    Each subclass needs its own forward code object: vLLM's guard-free compile
    cache must not alias stages with different attention-cache bindings.
    """

    def __init__(self, backbone):
        super().__init__()
        self._backbone = (backbone,)


@support_torch_compile
class DoryPrefixStage(DoryStage):
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self._backbone[0]._shared_prefix(input_ids, positions)


@support_torch_compile
class DoryTargetStage(DoryStage):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_hidden_states: torch.Tensor,
    ):
        return self._backbone[0]._target_suffix(
            positions, hidden_states, input_hidden_states
        )


@support_torch_compile
class DoryDraftStage(DoryStage):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_hidden_states: torch.Tensor,
    ):
        return self._backbone[0]._draft_suffix(
            positions, hidden_states, input_hidden_states
        )


@support_torch_compile
class DoryFillStage(DoryStage):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_hidden_states: torch.Tensor,
    ):
        backbone = self._backbone[0]
        return backbone._draft_suffix(
            positions,
            hidden_states,
            input_hidden_states,
            output_end=backbone.draft_fill_end,
        )


@support_torch_compile(
    enable_if=lambda vllm_config: (
        vllm_config.speculative_config is None
        or vllm_config.speculative_config.method != "looped_self_draft"
    )
)
class DoryBackbone(nn.Module):
    def __init__(
        self,
        config,
        cache_config,
        quant_config,
        prefix,
        num_draft_loops=0,
        num_shared_loops=0,
    ):
        super().__init__()
        self.config = config
        self.num_loops = config.n_recurrent_loops
        self.num_draft_loops = num_draft_loops
        self.num_shared_loops = num_shared_loops
        self.draft_activation_cache: LoopedSelfDraftActivationCache | None = None
        self.draft_fill_end = max(
            (i + 1 for i, kind in enumerate(config.layer_types) if kind != "mlp"),
            default=0,
        )
        n_in = config.n_input_layers
        n_rec = config.n_recurrent_layers
        self.out_start = n_in + n_rec
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DoryDecoderLayer(
                    config,
                    i,
                    cache_config,
                    quant_config,
                    f"{prefix}.layers.{i}",
                    num_loops=(
                        self.num_loops + num_draft_loops - self.num_shared_loops
                        if n_in <= i < self.out_start
                        else 2
                        if num_draft_loops and i >= self.out_start
                        else 1
                    ),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.stages = nn.ModuleDict()
        if not self.num_draft_loops:
            return

        for name, stage in (
            ("prefix", DoryPrefixStage),
            ("target", DoryTargetStage),
            ("draft", DoryDraftStage),
            ("fill", DoryFillStage),
        ):
            with set_model_tag(f"dory_{name}"):
                self.stages[name] = stage(self)

    def _loops(
        self, positions, hidden_states, input_hidden_states, start, end, kv_offset=0
    ):
        for loop_idx in range(start, end):
            hidden_states = justnorm(hidden_states + input_hidden_states)
            for layer in self.layers[self.config.n_input_layers : self.out_start]:
                hidden_states = layer(
                    positions, hidden_states, loop_idx=loop_idx + kv_offset
                )
            hidden_states = justnorm(hidden_states)
        return hidden_states

    def _shared_prefix(self, input_ids, positions, inputs_embeds=None):
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embeddings(input_ids)
        )
        for layer in self.layers[: self.config.n_input_layers]:
            hidden_states = layer(positions, hidden_states)
        return hidden_states, self._loops(
            positions, hidden_states, hidden_states, 0, self.num_shared_loops
        )

    def _draft_suffix(
        self, positions, hidden_states, input_hidden_states, output_end=None
    ):
        hidden_states = self._loops(
            positions,
            hidden_states,
            input_hidden_states,
            self.num_shared_loops,
            self.num_draft_loops,
            kv_offset=self.num_loops - self.num_shared_loops,
        )
        return self._output_block(positions, hidden_states, loop_idx=1, end=output_end)

    def _output_block(self, positions, hidden_states, loop_idx=0, end=None):
        for layer in self.layers[self.out_start : end]:
            hidden_states = layer(positions, hidden_states, loop_idx=loop_idx)
        return hidden_states

    def _target_suffix(self, positions, hidden_states, input_hidden_states):
        hidden_states = self._loops(
            positions,
            hidden_states,
            input_hidden_states,
            self.num_shared_loops,
            self.num_loops,
        )
        return self._output_block(positions, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.num_draft_loops:
            input_hidden_states, hidden_states = self._shared_prefix(
                input_ids, positions, inputs_embeds
            )
            return self._target_suffix(positions, hidden_states, input_hidden_states)

        assert input_ids is not None and inputs_embeds is None
        cache = self.draft_activation_cache
        if cache is not None and cache.verify:
            width = cache.num_speculative_tokens + 1
            tail_positions = positions[width - 1 :: width].contiguous()
            tail_ids = input_ids[width - 1 :: width].contiguous()
            with cache.prefix_context():
                input_hidden_states, hidden_states = self.stages["prefix"](
                    tail_ids, tail_positions
                )
                # Earlier rows already filled the private draft KV while proposing.
                self.stages["fill"](
                    tail_positions,
                    hidden_states,
                    input_hidden_states,
                )
            input_hidden_states, hidden_states = cache.merge(
                input_hidden_states, hidden_states
            )
        else:
            input_hidden_states, hidden_states = self.stages["prefix"](
                input_ids, positions
            )
            self.stages["fill"](positions, hidden_states, input_hidden_states)
        return self.stages["target"](positions, hidden_states, input_hidden_states)

    def draft_forward(self, input_ids, positions, inputs_embeds=None):
        assert input_ids is not None and inputs_embeds is None
        input_hidden_states, hidden_states = self.stages["prefix"](input_ids, positions)
        if self.draft_activation_cache is not None:
            self.draft_activation_cache.record(input_hidden_states, hidden_states)
        return self.stages["draft"](positions, hidden_states, input_hidden_states)


class DoryForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        # hf_overrides can change architecture fields after config initialization.
        config.validate()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Dory currently supports tensor_parallel_size=1 only")
        speculative_config = vllm_config.speculative_config
        num_draft_loops = 0
        num_shared_loops = 0
        if (
            speculative_config is not None
            and speculative_config.method == "looped_self_draft"
        ):
            if vllm_config.parallel_config.prefill_context_parallel_size != 1:
                raise ValueError(
                    "Dory self-speculation requires prefill_context_parallel_size=1"
                )
            num_draft_loops = speculative_config.num_draft_loops
            assert speculative_config.num_shared_loops is not None
            num_shared_loops = speculative_config.num_shared_loops
            if num_draft_loops >= config.n_recurrent_loops:
                raise ValueError("num_draft_loops must be below n_recurrent_loops")
            if config.recurrent_kv_cache_mode != "per_loop":
                raise ValueError("Dory self-speculation requires per_loop KV caches")
        self.backbone = DoryBackbone(
            config,
            vllm_config.cache_config,
            vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "backbone"),
            num_draft_loops=num_draft_loops,
            num_shared_loops=num_shared_loops,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def get_looped_self_draft_model(self) -> nn.Module:
        return DoryDraftModel(self)

    def get_compilation_modules(self) -> list[nn.Module]:
        return list(self.backbone.stages.values()) or [self]

    def get_draft_attn_layer_names(self) -> list[str]:
        backbone = self.backbone
        names: list[str] = []
        for i, layer in enumerate(backbone.layers):
            if layer.is_attention:
                attn = layer.mixer.attn
                if i >= backbone.out_start:
                    attn = attn[1:]
                elif i >= self.config.n_input_layers:
                    attn = [
                        *attn[: backbone.num_shared_loops],
                        *attn[backbone.num_loops :],
                    ]
                names.extend(a.layer_name for a in attn)
        return names

    @property
    def draft_activation_cache(self) -> "LoopedSelfDraftActivationCache | None":
        return self.backbone.draft_activation_cache

    @draft_activation_cache.setter
    def draft_activation_cache(self, cache: "LoopedSelfDraftActivationCache | None"):
        self.backbone.draft_activation_cache = cache

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
                self.lm_head.logit_scale.float() / self.config.initializer_range
            )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.backbone.draft_activation_cache is not None:
            self.backbone.draft_activation_cache.invalidate()
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
                    rows = heads * self.config.head_dim
                else:
                    rows = self.config.intermediate_size
                if isinstance(param, PerTensorScaleParameter):
                    shape = () if w.ndim == 0 else (1,)
                else:
                    # Preserve packed input columns and per-block scale columns.
                    shape = (rows, *param.shape[1:])
                self._check_weight_shape(name, w, shape)
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
            if isinstance(param, PerTensorScaleParameter) and w.ndim == 0:
                shape = ()
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


class DoryDraftModel(nn.Module):
    """Self-speculation drafter: the target's weights at `num_draft_loops` loops.

    The input block and first `num_shared_loops` loops share the target's KV.
    Remaining draft loops and the output block use private draft caches, also
    filled during target prefill. The target is held by reference, so no parameters
    repeat.
    """

    def __init__(self, target):
        super().__init__()
        self._target = (target,)

    @property
    def target(self):
        return self._target[0]

    def forward(self, input_ids, positions, hidden_states=None, inputs_embeds=None):
        return self.target.backbone.draft_forward(
            input_ids, positions, inputs_embeds=inputs_embeds
        )

    def compute_logits(self, hidden_states):
        return self.target.compute_logits(hidden_states)
