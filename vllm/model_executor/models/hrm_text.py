# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HRM-Text: Hierarchical Reasoning Model — Text variant.

Reference Hugging Face implementation:
    src/transformers/models/hrm_text/modeling_hrm_text.py

The model performs a hierarchical recurrent forward over two transformer
stacks (``H`` slow, ``L`` fast) inside nested loops. Each recurrence step
gets its own KV cache slot via a unique vLLM-visible layer index. The
PrefixLM attention pattern (prompt bidirectional, response causal) is
realized by reusing ``EncoderOnlyAttention`` (which sets ``causal=False``
unconditionally on every metadata build) but with ``attn_type=DECODER``
so the KV cache is allocated; see ``HrmTextAttention`` for usage.

The on-disk ``attn.gqkv_proj.weight`` (rows concatenated as
``[gate | q | k | v]``) is loaded by a single
``MergedColumnParallelLinear`` with four equal-sized output partitions;
its weight loader auto-splits the fused tensor along the output dim by
``output_sizes`` (the same path used by Phi-3's fused gate_up_proj).
"""

from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
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

from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class HrmTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"HrmTextMLP only supports hidden_act='silu', got {hidden_act!r}"
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HrmTextAttention(nn.Module):
    """One self-attention block; weights shared across recurrence steps.

    HF transformers writes a single fused ``attn.gqkv_proj.weight`` on
    disk (per ``transformers/conversion_mapping.py`` ``"hrm_text"``
    mapping; rows are concatenated as ``[gate | q | k | v]`` along
    ``dim=0``). We mirror that on the model side with a single
    ``MergedColumnParallelLinear`` whose four equal output partitions
    are sharded along the head axis under TP; its weight loader
    auto-splits the fused tensor (same path used by Phi-3's fused
    gate_up_proj). HF's runtime config currently hardcodes MHA
    (``num_key_value_groups=1``); GQA would require ``QKVParallelLinear``
    semantics for q/k/v shard replication and is left for a follow-up
    if/when HF adds it.

    Holds:
      - parameters: gqkv_proj, o_proj, rotary_emb (shared across cycles).
      - ``attn_per_step``: a ``nn.ModuleDict`` keyed by recurrence step
        (as a string), each value an ``EncoderOnlyAttention`` (with
        ``attn_type=DECODER`` so the KV cache is allocated; the
        ``EncoderOnlyAttention`` wrapper sets ``causal=False`` on every
        metadata build). The L stack steps are
        ``[high_cycle_idx*(L_cycles+1)+low_cycle_idx]`` and the H stack
        steps are ``[high_cycle_idx*(L_cycles+1)+L_cycles]``; the two
        ranges are disjoint so each instance registers a unique vLLM
        ``layer_name``
        (``model.{H,L}_module.layers.{global_idx}.self_attn``) and gets
        its own KV cache slot. The global layer index per recurrence step
        is ``step * num_layers_per_stack + layer_idx_in_stack``, matching
        the HF transformers ``cycle_offset`` formula in
        ``modeling_hrm_text.py``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx_in_stack: int,
        stack_kind: Literal["L", "H"],
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0, (
            f"num_attention_heads={self.total_num_heads} must be divisible "
            f"by tp_size={tp_size}"
        )
        # HF main hardcodes MHA (num_key_value_groups=1). We follow.
        self.total_num_kv_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        bias = getattr(config, "attention_bias", False)

        # gqkv_proj: 4-way fused [gate | q | k | v] matching the on-disk
        # `attn.gqkv_proj.weight` row layout. MergedColumnParallelLinear's
        # weight_loader auto-splits the fused disk tensor along the output
        # dim by `output_sizes` (Phi-3's fused gate_up_proj path). MHA
        # only: GQA (num_kv_heads != num_heads) would need
        # QKVParallelLinear semantics for q/k/v shard replication.
        per_head_size = self.total_num_heads * self.head_dim
        self.gqkv_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[per_head_size] * 4,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gqkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # vllm get_rope accepts ``rope_parameters`` directly, matching
        # the dict-shaped HF config field.
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

        # Create one Attention instance per recurrence step actually used
        # by this stack. L runs at steps {h*(L+1)+l : 0 <= l < L_cycles},
        # H at steps {h*(L+1)+L : 0 <= h < H_cycles}; the sets are
        # disjoint, so one global index per (step, layer_in_stack) gives
        # each Attention its own ``layer_name`` and KV cache slot.
        H_cycles = config.H_cycles
        L_cycles = config.L_cycles
        num_layers_per_stack = config.num_layers_per_stack
        if stack_kind == "L":
            steps_used = [
                high_cycle_idx * (L_cycles + 1) + low_cycle_idx
                for high_cycle_idx in range(H_cycles)
                for low_cycle_idx in range(L_cycles)
            ]
        else:  # "H"
            steps_used = [
                high_cycle_idx * (L_cycles + 1) + L_cycles
                for high_cycle_idx in range(H_cycles)
            ]

        # `EncoderOnlyAttention` already wraps the attention backend so
        # `causal=False` is set on every metadata build (PrefixLM
        # bidirectional prefill); passing `attn_type=DECODER` keeps the
        # KV cache allocation needed by the recurrent forward.
        self.attn_per_step = nn.ModuleDict()
        for step in steps_used:
            global_idx = step * num_layers_per_stack + layer_idx_in_stack
            unique_prefix = prefix.replace(
                f"layers.{layer_idx_in_stack}", f"layers.{global_idx}"
            )
            self.attn_per_step[str(step)] = EncoderOnlyAttention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                attn_type=AttentionType.DECODER,
                prefix=f"{unique_prefix}.attn",
                raise_on_invalid_attn_type=False,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        gqkv, _ = self.gqkv_proj(hidden_states)
        g, q, k, v = gqkv.split(
            [self.q_size, self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn_per_step[str(current_step)](q, k, v)
        # Sigmoid gate. Shapes: attn_out is (..., q_size); g is (..., q_size).
        attn_out = torch.sigmoid(g) * attn_out
        out, _ = self.o_proj(attn_out)
        return out


class HrmTextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx_in_stack: int,
        stack_kind: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Attribute name `self_attn` matches HF's model class. The on-disk
        # `attn.{gqkv_proj,o_proj}.weight` keys are renamed to
        # `self_attn.{gqkv_proj,o_proj}.weight` by the `WeightsMapper` in
        # `HrmTextForCausalLM` so vLLM's standard `AutoWeightsLoader`
        # handles the rest.
        self.self_attn = HrmTextAttention(
            config=config,
            layer_idx_in_stack=layer_idx_in_stack,
            stack_kind=stack_kind,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = HrmTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            bias=getattr(config, "mlp_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Parameterless RMSNorm (HF main: HrmTextRMSNorm has no weight).
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            current_step=current_step,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HrmTextStack(nn.Module):
    """A single transformer stack — used twice (H and L)."""

    def __init__(
        self,
        config: PretrainedConfig,
        stack_kind: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HrmTextDecoderLayer(
                    config=config,
                    layer_idx_in_stack=i,
                    stack_kind=stack_kind,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_layers_per_stack)
            ]
        )
        self.final_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        current_step_base: int,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                current_step=current_step_base,
            )
        return self.final_norm(hidden_states)


@support_torch_compile
class HrmTextModel(nn.Module):
    """Hierarchical recurrent transformer body.

    Forward (matches HF main exactly,
    src/transformers/models/hrm_text/modeling_hrm_text.py:495-547):

        hidden_states_high_cycle = embed(input_ids) * embedding_scale
        hidden_states_low_cycle = z_L_init.expand_as(hidden_states_high_cycle)
        for high_cycle_idx in range(H_cycles):
            for low_cycle_idx in range(L_cycles):
                step = high_cycle_idx * (L_cycles + 1) + low_cycle_idx
                hidden_states_low_cycle = L_module(
                    hidden_states_low_cycle + hidden_states_high_cycle,
                    current_step=step,
                )
            step = high_cycle_idx * (L_cycles + 1) + L_cycles
            hidden_states_high_cycle = H_module(
                hidden_states_high_cycle + hidden_states_low_cycle,
                current_step=step,
            )
        return hidden_states_high_cycle
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.L_module = HrmTextStack(
            config=config,
            stack_kind="L",
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.L_module",
        )
        self.H_module = HrmTextStack(
            config=config,
            stack_kind="H",
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.H_module",
        )
        # Frozen learned initial L state. HF inits to zeros and sets
        # requires_grad_(False); for inference we just load the tensor.
        self.z_L_init = nn.Parameter(
            torch.zeros(config.hidden_size), requires_grad=False
        )

        # Embedding scale: HF uses config.embedding_scale (default
        # 1 / initializer_range = 50.0 when initializer_range=0.02). NOT
        # sqrt(hidden_size) like Gemma.
        self.embedding_scale = getattr(config, "embedding_scale", None)
        if self.embedding_scale is None:
            init_range = getattr(config, "initializer_range", 0.02)
            self.embedding_scale = 1.0 / init_range

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embedding_scale

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states_high_cycle = inputs_embeds
        hidden_states_low_cycle = self.z_L_init.to(
            dtype=hidden_states_high_cycle.dtype,
            device=hidden_states_high_cycle.device,
        ).expand_as(hidden_states_high_cycle)

        H_cycles = self.config.H_cycles
        L_cycles = self.config.L_cycles
        for high_cycle_idx in range(H_cycles):
            for low_cycle_idx in range(L_cycles):
                step = high_cycle_idx * (L_cycles + 1) + low_cycle_idx
                hidden_states_low_cycle = self.L_module(
                    positions=positions,
                    hidden_states=hidden_states_low_cycle + hidden_states_high_cycle,
                    current_step_base=step,
                )
            step = high_cycle_idx * (L_cycles + 1) + L_cycles
            hidden_states_high_cycle = self.H_module(
                positions=positions,
                hidden_states=hidden_states_high_cycle + hidden_states_low_cycle,
                current_step_base=step,
            )

        return hidden_states_high_cycle


class HrmTextForCausalLM(nn.Module):
    """Hierarchical Reasoning Model — Text variant, causal LM.

    Reference: src/transformers/models/hrm_text/modeling_hrm_text.py
    """

    # On-disk weight key remap: HF stores attention weights as
    # `attn.{gqkv_proj,o_proj}.weight`; our model uses `self_attn.*`
    # (matching HF's runtime model class). Both `gqkv_proj` (4-way fused
    # gate/q/k/v) and `mlp.gate_up_proj` (2-way fused gate/up) are loaded
    # directly via MergedColumnParallelLinear's fused-on-disk path; no
    # packed_modules_mapping entries are needed.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".attn.": ".self_attn."},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "HrmTextForCausalLM does not support pipeline parallelism."
            )

        self.model = HrmTextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["lm_head."] if self.config.tie_word_embeddings else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
