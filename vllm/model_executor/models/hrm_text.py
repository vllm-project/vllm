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
realized by wrapping the attention backend's metadata builder so that
``causal=False`` is set on every build; see ``_create_hrm_attention_backend``
for why this is correct for both prefill and decode rows.

The on-disk ``attn.gqkv_proj.weight`` (rows concatenated as
``[gate | q | k | v]``) is split at load time into a separate
``gate_proj`` (``ColumnParallelLinear``) and ``qkv_proj``
(``QKVParallelLinear``); see ``HrmTextForCausalLM.load_weights``. This
keeps the model TP-friendly without diverging from HF's fused on-disk
schema.
"""

import functools
from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
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
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1.attention.selector import get_attn_backend

from .utils import AutoWeightsLoader, maybe_prefix


@functools.lru_cache
def _create_hrm_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    """Wrap an attention backend so its builder unconditionally sets
    ``causal=False`` on every build.

    Mirrors ``EncoderOnlyAttention``'s subclass-the-backend pattern but
    keeps the KV cache (HRM-Text needs it for the recurrent forward).

    Why unconditional rather than gated on ``is_prefilling.all()``:

      vLLM v1 schedules continuous batching. Even with
      ``enable_chunked_prefill=False`` a single attention build can see a
      *mixed* batch — some requests already in decode, others entering
      prefill on this step. Gating on ``is_prefilling.all()`` keeps
      ``causal=True`` in that mixed case, which silently runs the
      newly-prefilling requests as pure causal. That diverges from
      HRM-Text's PrefixLM training distribution and tanks accuracy.

      Unconditional ``causal=False`` is correct because:
        - Prefill rows have ``query_len = N``, where ``causal=False``
          makes the prompt bidirectional (matches HF main +
          ``token_type_ids=1``).
        - Decode rows have ``query_len = 1``, where ``causal=True`` and
          ``causal=False`` are identical (a single query has no future
          tokens to mask).

      FlashAttention's varlen kernels apply ``causal`` per sub-sequence
      via ``cu_seqlens_q``, so a single global flag is sufficient — no
      cross-prompt contamination.
    """
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class HrmAttentionBuilder(underlying_builder):  # type: ignore[misc,valid-type]
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_md = copy(common_attn_metadata)
            new_md.causal = False
            return super().build(common_prefix_len, new_md, fast_build)

    return subclass_attention_backend(
        name_prefix="HrmAttention_",
        attention_backend_cls=underlying_attn_backend,
        builder_cls=HrmAttentionBuilder,
    )


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
    ``dim=0``). To stay TP-friendly we keep the gate as a separate
    ``ColumnParallelLinear`` and use a standard ``QKVParallelLinear``
    for q/k/v; the on-disk fused tensor is split into the two halves at
    load time by ``HrmTextForCausalLM.load_weights``.

    Holds:
      - parameters: gate_proj, qkv_proj, o_proj, rotary_emb (shared
        across cycles).
      - ``attn_per_step``: a ``nn.ModuleDict`` keyed by recurrence step
        (as a string), each value a vLLM ``Attention``. The L stack
        steps are ``[h*(L_cycles+1)+l]`` and the H stack steps are
        ``[h*(L_cycles+1)+L_cycles]``; the two ranges are disjoint so
        each ``Attention`` registers a unique vLLM ``layer_name``
        (``model.{H,L}_module.layers.{global_idx}.attn``) and gets its
        own KV cache slot. The global layer index per recurrence step is
        ``step * num_layers_per_stack + layer_idx_in_stack``, matching
        the HF transformers ``cycle_offset`` formula in
        ``modeling_hrm_text.py``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx_in_stack: int,
        stack_kind: str,  # "L" or "H"
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

        # gate_proj: separate from QKV so it shards cleanly under TP.
        # The on-disk gqkv_proj.weight is split into gate_proj.weight and
        # qkv_proj.weight at load time (see HrmTextForCausalLM.load_weights).
        self.gate_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": getattr(config, "rope_theta", 10000.0),
            }
        # vllm get_rope accepts ``rope_parameters`` directly, matching
        # the dict-shaped HF config field.
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        # Build one wrapped backend (lru_cached, so all layers/steps share it).
        dtype = torch.get_default_dtype()
        kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        underlying = get_attn_backend(
            self.head_dim,
            dtype,
            kv_cache_dtype,
            attn_type=AttentionType.DECODER,
        )
        wrapped_backend = _create_hrm_attention_backend(underlying)

        # Create one Attention instance per recurrence step actually used
        # by this stack. L runs at steps {h*(L+1)+l : 0 <= l < L_cycles},
        # H at steps {h*(L+1)+L : 0 <= h < H_cycles}; the sets are
        # disjoint, so one global index per (step, layer_in_stack) gives
        # each Attention its own ``layer_name`` and KV cache slot.
        if stack_kind not in ("L", "H"):
            raise ValueError(f"stack_kind must be 'L' or 'H', got {stack_kind!r}")
        H_cycles = config.H_cycles
        L_cycles = config.L_cycles
        num_layers_per_stack = config.num_layers_per_stack
        if stack_kind == "L":
            steps_used = [
                h * (L_cycles + 1) + l_step
                for h in range(H_cycles)
                for l_step in range(L_cycles)
            ]
        else:  # "H"
            steps_used = [h * (L_cycles + 1) + L_cycles for h in range(H_cycles)]

        self.attn_per_step = nn.ModuleDict()
        for step in steps_used:
            global_idx = step * num_layers_per_stack + layer_idx_in_stack
            unique_prefix = prefix.replace(
                f"layers.{layer_idx_in_stack}", f"layers.{global_idx}"
            )
            self.attn_per_step[str(step)] = Attention(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                attn_type=AttentionType.DECODER,
                attn_backend=wrapped_backend,
                prefix=f"{unique_prefix}.attn",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        g, _ = self.gate_proj(hidden_states)
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
        # Named `attn` (not `self_attn`) to match HF's on-disk weight schema:
        # transformers/conversion_mapping.py renames `attn.o_proj` -> `self_attn.o_proj`
        # and splits `attn.gqkv_proj` into 4 separate self_attn projections at
        # load-into-HF-model time. We keep the on-disk name to make vLLM's
        # AutoWeightsLoader work without a custom WeightsMapper.
        self.attn = HrmTextAttention(
            config=config,
            layer_idx_in_stack=layer_idx_in_stack,
            stack_kind=stack_kind,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
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
        hidden_states = self.attn(
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

        z_H = embed(input_ids) * embedding_scale
        z_L = z_L_init.expand_as(z_H)
        for h in range(H_cycles):
            for l_step in range(L_cycles):
                z_L = L_module(z_L + z_H,
                               current_step=h * (L_cycles + 1) + l_step)
            z_H = H_module(z_H + z_L,
                           current_step=h * (L_cycles + 1) + L_cycles)
        return z_H
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

        z_H = inputs_embeds
        z_L = self.z_L_init.to(dtype=z_H.dtype, device=z_H.device).expand_as(z_H)

        H_cycles = self.config.H_cycles
        L_cycles = self.config.L_cycles
        for h in range(H_cycles):
            for l_step in range(L_cycles):
                step = h * (L_cycles + 1) + l_step
                z_L = self.L_module(
                    positions=positions,
                    hidden_states=z_L + z_H,
                    current_step_base=step,
                )
            step = h * (L_cycles + 1) + L_cycles
            z_H = self.H_module(
                positions=positions,
                hidden_states=z_H + z_L,
                current_step_base=step,
            )

        return z_H


class HrmTextForCausalLM(nn.Module):
    """Hierarchical Reasoning Model — Text variant, causal LM.

    Reference: src/transformers/models/hrm_text/modeling_hrm_text.py
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

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

    def _split_disk_gqkv(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Split each on-disk ``attn.gqkv_proj.weight`` into the
        ``attn.gate_proj.weight`` (first ``num_heads * head_dim`` rows)
        and ``attn.qkv_proj.weight`` (remaining rows packed
        ``[q | k | v]``) that our model expects. Other weights pass
        through unchanged.

        The on-disk packing comes from ``transformers/conversion_mapping.py``
        ``"hrm_text"`` mapping, which concatenates rows as
        ``[gate | q | k | v]``. We keep ``[q | k | v]`` as a single
        ``QKVParallelLinear``-shaped tensor so vLLM's standard QKV
        weight loader handles TP partitioning.
        """
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        gate_rows = self.config.num_attention_heads * head_dim
        for name, weight in weights:
            if name.endswith(".attn.gqkv_proj.weight"):
                base = name[: -len(".gqkv_proj.weight")]
                yield f"{base}.gate_proj.weight", weight[:gate_rows]
                yield f"{base}.qkv_proj.weight", weight[gate_rows:]
            else:
                yield name, weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF main HRM-Text weights.

        HF transformers writes a fused ``attn.gqkv_proj.weight`` per
        layer (rows concatenated as ``[gate | q | k | v]`` along
        ``dim=0``). ``_split_disk_gqkv`` splits that into the
        ``gate_proj.weight`` and ``qkv_proj.weight`` tensors our
        ``HrmTextAttention`` expects before handing them to
        ``AutoWeightsLoader``; the ``packed_modules_mapping`` then takes
        care of the standard ``q/k/v`` and ``mlp.gate/up`` fusion.
        """
        skip_prefixes = ["lm_head."] if self.config.tie_word_embeddings else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(self._split_disk_gqkv(weights))
