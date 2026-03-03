# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic AnyModel for NAS-optimized heterogeneous architectures.

AnyModel reuses existing decoder layer classes (LlamaDecoderLayer,
Qwen2DecoderLayer, etc.) directly, feeding them a per-layer config
derived from block_configs.

Each supported architecture is described by a lightweight :class:`ArchInfo`
dataclass that lives in :data:`_ARCH_REGISTRY`, keyed by the HuggingFace
``architectures`` class name (e.g. ``"LlamaForCausalLM"``).  Adding support
for a new architecture requires only a new :class:`ArchInfo` entry — no
subclassing needed.

**Canonical block_configs schema (Stage 1)**::

    {
      "attention": {
        "no_op": false,              // skip attention entirely
        "num_key_value_heads": 4     // GQA override (optional)
      },
      "ffn": {
        "no_op": false,              // skip FFN entirely
        "intermediate_size": 8192,   // override (optional)
        "hidden_act": "silu",        // override (optional)
        "moe": {                     // present only for MoE layers (optional)
          "num_local_experts": 8,
          "expert_intermediate_size": 1024
        }
      }
    }

**Constructor styles** (``ArchInfo.ctor_style``):

* ``"standard"`` — ``cls(config, cache_config, quant_config, prefix)``
* ``"vllm_config"`` — ``cls(vllm_config, prefix, config)``
* ``"nemotron_h"`` — ``cls(config, layer_idx, model_config,
  cache_config, quant_config, parallel_config, prefix)``
* ``"gpt_oss"`` — ``cls(vllm_config*, quant_config, prefix)``
  where ``vllm_config*`` is a shallow copy with ``hf_config`` replaced
  by the per-layer config.

**Hybrid architectures** (NemotronH):

When ``ArchInfo.decoder_layer_class_map`` is set, the layer class is
selected per position using the character at
``config.<hybrid_pattern_field>[layer_idx]``.

The ``ArchInfo`` fields are intentionally kept as plain strings so they
can later be overridden directly from the model's ``config.json``.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .interfaces import HasNoOps, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

# ---------------------------------------------------------------------------
# Block config access helpers
# ---------------------------------------------------------------------------


def _get_block_section(block_config, section: str):
    """Get a section (e.g. 'attention', 'ffn') from a block_config entry.

    Handles both dict and namespace-object representations.
    """
    if isinstance(block_config, dict):
        return block_config.get(section, {})
    return getattr(block_config, section, {})


def _get_attr(obj, key: str, default=None):
    """Get an attribute from either a dict or namespace object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_block_attr(block_config, section: str, key: str, default=None):
    """Shortcut: get a nested attribute from block_config[section][key]."""
    section_data = _get_block_section(block_config, section)
    return _get_attr(section_data, key, default)


# ---------------------------------------------------------------------------
# No-op modules
# ---------------------------------------------------------------------------


class NoOpAttention(nn.Module):
    """No-op replacement for attention. Returns zeros so residual is
    preserved when added back (zeros + residual = residual)."""

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class NoOpMLP(nn.Module):
    """No-op replacement for MLP / MoE block. Returns zeros so residual
    is preserved when added back."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class Same(nn.Module):
    """Identity replacement for layer norms. Must handle vLLM's fused
    RMSNorm calling convention: (hidden_states) or
    (hidden_states, residual) -> (hidden_states, residual)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return hidden_states, residual
        return hidden_states


# ---------------------------------------------------------------------------
# ArchInfo — per-architecture descriptor (data only, no subclassing needed)
# ---------------------------------------------------------------------------


@dataclass
class ArchInfo:
    """Describes how to build and patch heterogeneous layers for one
    base architecture.

    All string fields are intentionally simple so they can later be
    overridden via model ``config.json`` without code changes.
    """

    # Lazy import path for the decoder layer class --------------------------
    decoder_layer_module: str
    """Dotted module path, absolute or relative to this package.
    Examples: ``".llama"``, ``"vllm.model_executor.models.qwen2"``."""

    decoder_layer_class: str
    """Default class name within ``decoder_layer_module``.
    Used when ``decoder_layer_class_map`` is absent or has no match."""

    # Constructor calling convention ----------------------------------------
    ctor_style: str = "standard"
    """How to call the decoder layer constructor.  One of:

    * ``"standard"`` — ``cls(config, cache_config, quant_config, prefix)``
    * ``"vllm_config"`` — ``cls(vllm_config, prefix, config)``
    * ``"nemotron_h"`` — ``cls(config, layer_idx, model_config,
      cache_config, quant_config, parallel_config, prefix)``
    * ``"gpt_oss"`` — ``cls(vllm_config*, quant_config, prefix)`` where
      ``vllm_config*`` is a shallow copy with ``hf_config`` substituted
      by the per-layer config.
    """

    # Hybrid / multi-type layer support -------------------------------------
    decoder_layer_class_map: dict[str, str] | None = None
    """Maps layer-type code → class name for hybrid architectures.
    E.g. ``{"*": "NemotronHAttentionDecoderLayer",
             "-": "NemotronHMLPDecoderLayer",
             "E": "NemotronHMoEDecoderLayer",
             "M": "NemotronHMambaDecoderLayer"}``.
    If ``None``, ``decoder_layer_class`` is always used."""

    hybrid_pattern_field: str | None = None
    """Config attribute whose value is a per-layer type string, e.g.
    ``"hybrid_override_pattern"`` → ``"*-*E*M"``.  Position ``layer_idx``
    gives the type code used to select from ``decoder_layer_class_map``."""

    # Module attribute names on the decoder layer instance ------------------
    attn_module: str = "self_attn"
    attn_norm_module: str = "input_layernorm"
    ffn_module: str = "mlp"
    ffn_norm_module: str = "post_attention_layernorm"

    # Config attribute names (canonical block_config name → arch name) ------
    kv_heads_field: str = "num_key_value_heads"
    """Config attribute that holds the number of KV heads."""

    intermediate_size_field: str = "intermediate_size"
    """Config attribute that holds the FFN intermediate size."""

    hidden_act_field: str = "hidden_act"
    """Config attribute that holds the activation function name."""

    # MoE-specific config attribute names (None = not a MoE architecture) --
    moe_num_experts_field: str | None = None
    """Config attribute for total number of experts, e.g. ``"num_experts"``
    or ``"num_local_experts"``.  ``None`` means the arch has no MoE."""

    moe_intermediate_size_field: str | None = None
    """Config attribute for per-expert intermediate size.
    ``None`` means fall back to ``intermediate_size_field``."""

    # Future extensibility --------------------------------------------------
    extra_config_fields: dict[str, str] = field(default_factory=dict)
    """Reserved for Stage 2 (hidden size, latent dim, window size, …).
    Maps canonical block_config field paths to config attribute names."""


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_ARCH_REGISTRY: dict[str, ArchInfo] = {
    # ---- Dense: Llama family ------------------------------------------------
    "LlamaForCausalLM": ArchInfo(
        decoder_layer_module=".llama",
        decoder_layer_class="LlamaDecoderLayer",
        ctor_style="vllm_config",
    ),
    # MistralForCausalLM is dense (no MoE).  Uses the same vllm_config ctor
    # as Llama via MistralDecoderLayer(LlamaDecoderLayer).
    "MistralForCausalLM": ArchInfo(
        decoder_layer_module=".mistral",
        decoder_layer_class="MistralDecoderLayer",
        ctor_style="vllm_config",
    ),
    # ---- Dense: Qwen2/3 family ----------------------------------------------
    "Qwen2ForCausalLM": ArchInfo(
        decoder_layer_module=".qwen2",
        decoder_layer_class="Qwen2DecoderLayer",
    ),
    "Qwen3ForCausalLM": ArchInfo(
        decoder_layer_module=".qwen3",
        decoder_layer_class="Qwen3DecoderLayer",
    ),
    # ---- MoE: Qwen family ---------------------------------------------------
    "Qwen2MoeForCausalLM": ArchInfo(
        decoder_layer_module=".qwen2_moe",
        decoder_layer_class="Qwen2MoeDecoderLayer",
        moe_num_experts_field="num_experts",
        moe_intermediate_size_field="moe_intermediate_size",
    ),
    # ---- MoE: Mixtral family ------------------------------------------------
    "MixtralForCausalLM": ArchInfo(
        decoder_layer_module=".mixtral",
        decoder_layer_class="MixtralDecoderLayer",
        ffn_module="block_sparse_moe",
        moe_num_experts_field="num_local_experts",
    ),
    # ---- Hybrid: NemotronH --------------------------------------------------
    # nemotron-nano-12b-v2  (hybrid_override_pattern "*-")
    # nemotron-3-nano-30b   (hybrid_override_pattern "*E")
    #
    # Layer type is selected per-position from config.hybrid_override_pattern:
    #   "*" → NemotronHAttentionDecoderLayer
    #   "-" → NemotronHMLPDecoderLayer
    #   "E" → NemotronHMoEDecoderLayer
    #   "M" → NemotronHMambaDecoderLayer
    #
    # No-op patching uses "mixer" for the compute module and "norm" for the
    # layer norm, which is the naming convention for all NemotronH layer types.
    "NemotronHForCausalLM": ArchInfo(
        decoder_layer_module=".nemotron_h",
        decoder_layer_class="NemotronHAttentionDecoderLayer",  # fallback
        ctor_style="nemotron_h",
        decoder_layer_class_map={
            "*": "NemotronHAttentionDecoderLayer",
            "-": "NemotronHMLPDecoderLayer",
            "E": "NemotronHMoEDecoderLayer",
            "M": "NemotronHMambaDecoderLayer",
        },
        hybrid_pattern_field="hybrid_override_pattern",
        attn_module="mixer",
        attn_norm_module="norm",
        ffn_module="mixer",
        ffn_norm_module="norm",
        moe_num_experts_field="n_routed_experts",
        moe_intermediate_size_field="moe_intermediate_size",
    ),
    # NemotronHPuzzleForCausalLM is an alias used by some Puzzletron
    # checkpoints; it resolves to the same vLLM class.
    "NemotronHPuzzleForCausalLM": ArchInfo(
        decoder_layer_module=".nemotron_h",
        decoder_layer_class="NemotronHAttentionDecoderLayer",
        ctor_style="nemotron_h",
        decoder_layer_class_map={
            "*": "NemotronHAttentionDecoderLayer",
            "-": "NemotronHMLPDecoderLayer",
            "E": "NemotronHMoEDecoderLayer",
            "M": "NemotronHMambaDecoderLayer",
        },
        hybrid_pattern_field="hybrid_override_pattern",
        attn_module="mixer",
        attn_norm_module="norm",
        ffn_module="mixer",
        ffn_norm_module="norm",
        moe_num_experts_field="n_routed_experts",
        moe_intermediate_size_field="moe_intermediate_size",
    ),
    # ---- MoE: GptOss --------------------------------------------------------
    # gpt-oss-20b: TransformerBlock(vllm_config, quant_config, prefix).
    # The per-layer config is injected by substituting hf_config in a
    # shallow copy of vllm_config (ctor_style="gpt_oss").
    "GptOssForCausalLM": ArchInfo(
        decoder_layer_module=".gpt_oss",
        decoder_layer_class="TransformerBlock",
        ctor_style="gpt_oss",
        attn_module="attn",
        moe_num_experts_field="num_local_experts",
    ),
    # ---- NOT YET SUPPORTED --------------------------------------------------
    # Qwen3VLForConditionalGeneration is a multimodal (vision-language) model.
    # AnyModelForCausalLM only handles text-only LMs.  Supporting Qwen3VL
    # requires a separate AnyModelForConditionalGeneration wrapper that also
    # manages the vision encoder.  Tracked as future work.
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _resolve_layer_class(
    info: ArchInfo,
    global_config,
    layer_idx: int,
) -> type[nn.Module]:
    """Return the decoder layer class for this position.

    For hybrid architectures (``decoder_layer_class_map`` is set) the class
    is chosen by reading ``global_config.<hybrid_pattern_field>[layer_idx]``.
    Falls back to ``info.decoder_layer_class`` when the map is absent or the
    pattern character is not in the map.
    """
    class_name = info.decoder_layer_class
    if info.decoder_layer_class_map and info.hybrid_pattern_field:
        pattern = getattr(global_config, info.hybrid_pattern_field, "") or ""
        if layer_idx < len(pattern):
            class_name = info.decoder_layer_class_map.get(
                pattern[layer_idx], class_name
            )
    mod = importlib.import_module(info.decoder_layer_module, package=__package__)
    return getattr(mod, class_name)


def _create_layer_config(global_config, block_config, info: ArchInfo):
    """Return a shallow copy of *global_config* with per-layer overrides
    from *block_config* applied, using the field names in *info*."""
    config = copy.copy(global_config)

    attn = _get_block_section(block_config, "attention")
    ffn = _get_block_section(block_config, "ffn")

    if not _get_attr(attn, "no_op", False):
        kv = _get_attr(attn, "num_key_value_heads")
        if kv is not None:
            setattr(config, info.kv_heads_field, kv)

    if not _get_attr(ffn, "no_op", False):
        intermediate = _get_attr(ffn, "intermediate_size")
        if intermediate is not None:
            setattr(config, info.intermediate_size_field, intermediate)

        hidden_act = _get_attr(ffn, "hidden_act")
        if hidden_act is not None:
            setattr(config, info.hidden_act_field, hidden_act)

        moe = _get_attr(ffn, "moe")
        if moe is not None:
            if info.moe_num_experts_field is not None:
                n = _get_attr(moe, "num_local_experts")
                if n is not None:
                    setattr(config, info.moe_num_experts_field, n)
            moe_size_field = (
                info.moe_intermediate_size_field or info.intermediate_size_field
            )
            s = _get_attr(moe, "expert_intermediate_size")
            if s is not None:
                setattr(config, moe_size_field, s)

    return config


def _apply_no_ops(layer: nn.Module, block_config, info: ArchInfo) -> None:
    """Replace sub-modules with no-ops according to *block_config*."""
    if _get_block_attr(block_config, "attention", "no_op", False):
        setattr(layer, info.attn_module, NoOpAttention())
        setattr(layer, info.attn_norm_module, Same())
    if _get_block_attr(block_config, "ffn", "no_op", False):
        setattr(layer, info.ffn_module, NoOpMLP())
        setattr(layer, info.ffn_norm_module, Same())


def _instantiate_layer(
    info: ArchInfo,
    layer_cls: type[nn.Module],
    vllm_config: VllmConfig,
    prefix: str,
    per_layer_config,
    layer_idx: int,
) -> nn.Module:
    """Instantiate a decoder layer using the calling convention in *info*."""
    if info.ctor_style == "vllm_config":
        return layer_cls(
            vllm_config=vllm_config, prefix=prefix, config=per_layer_config
        )
    if info.ctor_style == "nemotron_h":
        return layer_cls(
            config=per_layer_config,
            layer_idx=layer_idx,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            parallel_config=vllm_config.parallel_config,
            prefix=prefix,
        )
    if info.ctor_style == "gpt_oss":
        # TransformerBlock reads config from vllm_config.model_config.hf_config,
        # so inject the per-layer config via a shallow copy.
        mock_mc = copy.copy(vllm_config.model_config)
        mock_mc.hf_config = per_layer_config
        mock_vc = copy.copy(vllm_config)
        mock_vc.model_config = mock_mc
        return layer_cls(
            vllm_config=mock_vc,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )
    # "standard" (default)
    return layer_cls(
        config=per_layer_config,
        cache_config=vllm_config.cache_config,
        quant_config=vllm_config.quant_config,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class AnyModel(nn.Module):
    """Generic transformer container that creates heterogeneous decoder
    layers from ``block_configs`` using the appropriate :class:`ArchInfo`."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        arch_info: ArchInfo,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self._create_layer(prefix, vllm_config, arch_info),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    @staticmethod
    def _create_layer(
        prefix: str,
        vllm_config: VllmConfig,
        arch_info: ArchInfo,
    ) -> nn.Module:
        layer_idx = extract_layer_index(prefix)
        config = vllm_config.model_config.hf_config
        block_config = config.block_configs[layer_idx]
        per_layer_config = _create_layer_config(config, block_config, arch_info)
        layer_cls = _resolve_layer_class(arch_info, config, layer_idx)
        layer = _instantiate_layer(
            arch_info,
            layer_cls,
            vllm_config,
            prefix,
            per_layer_config,
            layer_idx,
        )
        _apply_no_ops(layer, block_config, arch_info)
        return layer

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
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

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class AnyModelForCausalLM(nn.Module, SupportsPP, HasNoOps):
    """Top-level causal LM wrapper for NAS-optimized heterogeneous models.

    Auto-detected when ``block_configs`` is present in the HF config and
    ``architectures[0]`` maps to a known entry in :data:`_ARCH_REGISTRY`.

    To add support for a new architecture, add a single :class:`ArchInfo`
    entry to :data:`_ARCH_REGISTRY` — no subclassing required.
    """

    has_noops = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        architectures = getattr(config, "architectures", None) or []
        arch_name = architectures[0] if architectures else None
        arch_info = _ARCH_REGISTRY.get(arch_name) if arch_name else None
        if arch_info is None:
            raise ValueError(
                f"No AnyModel support for architecture {arch_name!r}. "
                f"Supported: {sorted(_ARCH_REGISTRY)}"
            )

        self.model = AnyModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            arch_info=arch_info,
        )

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
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = None
        if self.config.tie_word_embeddings:
            skip_prefixes = ["lm_head."]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)
