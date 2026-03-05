# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic AnyModel for NAS-optimized heterogeneous architectures.

AnyModel uses a **dynamic parent** approach: :class:`AnyModel`
changes its own ``__class__`` at runtime to a dynamically created subclass
of the target base model (e.g. ``LlamaForCausalLM``).  The base model's
``__init__`` then builds the full model structure with the *global* config.
Afterwards, :func:`_patch_anymodel_layers` replaces layers whose per-layer
config (from ``block_configs``) differs from the global config — changing
the weight shapes — and injects no-op modules where required.

This design means:

* No separate ``AnyModelForConditionalGeneration`` is needed; VL models
  (e.g. Qwen3VL) are supported automatically because their vision tower,
  ``forward``, and ``load_weights`` are inherited from the base class.
* Adding a new architecture requires only a single :class:`ArchInfo` entry
  in :data:`_ARCH_REGISTRY` — no subclassing.

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

**Hybrid architectures** (NemotronH):

When ``ArchInfo.decoder_layer_class_map`` is set, the layer class is
selected per position using the character at
``config.<hybrid_pattern_field>[layer_idx]``.

The ``ArchInfo`` fields are intentionally kept as plain strings so they
can later be overridden directly from the model's ``config.json``.
"""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .interfaces import HasNoOps
from .utils import PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# AttrDict – JSON-serializable dict with attribute access
# ---------------------------------------------------------------------------


class _AttrDict(dict):
    """A JSON-serializable dict that also supports attribute access.

    Used for ``block_configs`` entries so they can be accessed as
    ``bc.attention.no_op`` (attribute style) while remaining serializable
    by :func:`json.dumps` (needed for config hashing via
    ``to_json_string()``).
    """

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


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
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class NoOpMLP(nn.Module):
    """No-op replacement for MLP / MoE block. Returns zeros so residual
    is preserved when added back."""

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class NoOpNorm(nn.Module):
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

    # Dynamic-parent support ------------------------------------------------
    base_model_module: str | None = None
    """Dotted module path for the *base model* class.
    ``None`` means fall back to ``decoder_layer_module``.  Set this when
    the base model class lives in a different file from the decoder layer
    (e.g. Qwen3VL: layers in ``\".qwen3\"``, model in ``\".qwen3_vl\"``)."""

    layers_path: str = "model.layers"
    """Dotted attribute path from the model root to the decoder layers
    ``nn.ModuleList``.  Examples: ``"model.layers"`` (most architectures),
    ``"language_model.model.layers"`` (Qwen3VL)."""

    init_prefix: str | None = None
    """Prefix to pass to ``base_cls.__init__`` instead of the outer
    ``prefix``.

    * ``None`` (default) — inherit the engine-provided prefix.
    * ``""`` — explicitly pass an empty string (forces blank prefix).
    * ``"model"`` — used by Qwen3VL whose checkpoint weights are stored
      under a ``model`` sub-key."""

    layer_hf_config: str | None = None
    """Attribute path on ``hf_config`` to use as the *base* for per-layer
    config shallow-copies.  ``None`` means use ``hf_config`` directly.
    Set to ``"text_config"`` for Qwen3VL where the language model layers
    are configured by the text sub-config, not the top-level VL config."""


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_ARCH_REGISTRY: dict[str, ArchInfo] = {
    # ---- Dense: Llama family ------------------------------------------------
    "LlamaForCausalLM": ArchInfo(
        decoder_layer_module=".llama",
        decoder_layer_class="LlamaDecoderLayer",
    ),
    "MistralForCausalLM": ArchInfo(
        decoder_layer_module=".mistral",
        decoder_layer_class="MistralDecoderLayer",
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
    "GptOssForCausalLM": ArchInfo(
        decoder_layer_module=".gpt_oss",
        decoder_layer_class="TransformerBlock",
        attn_module="attn",
        moe_num_experts_field="num_local_experts",
    ),
    # ---- Multimodal: Qwen3VL ------------------------------------------------
    # qwen3-vl-30b: language-model layers are Qwen3DecoderLayer instances
    # (via Qwen3LLMModel < Qwen3Model inside Qwen3LLMForCausalLM).
    # Weights are stored under "model.*" in the checkpoint, so init_prefix
    # must be "model".  Per-layer configs are based on config.text_config
    # (the Qwen3 language model sub-config, not the top-level VL config).
    "Qwen3VLForConditionalGeneration": ArchInfo(
        decoder_layer_module=".qwen3",
        decoder_layer_class="Qwen3DecoderLayer",
        base_model_module=".qwen3_vl",
        layers_path="language_model.model.layers",
        init_prefix="model",
        layer_hf_config="text_config",
    ),
}

# NemotronHPuzzleForCausalLM is an alias used by some Puzzletron checkpoints;
# it is identical to NemotronHForCausalLM so we share the same object.
_ARCH_REGISTRY["NemotronHPuzzleForCausalLM"] = _ARCH_REGISTRY["NemotronHForCausalLM"]


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
    try:
        mod = importlib.import_module(info.decoder_layer_module, package=__package__)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        raise type(exc)(
            f"Failed to resolve layer class {class_name!r} from "
            f"module {info.decoder_layer_module!r} for layer {layer_idx}: "
            f"{exc}"
        ) from exc


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

    # Stage-2 / extra fields (e.g. {"ffn.hidden_size": "hidden_size"})
    for block_path, config_attr in info.extra_config_fields.items():
        parts = block_path.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"extra_config_fields key {block_path!r} must be "
                f"'section.key' format (e.g. 'ffn.hidden_size')"
            )
        val = _get_block_attr(block_config, parts[0], parts[1])
        if val is not None:
            setattr(config, config_attr, val)

    return config


def _apply_no_ops(layer: nn.Module, block_config, info: ArchInfo) -> None:
    """Replace sub-modules with no-ops according to *block_config*."""
    if _get_block_attr(block_config, "attention", "no_op", False):
        setattr(layer, info.attn_module, NoOpAttention())
        setattr(layer, info.attn_norm_module, NoOpNorm())
    if _get_block_attr(block_config, "ffn", "no_op", False):
        setattr(layer, info.ffn_module, NoOpMLP())
        setattr(layer, info.ffn_norm_module, NoOpNorm())


@functools.cache
def _layer_init_params(layer_cls: type) -> frozenset[str]:
    """Return the parameter names of *layer_cls.__init__*, cached per class."""
    return frozenset(inspect.signature(layer_cls.__init__).parameters)


def _instantiate_layer(
    layer_cls: type[nn.Module],
    vllm_config: VllmConfig,
    prefix: str,
    per_layer_config,
    layer_idx: int,
) -> nn.Module:
    """Instantiate a decoder layer via signature introspection.

    A patched ``vllm_config`` (with ``model_config.hf_config`` swapped to
    ``per_layer_config``) is included in the pool so that classes which read
    config through ``vllm_config.model_config.hf_config`` (e.g.
    ``TransformerBlock``) receive the per-layer config automatically.
    Classes with an explicit ``config`` kwarg use that instead.
    """
    mock_mc = copy.copy(vllm_config.model_config)
    mock_mc.hf_config = per_layer_config
    mock_vc = copy.copy(vllm_config)
    mock_vc.model_config = mock_mc

    _pool = {
        "config": per_layer_config,
        "vllm_config": mock_vc,
        "cache_config": vllm_config.cache_config,
        "quant_config": vllm_config.quant_config,
        "parallel_config": vllm_config.parallel_config,
        "model_config": mock_mc,
        "prefix": prefix,
        "layer_idx": layer_idx,
    }
    params = _layer_init_params(layer_cls)
    kwargs = {k: v for k, v in _pool.items() if k in params}
    return layer_cls(**kwargs)


# ---------------------------------------------------------------------------
# Post-init patching helpers
# ---------------------------------------------------------------------------


def _has_overrides(block_config, info: ArchInfo | None = None) -> bool:
    """Return True if block_config contains overrides that require the layer
    to be rebuilt (KV heads, FFN size, activation function, MoE fields, or
    any extra_config_fields declared in *info*).

    No-ops alone do not require layer recreation — they are handled by
    :func:`_apply_no_ops` which simply replaces sub-modules in place.

    .. note::

       This checks for *presence* (non-None) of override fields, not whether
       they differ from the global config.  Layers whose block_config carries
       the same value as the global config will still be rebuilt.  This is
       intentional for simplicity; comparing against global is a potential
       future optimisation.
    """
    attn = _get_block_section(block_config, "attention")
    ffn = _get_block_section(block_config, "ffn")
    has_base = (
        _get_attr(attn, "num_key_value_heads") is not None
        or _get_attr(ffn, "intermediate_size") is not None
        or _get_attr(ffn, "hidden_act") is not None
        or _get_attr(ffn, "moe") is not None
    )
    if has_base:
        return True
    if info and info.extra_config_fields:
        for block_path in info.extra_config_fields:
            parts = block_path.split(".", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"extra_config_fields key {block_path!r} must be "
                    f"'section.key' format (e.g. 'ffn.hidden_size')"
                )
            val = _get_block_attr(block_config, parts[0], parts[1])
            if val is not None:
                return True
    return False


def _unregister_layer(layer_prefix: str, vllm_config: VllmConfig) -> None:
    """Remove entries registered by the old layer from static_forward_context.

    Attention modules register themselves in ``compilation_config
    .static_forward_context`` during ``__init__``.  When a layer is about
    to be replaced, those entries must be removed first so the new layer
    can register under the same keys without triggering a
    "Duplicate layer name" error.
    """
    ctx = vllm_config.compilation_config.static_forward_context
    stale = [k for k in ctx if k.startswith(layer_prefix + ".")]
    for k in stale:
        del ctx[k]


def _patch_anymodel_layers(
    model: nn.Module,
    vllm_config: VllmConfig,
    arch_info: ArchInfo,
    base_init_prefix: str,
) -> None:
    """Post-init: replace layers that have per-layer config overrides and
    apply no-op module substitutions.

    Must be called *after* ``base_cls.__init__`` so that all layers are
    already built.  vLLM's subsequent profiling and memory estimation will
    then observe the correct (per-layer) weight shapes.

    Args:
        model: The model instance (already initialised by base_cls.__init__).
        vllm_config: VllmConfig used during initialisation.
        arch_info: ArchInfo for the current architecture.
        base_init_prefix: The prefix that was passed to base_cls.__init__,
            used to reconstruct per-layer weight-name prefixes.
    """
    config = vllm_config.model_config.hf_config

    # Use a sub-config as the base for per-layer config copies when specified
    # (e.g. Qwen3VL passes text_config to its LM layers, not the VL config).
    layer_base_config = (
        getattr(config, arch_info.layer_hf_config)
        if arch_info.layer_hf_config
        else config
    )

    # block_configs lives on the same config as the layer parameters: the
    # sub-config for VL models (e.g. text_config), top-level otherwise.
    block_configs = layer_base_config.block_configs

    # Navigate from the model root to the nn.ModuleList of decoder layers.
    obj = model
    for part in arch_info.layers_path.split("."):
        obj = getattr(obj, part)
    layers: nn.ModuleList = obj

    # Full dotted prefix for the layers list; used to construct the
    # weight-name prefix when instantiating replacement layers.
    layers_prefix = maybe_prefix(base_init_prefix, arch_info.layers_path)

    if len(block_configs) != len(layers):
        logger.warning(
            "block_configs length (%d) != layers length (%d); "
            "extra entries will be ignored.",
            len(block_configs),
            len(layers),
        )

    for layer_idx, block_config in enumerate(block_configs):
        if layer_idx >= len(layers):
            break
        layer = layers[layer_idx]
        # Skip pipeline-parallel placeholder layers on other ranks.
        if isinstance(layer, PPMissingLayer):
            continue

        per_layer_config = _create_layer_config(
            layer_base_config, block_config, arch_info
        )

        if _has_overrides(block_config, arch_info):
            # Weight shapes differ from the global config — rebuild the
            # layer with the per-layer config so load_weights maps correctly
            # to the pruned checkpoint tensors.
            layer_cls = _resolve_layer_class(arch_info, layer_base_config, layer_idx)
            layer_prefix = f"{layers_prefix}.{layer_idx}"

            # The original layer registered its attention modules in the
            # compilation config's static_forward_context.  Remove those
            # entries before creating the replacement to avoid
            # "Duplicate layer name" errors.
            _unregister_layer(layer_prefix, vllm_config)

            new_layer = _instantiate_layer(
                layer_cls,
                vllm_config,
                layer_prefix,
                per_layer_config,
                layer_idx,
            )
            layers[layer_idx] = new_layer
            layer = new_layer

        _apply_no_ops(layer, block_config, arch_info)


# ---------------------------------------------------------------------------
# Config-driven ArchInfo loading
# ---------------------------------------------------------------------------


def _arch_info_from_config(hf_config) -> ArchInfo | None:
    """Load an :class:`ArchInfo` from ``hf_config.anymodel_arch_info``.

    Returns ``None`` when the field is absent or falsy.  Accepts both plain
    dicts (as deserialized from JSON) and namespace/object representations.
    Unknown keys are silently ignored so that configs remain forward-compatible
    as ``ArchInfo`` fields are added or removed.

    Example ``config.json`` snippet::

        {
            "architectures": ["MyCustomForCausalLM"],
            "anymodel_arch_info": {
                "decoder_layer_module": ".llama",
                "decoder_layer_class": "LlamaDecoderLayer",
            },
            "block_configs": [...],
        }
    """
    data = getattr(hf_config, "anymodel_arch_info", None)
    if not data:
        return None
    if not isinstance(data, dict):
        data = vars(data)
    known = {f.name for f in dataclass_fields(ArchInfo)}
    return ArchInfo(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Named wrapper class factory
# ---------------------------------------------------------------------------


def _make_wrapper_cls(
    arch_name: str,
    arch_info: ArchInfo,
) -> type:
    """Create a named wrapper class for *arch_name*.

    The wrapper inherits from both :class:`AnyModel` and the target base
    model class (e.g. ``LlamaForCausalLM``), giving it the name
    ``AnyModel{arch_name}`` (e.g. ``AnyModelLlamaForCausalLM``).

    MRO example for Llama::

        AnyModelLlamaForCausalLM
          ├── AnyModel            ← defines __init__ with post-init patching
          └── LlamaForCausalLM   ← base model; super().__init__() resolves here

    Imports are deferred to this function so the module stays fast to import.
    """
    base_mod_path = arch_info.base_model_module or arch_info.decoder_layer_module
    mod = importlib.import_module(base_mod_path, package=__package__)
    base_cls = getattr(mod, arch_name)
    return type(
        f"AnyModel{arch_name}",
        (AnyModel, base_cls),
        {"_anymodel_arch_info": arch_info, "has_noops": True},
    )


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class AnyModel(nn.Module, HasNoOps):
    """Entry point for NAS-optimized heterogeneous models.

    Handles both text-only (``ForCausalLM``) and vision-language
    (``ForConditionalGeneration``) architectures — no separate class needed.

    When instantiated, :meth:`__new__` acts as a *factory*: it creates (and
    caches) a named **wrapper subclass** that inherits from both this class
    and the target base model.  For example, for a Llama checkpoint the
    wrapper is ``AnyModelLlamaForCausalLM(AnyModel, LlamaForCausalLM)``.

    :meth:`__init__` then calls ``super().__init__()`` which, thanks to the
    wrapper's MRO, resolves to the base model's ``__init__``
    (e.g. ``LlamaForCausalLM.__init__``).  This builds the full model
    structure with the *global* HF config.  Afterwards,
    :func:`_patch_anymodel_layers` replaces any decoder layers whose per-
    layer ``block_configs`` differ in weight shape and applies no-op
    module substitutions.  All patching completes before vLLM starts
    profiling or memory estimation.

    The resulting instance inherits every method from the base model
    (``forward``, ``compute_logits``, ``load_weights``, the vision tower
    for VL models, etc.) — no separate ``AnyModelForConditionalGeneration``
    is needed.

    To add support for a new architecture, add a single :class:`ArchInfo`
    entry to :data:`_ARCH_REGISTRY` — no subclassing required.
    """

    has_noops = True

    # Set on each named wrapper subclass by __new__ / _make_wrapper_cls.
    _anymodel_arch_info: ClassVar[ArchInfo | None] = None

    # Cache of arch_name → named wrapper class (lazily populated).
    _wrapper_cache: ClassVar[dict[str, type[AnyModel]]] = {}

    @staticmethod
    def _resolve_arch(config) -> tuple[str, ArchInfo, bool]:
        """Determine the base architecture name and :class:`ArchInfo`.

        Reads ``base_architecture`` from the HF config to identify the
        underlying model class, then resolves the corresponding
        :class:`ArchInfo` (config-driven descriptor takes priority over
        the hardcoded :data:`_ARCH_REGISTRY`).

        Returns:
            ``(arch_name, arch_info, from_config)`` — *from_config* is
            ``True`` when the descriptor was loaded from the HF config's
            ``anymodel_arch_info`` field rather than the hardcoded registry.

        Raises:
            ValueError: if the architecture is not supported.
        """
        arch_name = getattr(config, "base_architecture", None)
        if not arch_name:
            raise ValueError(
                "AnyModel config must set 'base_architecture' to the name "
                "of the underlying model class (e.g. 'LlamaForCausalLM')."
            )

        from_config = False
        arch_info = _arch_info_from_config(config)
        if arch_info is not None:
            from_config = True
        else:
            arch_info = _ARCH_REGISTRY.get(arch_name)

        if arch_info is None:
            raise ValueError(
                f"No AnyModel support for architecture {arch_name!r}. "
                f"Supported: {sorted(_ARCH_REGISTRY)}"
            )

        return arch_name, arch_info, from_config

    @staticmethod
    def _get_or_create_wrapper(
        arch_name: str,
        arch_info: ArchInfo,
        *,
        from_config: bool = False,
    ) -> type[AnyModel]:
        """Return (or lazily create) the named wrapper class for *arch_name*.

        Args:
            from_config: True when *arch_info* was loaded from the HF config
                (``anymodel_arch_info`` field) rather than the hardcoded
                :data:`_ARCH_REGISTRY`.  Config-driven descriptors use a
                repr-based cache key so that distinct descriptors produce
                distinct wrapper classes even for the same architecture name.
        """
        cache_key: str = f"config:{repr(arch_info)}" if from_config else arch_name
        if cache_key not in AnyModel._wrapper_cache:
            AnyModel._wrapper_cache[cache_key] = _make_wrapper_cls(arch_name, arch_info)
        return AnyModel._wrapper_cache[cache_key]

    @classmethod
    def resolve_wrapper_cls(
        cls,
        model_config,
    ) -> type[AnyModel]:
        """Return the concrete wrapper class for *model_config*.

        The wrapper inherits from both :class:`AnyModel` and the target
        base model (e.g. ``LlamaForCausalLM``), giving callers access to
        all class-level methods and attributes defined on the base model
        (``get_mamba_state_shape_from_config``, ``_processor_factory``,
        etc.).

        This is the public API consumed by the model loader and other
        subsystems that need the wrapper class *before* instantiation.
        """
        config = model_config.hf_config
        arch_name, arch_info, from_config = cls._resolve_arch(config)
        return cls._get_or_create_wrapper(arch_name, arch_info, from_config=from_config)

    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""):
        if cls is not AnyModel:
            return super().__new__(cls)

        config = vllm_config.model_config.hf_config
        arch_name, arch_info, from_config = AnyModel._resolve_arch(config)
        wrapper_cls = AnyModel._get_or_create_wrapper(
            arch_name, arch_info, from_config=from_config
        )
        return object.__new__(wrapper_cls)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        arch_info = type(self)._anymodel_arch_info
        base_init_prefix = (
            arch_info.init_prefix if arch_info.init_prefix is not None else prefix
        )

        # super() resolves to the base model class (e.g. LlamaForCausalLM)
        # because it is the next entry after AnyModel in the wrapper's MRO.
        # This builds the full model: embeddings, all layers with the *global*
        # config, LM head, weight-tying, etc.
        super().__init__(vllm_config=vllm_config, prefix=base_init_prefix)

        # Post-init: replace layers with shape-changing overrides and inject
        # no-op modules.  Completes before vLLM profiling / memory estimation.
        _patch_anymodel_layers(self, vllm_config, arch_info, base_init_prefix)
