# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AnyModel: dynamic-parent wrapper for NAS-optimized heterogeneous models.

At init time, ``__new__`` creates a wrapper subclass inheriting from both
AnyModel and the target base model (e.g. ``LlamaForCausalLM``).  The base
model builds the full structure with the global config; then
``_patch_anymodel_layers`` replaces layers whose ``block_configs`` differ
and injects no-ops.  VL models work automatically (vision tower, forward,
load_weights all inherited).

No-op handling
~~~~~~~~~~~~~~
Layers marked ``"no_op": true`` are replaced with identity pass-throughs
(``NoOpAttention``, ``NoOpMLP``) paired with ``NoOpNorm``.  The identity
approach is necessary because vLLM fuses the residual add into the layer
norm via an **in-place** CUDA kernel (``fused_add_rms_norm``).  The
identity norms defer the residual add to the next real norm, which
correctly accumulates the residual stream without in-place aliasing issues.

Canonical block_configs schema::

    {
        "attention": {"no_op": false, "num_key_value_heads": 4},
        "ffn": {
            "no_op": false,
            "intermediate_size": 8192,
            "hidden_act": "silu",
            "moe": {"num_local_experts": 8, "expert_intermediate_size": 1024},
        },
    }
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


class _AttrDict(dict):
    """Dict with attribute access, stays JSON-serializable for
    ``to_json_string()`` config hashing."""

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


def _get_block_section(block_config, section: str):
    """Get a section (e.g. 'attention', 'ffn'); handles dict or namespace."""
    if isinstance(block_config, dict):
        return block_config.get(section, {})
    return getattr(block_config, section, {})


def _get_attr(obj, key: str, default=None):
    """Attribute lookup that works on both dicts and namespace objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_block_attr(block_config, section: str, key: str, default=None):
    """Shortcut for block_config[section][key]."""
    section_data = _get_block_section(block_config, section)
    return _get_attr(section_data, key, default)


class NoOpAttention(nn.Module):
    """Identity pass-through replacing a skipped attention block.

    Uses ``*args, **kwargs`` to handle varying call conventions across
    architectures (keyword-only, positional, reversed order, etc.).
    """

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        return args[0]


class NoOpMLP(nn.Module):
    """Identity pass-through replacing a skipped feed-forward block."""

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class NoOpNorm(nn.Module):
    """Identity replacement for layer norms adjacent to no-op attention/MLP.

    Returns (hidden_states, residual) unchanged so the decoder block acts as
    a pure identity. The real residual accumulation is left to the surrounding
    real norms in non-no-op blocks.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return hidden_states, residual
        return hidden_states


@dataclass
class ArchInfo:
    """Per-architecture descriptor for building and patching layers.
    All fields are plain strings so they can be overridden from config.json."""

    # Decoder layer class location
    decoder_layer_module: str
    """Dotted module path, e.g. ``".llama"``."""

    decoder_layer_class: str
    """Default class name (fallback when ``decoder_layer_class_map`` is absent)."""

    # Hybrid / multi-type layer support
    decoder_layer_class_map: dict[str, str] | None = None
    """Maps layer-type code -> class name (e.g. ``{"*": "AttentionLayer"}``).
    ``None`` means ``decoder_layer_class`` is always used."""

    hybrid_pattern_field: str | None = None
    """Config attr holding a per-layer type string (e.g. ``"*-*E*M"``).
    ``pattern[layer_idx]`` selects from ``decoder_layer_class_map``."""

    # Sub-module attribute names on the decoder layer
    attn_module: str = "self_attn"
    attn_norm_module: str = "input_layernorm"
    ffn_module: str = "mlp"
    ffn_norm_module: str = "post_attention_layernorm"

    # Config field names (canonical block_config key -> HF config attr)
    kv_heads_field: str = "num_key_value_heads"
    intermediate_size_field: str = "intermediate_size"
    hidden_act_field: str = "hidden_act"

    # MoE (None = not a MoE architecture)
    moe_num_experts_field: str | None = None
    moe_intermediate_size_field: str | None = None
    """Falls back to ``intermediate_size_field`` when None."""

    extra_config_fields: dict[str, str] = field(default_factory=dict)
    """Maps ``"section.key"`` block_config paths to config attr names."""

    # Dynamic-parent support
    base_model_module: str | None = None
    """Module path for the base model class. ``None`` = use
    ``decoder_layer_module``. Set when the base model lives in a
    different file (e.g. Qwen3VL)."""

    layers_path: str = "model.layers"
    """Dotted path from model root to the decoder ``nn.ModuleList``."""

    init_prefix: str | None = None
    """Prefix for ``base_cls.__init__``. ``None`` = inherit from engine.
    Set to ``"model"`` for Qwen3VL (checkpoint weights under ``model.*``)."""

    layer_hf_config: str | None = None
    """Attr path on ``hf_config`` to use as per-layer config base.
    ``None`` = use ``hf_config`` directly. E.g. ``"text_config"`` for
    Qwen3VL where LM layers use the text sub-config."""


_ARCH_REGISTRY: dict[str, ArchInfo] = {
    # Dense: Llama family
    "LlamaForCausalLM": ArchInfo(
        decoder_layer_module=".llama",
        decoder_layer_class="LlamaDecoderLayer",
    ),
    "MistralForCausalLM": ArchInfo(
        decoder_layer_module=".mistral",
        decoder_layer_class="MistralDecoderLayer",
    ),
    # Dense: Qwen2/3 family
    "Qwen2ForCausalLM": ArchInfo(
        decoder_layer_module=".qwen2",
        decoder_layer_class="Qwen2DecoderLayer",
    ),
    "Qwen3ForCausalLM": ArchInfo(
        decoder_layer_module=".qwen3",
        decoder_layer_class="Qwen3DecoderLayer",
    ),
    # MoE: Qwen family
    "Qwen2MoeForCausalLM": ArchInfo(
        decoder_layer_module=".qwen2_moe",
        decoder_layer_class="Qwen2MoeDecoderLayer",
        moe_num_experts_field="num_experts",
        moe_intermediate_size_field="moe_intermediate_size",
    ),
    # MoE: Mixtral family
    "MixtralForCausalLM": ArchInfo(
        decoder_layer_module=".mixtral",
        decoder_layer_class="MixtralDecoderLayer",
        ffn_module="block_sparse_moe",
        moe_num_experts_field="num_local_experts",
    ),
    # Hybrid: NemotronH (layer type from config.hybrid_override_pattern)
    "NemotronHForCausalLM": ArchInfo(
        decoder_layer_module=".nemotron_h",
        decoder_layer_class="NemotronHAttentionDecoderLayer",
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
    # MoE: GptOss
    "GptOssForCausalLM": ArchInfo(
        decoder_layer_module=".gpt_oss",
        decoder_layer_class="TransformerBlock",
        attn_module="attn",
        moe_num_experts_field="num_local_experts",
    ),
    # Multimodal: Qwen3VL
    "Qwen3VLForConditionalGeneration": ArchInfo(
        decoder_layer_module=".qwen3",
        decoder_layer_class="Qwen3DecoderLayer",
        base_model_module=".qwen3_vl",
        layers_path="language_model.model.layers",
        init_prefix="model",
        layer_hf_config="text_config",
    ),
}

_ARCH_REGISTRY["NemotronHPuzzleForCausalLM"] = _ARCH_REGISTRY["NemotronHForCausalLM"]


def _resolve_layer_class(
    info: ArchInfo,
    global_config,
    layer_idx: int,
) -> type[nn.Module]:
    """Return the decoder layer class for this position (hybrid-aware)."""
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
    """Deep-copy *global_config* with per-layer overrides applied."""
    config = copy.deepcopy(global_config)

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
    """Replace sub-modules with no-ops per *block_config*."""
    if _get_block_attr(block_config, "attention", "no_op", False):
        setattr(layer, info.attn_module, NoOpAttention())
        setattr(layer, info.attn_norm_module, NoOpNorm())
    if _get_block_attr(block_config, "ffn", "no_op", False):
        setattr(layer, info.ffn_module, NoOpMLP())
        setattr(layer, info.ffn_norm_module, NoOpNorm())


@functools.cache
def _layer_init_params(layer_cls: type) -> frozenset[str]:
    """Cached ``__init__`` parameter names for *layer_cls*."""
    return frozenset(inspect.signature(layer_cls.__init__).parameters)


def _instantiate_layer(
    layer_cls: type[nn.Module],
    vllm_config: VllmConfig,
    prefix: str,
    per_layer_config,
    layer_idx: int,
) -> nn.Module:
    """Instantiate a decoder layer via signature introspection.
    Patches ``vllm_config.model_config.hf_config`` with the per-layer config
    so classes that read config either way get the right values."""
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


def _has_overrides(block_config, info: ArchInfo | None = None) -> bool:
    """True if block_config has overrides requiring a layer rebuild.
    Checks presence (non-None), not equality with global config.
    No-ops alone don't trigger a rebuild."""
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
    """Remove old layer's entries from static_forward_context to avoid
    'Duplicate layer name' errors when the replacement registers."""
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
    """Post-init: rebuild layers with overrides and apply no-ops.
    Must run after ``base_cls.__init__`` so layers already exist."""
    config = vllm_config.model_config.hf_config
    layer_base_config = (
        getattr(config, arch_info.layer_hf_config)
        if arch_info.layer_hf_config
        else config
    )
    block_configs = layer_base_config.block_configs

    obj = model
    for part in arch_info.layers_path.split("."):
        obj = getattr(obj, part)
    layers: nn.ModuleList = obj
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
        if isinstance(layer, PPMissingLayer):
            continue

        per_layer_config = _create_layer_config(
            layer_base_config, block_config, arch_info
        )

        if _has_overrides(block_config, arch_info):
            layer_cls = _resolve_layer_class(arch_info, layer_base_config, layer_idx)
            layer_prefix = f"{layers_prefix}.{layer_idx}"
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


def _arch_info_from_config(hf_config) -> ArchInfo | None:
    """Load ArchInfo from ``hf_config.anymodel_arch_info`` if present.
    Unknown keys are ignored for forward-compatibility."""
    data = getattr(hf_config, "anymodel_arch_info", None)
    if not data:
        return None
    if not isinstance(data, dict):
        data = vars(data)
    known = {f.name for f in dataclass_fields(ArchInfo)}
    return ArchInfo(**{k: v for k, v in data.items() if k in known})


def _make_wrapper_cls(
    arch_name: str,
    arch_info: ArchInfo,
) -> type:
    """Create ``AnyModel{arch_name}(AnyModel, BaseModelCls)`` wrapper.
    MRO: AnyModel.__init__ -> base model's __init__ via super()."""
    base_mod_path = arch_info.base_model_module or arch_info.decoder_layer_module
    mod = importlib.import_module(base_mod_path, package=__package__)
    base_cls = getattr(mod, arch_name)
    return type(
        f"AnyModel{arch_name}",
        (AnyModel, base_cls),
        {"_anymodel_arch_info": arch_info, "has_noops": True},
    )


class AnyModel(nn.Module, HasNoOps):
    """Factory entry point for NAS-optimized heterogeneous models.

    ``__new__`` creates a wrapper subclass ``(AnyModel, BaseModelCls)``.
    ``__init__`` delegates to the base model, then patches layers per
    ``block_configs``.  All base model methods (forward, load_weights,
    vision tower, etc.) are inherited automatically."""

    has_noops = True
    _anymodel_arch_info: ClassVar[ArchInfo | None] = None
    _wrapper_cache: ClassVar[dict[str, type[AnyModel]]] = {}

    @staticmethod
    def _resolve_arch(config) -> tuple[str, ArchInfo, bool]:
        """Return ``(arch_name, arch_info, from_config)`` from
        ``config.base_architecture``. Config-driven ArchInfo takes
        priority over the hardcoded registry."""
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
        """Return (or lazily create) the named wrapper class."""
        cache_key: str = f"config:{repr(arch_info)}" if from_config else arch_name
        if cache_key not in AnyModel._wrapper_cache:
            AnyModel._wrapper_cache[cache_key] = _make_wrapper_cls(arch_name, arch_info)
        return AnyModel._wrapper_cache[cache_key]

    @classmethod
    def resolve_wrapper_cls(
        cls,
        model_config,
    ) -> type[AnyModel]:
        """Public API: return the concrete wrapper class for *model_config*.
        Used by model loader and subsystems that need the class pre-init."""
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
        super().__init__(vllm_config=vllm_config, prefix=base_init_prefix)
        _patch_anymodel_layers(self, vllm_config, arch_info, base_init_prefix)
