# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AnyModel: dynamic-parent wrapper for NAS-optimized heterogeneous models.

At init time, ``__new__`` creates a wrapper subclass inheriting from both
AnyModel and the target base model (e.g. ``LlamaForCausalLM``).  The base
model builds the full structure with the global config; then
``_patch_anymodel_layers`` replaces layers whose ``block_configs`` differ
and injects no-ops.  VL models work automatically (vision tower, forward,
load_weights all inherited).

No-op layers are replaced with identity pass-throughs (``NoOpAttention``,
``NoOpMLP``) paired with ``NoOpNorm``.  The identity approach defers the
residual add to the next real norm, avoiding in-place aliasing issues from
vLLM's fused ``fused_add_rms_norm`` kernel.
"""

from __future__ import annotations

import copy
import functools
import importlib
import importlib.util
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import ClassVar

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .interfaces import HasNoOps
from .utils import PPMissingLayer, maybe_prefix

logger = init_logger(__name__)

# Security: all config-supplied module paths must resolve within this package.
_VLLM_MODELS_PKG = "vllm.model_executor.models"
# Security: only simple identifier segments allowed in layers_path.
_SAFE_ATTR_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_layers_path(layers_path: str) -> None:
    """Reject dunder or non-identifier segments in a config-supplied layers_path."""
    for part in layers_path.split("."):
        if not _SAFE_ATTR_RE.match(part):
            raise ValueError(
                f"Security: invalid attribute segment in layers_path: {part!r}"
            )
        if part.startswith("__"):
            raise ValueError(
                f"Security: dunder attributes not allowed in layers_path: {part!r}"
            )


def _validate_config_arch_info(arch_info: ArchInfo) -> None:
    """Ensure config-supplied module paths resolve inside vllm.model_executor.models."""
    for field_name in ("decoder_layer_module", "base_model_module"):
        val = getattr(arch_info, field_name, None)
        if val is None:
            continue
        try:
            abs_name = importlib.util.resolve_name(val, _VLLM_MODELS_PKG)
        except (ImportError, ValueError) as exc:
            raise ValueError(
                f"Security: ArchInfo.{field_name} is not a valid relative "
                f"module path: {val!r}"
            ) from exc
        if not abs_name.startswith(_VLLM_MODELS_PKG + "."):
            raise ValueError(
                f"Security: ArchInfo.{field_name} must resolve within "
                f"'{_VLLM_MODELS_PKG}'. Got resolved name: {abs_name!r}"
            )
    _validate_layers_path(arch_info.layers_path)


class _AttrDict(dict):
    """Dict with attribute access; stays JSON-serializable for config hashing."""

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
    if isinstance(block_config, dict):
        return block_config.get(section, {})
    return getattr(block_config, section, {})


def _get_attr(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_block_attr(block_config, section: str, key: str, default=None):
    return _get_attr(_get_block_section(block_config, section), key, default)


class NoOpAttention(nn.Module):
    """Identity pass-through replacing a skipped attention block."""

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        return args[0]


class NoOpMLP(nn.Module):
    """Identity pass-through replacing a skipped feed-forward block."""

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class NoOpNorm(nn.Module):
    """Identity norm for layers adjacent to no-op attention/MLP.

    Two-arg path: returns both inputs unchanged (defers residual accumulation).
    Single-arg path (first layer, residual is None): returns zeros to break
    tensor aliasing that would cause fused_add_rms_norm to double the residual.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return hidden_states, residual
        return torch.zeros_like(hidden_states)


@dataclass
class ArchInfo:
    """Per-architecture descriptor for building and patching layers."""

    # Decoder layer class
    decoder_layer_module: str  # dotted module path, e.g. ".llama"
    decoder_layer_class: str  # fallback class name

    # Hybrid / multi-type layer support
    decoder_layer_class_map: dict[str, str] | None = None
    hybrid_pattern_field: str | None = None

    # Sub-module attribute names on the decoder layer
    attn_module: str = "self_attn"
    attn_norm_module: str = "input_layernorm"
    ffn_module: str = "mlp"
    ffn_norm_module: str = "post_attention_layernorm"

    # Config field names (block_config key -> HF config attr)
    kv_heads_field: str = "num_key_value_heads"
    intermediate_size_field: str = "intermediate_size"
    hidden_act_field: str = "hidden_act"

    # MoE (None = not MoE)
    moe_num_experts_field: str | None = None
    moe_intermediate_size_field: str | None = (
        None  # falls back to intermediate_size_field
    )

    extra_config_fields: dict[str, str] = field(default_factory=dict)
    """Maps ``"section.key"`` block_config paths to config attr names."""

    # Dynamic-parent / multimodal support
    base_model_module: str | None = None  # None = use decoder_layer_module
    layers_path: str = "model.layers"
    init_prefix: str | None = None  # None = inherit from engine
    layer_hf_config: str | None = None  # None = use hf_config directly


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
    # Hybrid: NemotronH
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
            f"module {info.decoder_layer_module!r} for layer {layer_idx}: {exc}"
        ) from exc


def _create_layer_config(global_config, block_config, info: ArchInfo):
    """Deep-copy global_config with per-layer overrides applied."""
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
            s = _get_attr(moe, "expert_intermediate_dim")
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
    """Replace sub-modules with no-ops per block_config."""
    attn_noop = _get_block_attr(block_config, "attention", "no_op", False)
    ffn_noop = _get_block_attr(block_config, "ffn", "no_op", False)

    shared_module = info.attn_module == info.ffn_module
    shared_norm = info.attn_norm_module == info.ffn_norm_module

    if shared_module:
        if attn_noop and ffn_noop:
            setattr(layer, info.attn_module, NoOpAttention())
        if shared_norm and attn_noop and ffn_noop:
            setattr(layer, info.attn_norm_module, NoOpNorm())
    else:
        if attn_noop:
            setattr(layer, info.attn_module, NoOpAttention())
            setattr(layer, info.attn_norm_module, NoOpNorm())
        if ffn_noop:
            setattr(layer, info.ffn_module, NoOpMLP())
            if not shared_norm or attn_noop:
                setattr(layer, info.ffn_norm_module, NoOpNorm())


def _collect_noop_prefixes(block_configs: list, info: ArchInfo) -> frozenset[str]:
    """Build weight-name prefixes (ending with '.') for no-op sub-modules."""
    prefixes: set[str] = set()
    shared_module = info.attn_module == info.ffn_module
    shared_norm = info.attn_norm_module == info.ffn_norm_module

    for idx, bc in enumerate(block_configs):
        lp = f"{info.layers_path}.{idx}"
        attn_noop = _get_block_attr(bc, "attention", "no_op", False)
        ffn_noop = _get_block_attr(bc, "ffn", "no_op", False)

        if shared_module:
            if attn_noop and ffn_noop:
                prefixes.add(f"{lp}.{info.attn_module}.")
                prefixes.add(f"{lp}.{info.attn_norm_module}.")
        else:
            if attn_noop:
                prefixes.add(f"{lp}.{info.attn_module}.")
                prefixes.add(f"{lp}.{info.attn_norm_module}.")
            if ffn_noop:
                prefixes.add(f"{lp}.{info.ffn_module}.")
                if not shared_norm or attn_noop:
                    prefixes.add(f"{lp}.{info.ffn_norm_module}.")
    return frozenset(prefixes)


@functools.cache
def _layer_init_params(layer_cls: type) -> frozenset[str]:
    return frozenset(inspect.signature(layer_cls.__init__).parameters)


def _instantiate_layer(
    layer_cls: type[nn.Module],
    vllm_config: VllmConfig,
    prefix: str,
    per_layer_config,
    layer_idx: int,
) -> nn.Module:
    """Instantiate a decoder layer, patching vllm_config with per-layer config."""
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
    return layer_cls(**{k: v for k, v in _pool.items() if k in params})


def _has_overrides(block_config, info: ArchInfo | None = None) -> bool:
    """True if block_config has config overrides requiring a layer rebuild.
    No-ops alone don't trigger a rebuild."""
    attn = _get_block_section(block_config, "attention")
    ffn = _get_block_section(block_config, "ffn")
    if (
        _get_attr(attn, "num_key_value_heads") is not None
        or _get_attr(ffn, "intermediate_size") is not None
        or _get_attr(ffn, "hidden_act") is not None
        or _get_attr(ffn, "moe") is not None
    ):
        return True
    if info and info.extra_config_fields:
        for block_path in info.extra_config_fields:
            parts = block_path.split(".", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"extra_config_fields key {block_path!r} must be "
                    f"'section.key' format (e.g. 'ffn.hidden_size')"
                )
            if _get_block_attr(block_config, parts[0], parts[1]) is not None:
                return True
    return False


def _overrides_differ(block_config, global_config, info: ArchInfo) -> bool:
    """True if block_config values actually differ from global_config defaults.

    Companion to ``_has_overrides``: while that function checks whether
    override *keys* are present (not None), this function checks whether
    the values are different from what the model was originally built with.
    Skipping rebuilds for identical values avoids unnecessary GPU allocation
    churn during ``_patch_anymodel_layers``.
    """
    attn = _get_block_section(block_config, "attention")
    ffn = _get_block_section(block_config, "ffn")

    kv = _get_attr(attn, "num_key_value_heads")
    if kv is not None and kv != getattr(global_config, info.kv_heads_field, None):
        return True

    intermediate = _get_attr(ffn, "intermediate_size")
    if intermediate is not None and intermediate != getattr(
        global_config, info.intermediate_size_field, None
    ):
        return True

    hidden_act = _get_attr(ffn, "hidden_act")
    if hidden_act is not None and hidden_act != getattr(
        global_config, info.hidden_act_field, None
    ):
        return True

    moe = _get_attr(ffn, "moe")
    if moe is not None:
        if info.moe_num_experts_field is None:
            return True
        n = _get_attr(moe, "num_local_experts")
        if n is not None and n != getattr(
            global_config, info.moe_num_experts_field, None
        ):
            return True
        moe_size_field = (
            info.moe_intermediate_size_field or info.intermediate_size_field
        )
        s = _get_attr(moe, "expert_intermediate_dim")
        if s is not None and s != getattr(global_config, moe_size_field, None):
            return True

    if info and info.extra_config_fields:
        for block_path, config_attr in info.extra_config_fields.items():
            parts = block_path.split(".", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"extra_config_fields key {block_path!r} must be "
                    f"'section.key' format (e.g. 'ffn.hidden_size')"
                )
            val = _get_block_attr(block_config, parts[0], parts[1])
            if val is not None and val != getattr(global_config, config_attr, None):
                return True

    return False


def _unregister_layer(layer_prefix: str, vllm_config: VllmConfig) -> None:
    """Remove a layer's entries from static_forward_context and
    static_all_moe_layers to avoid stale references after replacement."""
    cc = vllm_config.compilation_config
    prefix_dot = layer_prefix + "."

    stale = [k for k in cc.static_forward_context if k.startswith(prefix_dot)]
    for k in stale:
        del cc.static_forward_context[k]

    cc.static_all_moe_layers = [
        k for k in cc.static_all_moe_layers if not k.startswith(prefix_dot)
    ]


def _patch_anymodel_layers(
    model: nn.Module,
    vllm_config: VllmConfig,
    arch_info: ArchInfo,
    base_init_prefix: str,
) -> None:
    """Post-init: rebuild layers with overrides and apply no-ops."""
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

        if _has_overrides(block_config, arch_info) and _overrides_differ(
            block_config, layer_base_config, arch_info
        ):
            per_layer_config = _create_layer_config(
                layer_base_config, block_config, arch_info
            )
            layer_cls = _resolve_layer_class(arch_info, layer_base_config, layer_idx)
            layer_prefix = f"{layers_prefix}.{layer_idx}"
            _unregister_layer(layer_prefix, vllm_config)
            target_device = next(layer.parameters()).device
            with torch.device("cpu"):
                new_layer = _instantiate_layer(
                    layer_cls,
                    vllm_config,
                    layer_prefix,
                    per_layer_config,
                    layer_idx,
                )
            layers[layer_idx] = new_layer
            del layer
            new_layer.to(target_device)
            layer = new_layer

        _apply_no_ops(layer, block_config, arch_info)

        layer_prefix = maybe_prefix(layers_prefix, str(layer_idx))
        attn_noop = _get_block_attr(block_config, "attention", "no_op", False)
        ffn_noop = _get_block_attr(block_config, "ffn", "no_op", False)
        shared_module = arch_info.attn_module == arch_info.ffn_module

        if shared_module:
            if attn_noop and ffn_noop:
                _unregister_layer(
                    f"{layer_prefix}.{arch_info.attn_module}", vllm_config
                )
        else:
            if attn_noop:
                _unregister_layer(
                    f"{layer_prefix}.{arch_info.attn_module}", vllm_config
                )
            if ffn_noop:
                _unregister_layer(f"{layer_prefix}.{arch_info.ffn_module}", vllm_config)


def _arch_info_from_config(hf_config) -> ArchInfo | None:
    """Load ArchInfo from hf_config.anymodel_arch_info if present."""
    data = getattr(hf_config, "anymodel_arch_info", None)
    if not data:
        return None
    if not isinstance(data, dict):
        data = vars(data)
    known = {f.name for f in dataclass_fields(ArchInfo)}
    arch_info = ArchInfo(**{k: v for k, v in data.items() if k in known})
    _validate_config_arch_info(arch_info)
    return arch_info


def _make_wrapper_cls(arch_name: str, arch_info: ArchInfo) -> type:
    """Create ``AnyModel{arch_name}(AnyModel, BaseModelCls)`` wrapper."""
    base_mod_path = arch_info.base_model_module or arch_info.decoder_layer_module
    mod = importlib.import_module(base_mod_path, package=__package__)
    base_cls = getattr(mod, arch_name)
    return type(
        f"AnyModel{arch_name}",
        (AnyModel, base_cls),
        {"_anymodel_arch_info": arch_info, "has_noops": True},
    )


def _expand_noop_prefixes_for_mapper(
    prefixes: frozenset[str], model_cls: type
) -> frozenset[str]:
    """Expand noop prefixes with HF-style equivalents via the weight mapper.

    ``AnyModel.load_weights`` filters weights *before* the base model's
    ``hf_to_vllm_mapper`` is applied, so noop prefixes (which use vLLM
    names) won't match HF-style checkpoint names.  This reverse-maps the
    prefixes so both naming conventions are covered.
    """
    mapper = getattr(model_cls, "hf_to_vllm_mapper", None)
    if mapper is None:
        return prefixes

    expanded: set[str] = set(prefixes)
    for orig, new in getattr(mapper, "orig_to_new_prefix", {}).items():
        if new is None:
            continue
        for p in prefixes:
            if p.startswith(new):
                expanded.add(orig + p[len(new) :])
    for orig, new in getattr(mapper, "orig_to_new_substr", {}).items():
        if new is None:
            continue
        for p in list(expanded):
            if new in p:
                expanded.add(p.replace(new, orig, 1))
    return frozenset(expanded)


class AnyModel(nn.Module, HasNoOps):
    """Factory entry point for NAS-optimized heterogeneous models.

    ``__new__`` creates a wrapper subclass ``(AnyModel, BaseModelCls)``.
    ``__init__`` delegates to the base model, then patches layers per
    ``block_configs``.  All base model methods are inherited automatically."""

    has_noops = True
    _anymodel_arch_info: ClassVar[ArchInfo | None] = None
    _wrapper_cache: ClassVar[dict[str, type[AnyModel]]] = {}

    @staticmethod
    def _resolve_arch(config) -> tuple[str, ArchInfo, bool]:
        """Return (arch_name, arch_info, from_config) for the given hf_config."""
        arch_name = getattr(config, "base_architecture", None)
        if not arch_name:
            raise ValueError(
                "AnyModel config must set 'base_architecture' to the name "
                "of the underlying model class (e.g. 'LlamaForCausalLM')."
            )

        arch_info = _arch_info_from_config(config)
        from_config = arch_info is not None
        if arch_info is None:
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
        cache_key = f"config:{repr(arch_info)}" if from_config else arch_name
        if cache_key not in AnyModel._wrapper_cache:
            AnyModel._wrapper_cache[cache_key] = _make_wrapper_cls(arch_name, arch_info)
        return AnyModel._wrapper_cache[cache_key]

    @classmethod
    def resolve_wrapper_cls(cls, model_config) -> type[AnyModel]:
        """Return the concrete wrapper class for model_config."""
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Filter out no-op module weights, then delegate to base class."""
        arch_info = type(self)._anymodel_arch_info
        if arch_info is not None:
            config = self.config
            layer_base_config = (
                getattr(config, arch_info.layer_hf_config)
                if arch_info.layer_hf_config
                else config
            )
            block_configs = getattr(layer_base_config, "block_configs", None)
            if block_configs:
                noop_prefixes = _collect_noop_prefixes(block_configs, arch_info)
                if noop_prefixes:
                    noop_prefixes = _expand_noop_prefixes_for_mapper(
                        noop_prefixes, type(self)
                    )
                    weights = (
                        (name, tensor)
                        for name, tensor in weights
                        if not any(name.startswith(p) for p in noop_prefixes)
                    )
        return super().load_weights(weights)
