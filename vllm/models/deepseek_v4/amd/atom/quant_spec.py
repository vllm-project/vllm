# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Typed quantization specification and parser registry.

This module introduces:
- :class:`LayerQuantConfig` — a frozen dataclass for type-safe, immutable
  layer quant descriptions.
- :class:`ParsedQuantConfig` — structured output of parsing ``quantization_config``
  from a HuggingFace ``PretrainedConfig``.
- A parser registry (:func:`register_quant_parser`, :func:`get_quant_parser`) so
  new quantizer back-ends (Quark, compressed-tensors, …) can each provide their
  own parsing logic without bloating ``config.py``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from aiter import QuantType
from aiter.utility.dtypes import d_dtypes

# ──────────────────────────────────────────────────────────────────────
# Typed layer-level spec
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerQuantConfig:
    """Immutable description of how a single layer (or default) is quantized."""

    quant_type: QuantType = QuantType.No
    quant_dtype: Any = torch.bfloat16  # torch.dtype (use Any for forward compat)
    is_dynamic: bool = True
    quant_method: str | None = None

    @property
    def is_quantized(self) -> bool:
        return self.quant_type != QuantType.No

    @classmethod
    def no_quant(cls, dtype: Any = torch.bfloat16) -> LayerQuantConfig:
        """Convenience: unquantized spec with a given storage dtype."""
        return cls(quant_type=QuantType.No, quant_dtype=dtype)


def should_skip_online_quant(cur_type, cur_dtype, online_cfg) -> bool:
    """Skip online re-quant when the layer is excluded (No) or already in target.

    Shared by ``LinearBase.online_quantize_weight``, ``FusedMoE._online_quant``
    and ``RMSNorm.online_quantize_activation``: re-quantizing is a no-op (and may
    corrupt already-quantized weights) when the online target is ``No`` or the
    layer already matches the target ``(quant_type, quant_dtype)``.
    """
    return online_cfg.quant_type == QuantType.No or (
        cur_type == online_cfg.quant_type and cur_dtype == online_cfg.quant_dtype
    )


# ──────────────────────────────────────────────────────────────────────
# Structured parsed config
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ParsedQuantConfig:
    """Result of parsing a ``quantization_config`` dict."""

    global_spec: LayerQuantConfig = field(default_factory=LayerQuantConfig)
    # Pattern specs as list of (pattern, spec) tuples to preserve order
    layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = field(
        default_factory=list
    )
    exclude_layers: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Parser registry
# ──────────────────────────────────────────────────────────────────────

_PARSER_REGISTRY: dict[str, type[QuantConfigParser]] = {}


class QuantConfigParser(ABC):
    """Base class for quantization config parsers."""

    @abstractmethod
    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        """Parse a ``quantization_config`` dict into :class:`ParsedQuantConfig`."""
        ...


def register_quant_parser(name: str):
    """Decorator: register a parser class under *name*."""

    def wrapper(cls: type[QuantConfigParser]):
        _PARSER_REGISTRY[name] = cls
        return cls

    return wrapper


def get_quant_parser(method_name: str) -> QuantConfigParser:
    """Return an instance of the parser for *method_name*.

    Falls back to the ``_generic`` parser if no specific one is registered.
    """
    cls = _PARSER_REGISTRY.get(method_name) or _PARSER_REGISTRY.get("_generic")
    if cls is None:
        raise ValueError(
            f"No quant config parser registered for {method_name!r} "
            f"and no _generic fallback available."
        )
    return cls()


# ──────────────────────────────────────────────────────────────────────
# Built-in parsers
# ──────────────────────────────────────────────────────────────────────


# -- helpers ----------------------------------------------------------------

_QSCHEME_TO_QUANT_TYPE: dict[str, QuantType] = {
    "per_channel": QuantType.per_Token,
    "per_tensor": QuantType.per_Tensor,
    "per_group": QuantType.per_1x32,
    "per_block": QuantType.per_1x128,
}


def _parse_quant_type(qscheme: str | None) -> QuantType:
    if qscheme is None:
        return QuantType.No
    return _QSCHEME_TO_QUANT_TYPE.get(qscheme, QuantType.No)


def _parse_quant_dtype(dtype_str: str | None) -> Any:
    if dtype_str is None:
        return torch.bfloat16
    # Normalise e.g. "fp8_e4m3" -> "fp8", "fp4_e2m1" -> "fp4"
    key = re.sub(r"_e\d+m\d+.*", "", dtype_str)
    # Direct lookup
    result = d_dtypes.get(key)
    if result is not None:
        return result
    # Try common suffixed variants: fp4 -> fp4x2, int4 -> int4x2, etc.
    for suffix in ("x2", "x4"):
        result = d_dtypes.get(key + suffix)
        if result is not None:
            return result
    return torch.bfloat16


def _parse_is_dynamic(input_tensors: dict | None) -> bool:
    if input_tensors is None:
        return True
    return input_tensors.get("is_dynamic", True)


def _build_quark_layer_spec(layer_dict: dict) -> LayerQuantConfig:
    """Build a :class:`LayerQuantConfig` from a single Quark per-layer dict."""
    weight = layer_dict.get("weight", {}) or {}
    return LayerQuantConfig(
        quant_type=_parse_quant_type(weight.get("qscheme")),
        quant_dtype=_parse_quant_dtype(weight.get("dtype")),
        is_dynamic=_parse_is_dynamic(layer_dict.get("input_tensors")),
        quant_method="quark",
    )


# -- Quark ------------------------------------------------------------------


@register_quant_parser("quark")
class QuarkParser(QuantConfigParser):
    """Parser for Quark-style ``quantization_config``."""

    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        global_dict = hf_quant_config.get("global_quant_config") or {}
        layer_dict = hf_quant_config.get("layer_quant_config") or {}
        exclude = list(hf_quant_config.get("exclude") or [])

        global_spec = (
            _build_quark_layer_spec(global_dict) if global_dict else LayerQuantConfig()
        )

        pattern_specs: list[tuple[str, LayerQuantConfig]] = []
        for pattern, cfg in layer_dict.items():
            pattern_specs.append((pattern, _build_quark_layer_spec(cfg)))

        return ParsedQuantConfig(
            global_spec=global_spec,
            layer_pattern_specs=pattern_specs,
            exclude_layers=exclude,
        )


# -- DeepSeek / vLLM block-FP8 ----------------------------------------------


@register_quant_parser("fp8")
class BlockFp8Parser(QuantConfigParser):
    """Parser for DeepSeek/vLLM-style block-FP8 ``quantization_config``.

    The V4-Pro checkpoint declares
    ``{'quant_method': 'fp8', 'fmt': 'e4m3', 'weight_block_size': [128, 128],
       'scale_fmt': 'ue8m0', 'activation_scheme': 'dynamic'}``.
    A ``weight_block_size`` of ``[128, 128]`` maps to per-1x128 block FP8 (e4m3)
    with dynamic activation quant. Layers that carry no on-disk ``.scale``
    (norms, compressor pooling, gates) are handled by ``make_v4_quant_config``'s
    BF16 special-cases layered on top of this global spec.
    """

    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        block = hf_quant_config.get("weight_block_size") or []
        fmt = str(hf_quant_config.get("fmt", "e4m3"))
        # [128, 128] block -> per_1x128; anything else falls back to per_1x128
        # (V4-Pro is always 128x128) but keep the map explicit for clarity.
        qtype = QuantType.per_1x128
        if block and int(block[-1]) == 32:
            qtype = QuantType.per_1x32
        qdtype = _parse_quant_dtype(f"fp8_{fmt}")
        is_dynamic = str(hf_quant_config.get("activation_scheme", "dynamic")) == "dynamic"
        global_spec = LayerQuantConfig(
            quant_type=qtype,
            quant_dtype=qdtype,
            is_dynamic=is_dynamic,
            quant_method="fp8",
        )
        return ParsedQuantConfig(
            global_spec=global_spec,
            layer_pattern_specs=[],
            exclude_layers=list(hf_quant_config.get("exclude") or []),
        )


# -- Online quantization ----------------------------------------------------


@register_quant_parser("online_quant")
class QuarkOnlineParser(QuantConfigParser):
    """Parser for Quark-style online ``quantization_config``."""

    def parse(self, online_quant_config: dict) -> ParsedQuantConfig:
        """Parse the user-facing online quantization dict and populate
        ``online_global_qconfig_dict``, ``online_layer_qconfig_dict``,
        and ``online_exclude_layers_list``.

        Supported format strings:
        - ``"ptpc_fp8"``  — per-tensor-per-channel FP8
        - ``"mxfp4"``     — microscaling FP4 (block size 32)
        """
        if not isinstance(online_quant_config, dict):
            raise TypeError("online_quant_config must be a dict parsed from JSON.")

        SCHEME_MAP = {
            "ptpc": QuantType.per_Token,
        }

        def _parse_online_quant_format(quant_format_str: str) -> LayerQuantConfig:
            quant_format_str = quant_format_str.strip().lower()
            quant_type = None
            dtype_str = None

            if quant_format_str.startswith("mx"):
                quant_type = QuantType.per_1x32
                dtype_str = quant_format_str[2:]
            else:
                parts = quant_format_str.split("_", 1)
                if len(parts) == 2 and parts[0] in SCHEME_MAP:
                    quant_type = SCHEME_MAP[parts[0]]
                    dtype_str = parts[1]
                else:
                    raise ValueError(
                        f"Unsupported online quant format: '{quant_format_str}'. "
                        f"Expected '<scheme>_<dtype>' (e.g. ptpc_fp8) or 'mx<dtype>' (e.g. mxfp4)."
                    )

            dtype_str = dtype_str.split("_")[0]
            if dtype_str.endswith("4"):
                dtype_str += "x2"
            quant_dtype = d_dtypes.get(dtype_str)
            if quant_dtype is None:
                raise ValueError(
                    f"Unsupported online quant dtype: '{dtype_str}' "
                    f"(from '{quant_format_str}')"
                )

            return LayerQuantConfig(
                quant_type=quant_type,
                quant_dtype=quant_dtype,
                is_dynamic=True,
                quant_method="quark",
            )

        global_quant_str = online_quant_config.get("global_quant_config", "")
        if global_quant_str:
            online_global_qconfig_dict = _parse_online_quant_format(global_quant_str)
        else:
            online_global_qconfig_dict = LayerQuantConfig()

        layer_quant_dict = online_quant_config.get("layer_quant_config", {})
        layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
        if isinstance(layer_quant_dict, dict):
            for layer_pattern, quant_str in layer_quant_dict.items():
                layer_pattern_specs.append(
                    (layer_pattern, _parse_online_quant_format(quant_str))
                )

        exclude_layers = online_quant_config.get("exclude_layer", [])
        if isinstance(exclude_layers, str):
            online_exclude_layers_list = [exclude_layers] if exclude_layers else []
        elif isinstance(exclude_layers, list):
            online_exclude_layers_list = exclude_layers
        else:
            online_exclude_layers_list = []
        return ParsedQuantConfig(
            global_spec=online_global_qconfig_dict,
            layer_pattern_specs=layer_pattern_specs,
            exclude_layers=online_exclude_layers_list,
        )


# -- Generic (compressed-tensors, GPTQ, AWQ, …) ----------------------------


@register_quant_parser("_generic")
class GenericParser(QuantConfigParser):
    """Fallback parser that uses heuristics for compressed-tensors, etc."""

    # Regex patterns for identifying quantization types from config keys/values
    _DTYPE_PATTERNS = {
        r"fp8|float8": "fp8",
        r"fp4|float4|mxfp4": "fp4x2",
        r"int8|w8a8": "int8",
        r"int4|w4a16|gptq|awq": "int4x2",
    }

    _QTYPE_PATTERNS = {
        r"block|per_block|blockwise|1x128": QuantType.per_1x128,
        r"per_channel|channel|per_token|token": QuantType.per_Token,
        r"per_tensor|tensor": QuantType.per_Tensor,
        r"per_group|group": QuantType.per_1x32,
    }

    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        quant_method = hf_quant_config.get("quant_method", "")
        config_str = str(hf_quant_config).lower()

        quant_dtype = self._infer_dtype(hf_quant_config, config_str)
        quant_type = self._infer_qtype(hf_quant_config, config_str)
        # MXFP4 (fp4x2) uses microscaling with 1x32 block scaling by definition
        if quant_dtype == d_dtypes.get("fp4x2") and quant_type not in (
            QuantType.per_1x32,
            QuantType.per_1x128,
        ):
            quant_type = QuantType.per_1x32
        # Mxfp8 ``[1, K]`` block to per_1x32.
        weight_block_size = hf_quant_config.get("weight_block_size")
        if (
            isinstance(weight_block_size, (list, tuple))
            and len(weight_block_size) == 2
            and weight_block_size[0] == 1
        ):
            quant_type = QuantType.per_1x32
        is_dynamic = hf_quant_config.get("is_dynamic", True)
        # Each quantizer uses a different key for excluded layers:
        # Quark -> "exclude", compressed-tensors -> "ignore",
        # gpt-oss/HF transformers -> "modules_to_not_convert",
        # MiMo-V2-Flash/HF transformers -> "ignored_layers"
        exclude = list(
            hf_quant_config.get("ignore")
            or hf_quant_config.get("modules_to_not_convert")
            or hf_quant_config.get("exclude")
            or hf_quant_config.get("ignored_layers")
            or []
        )

        global_spec = LayerQuantConfig(
            quant_type=quant_type,
            quant_dtype=quant_dtype,
            is_dynamic=is_dynamic,
            quant_method=quant_method or None,
        )

        return ParsedQuantConfig(global_spec=global_spec, exclude_layers=exclude)

    def _infer_dtype(self, cfg: dict, config_str: str) -> Any:
        # Check explicit fields first
        for key in ("weight_dtype", "activation_dtype", "dtype"):
            val = cfg.get(key)
            if val and isinstance(val, str):
                parsed = _parse_quant_dtype(val)
                if parsed != torch.bfloat16:
                    return parsed
        # Check compressed-tensors config_groups (type + num_bits encoding)
        config_groups = cfg.get("config_groups")
        if isinstance(config_groups, dict):
            for group in config_groups.values():
                if not isinstance(group, dict):
                    continue
                weights = group.get("weights") or {}
                wtype = weights.get("type", "")
                num_bits = weights.get("num_bits")
                if wtype == "float" and num_bits == 8:
                    return d_dtypes.get("fp8", torch.bfloat16)
                if wtype == "float" and num_bits == 4:
                    return d_dtypes.get("fp4x2", torch.bfloat16)
                if wtype == "int" and num_bits == 8:
                    return d_dtypes.get("i8", torch.bfloat16)
        # Fall back to regex heuristics
        for pattern, dtype_key in self._DTYPE_PATTERNS.items():
            if re.search(pattern, config_str):
                return d_dtypes.get(dtype_key, torch.bfloat16)
        return torch.bfloat16

    def _infer_qtype(self, cfg: dict, config_str: str) -> QuantType:
        # Prefer explicit HF/compressed-tensors block size over text heuristics
        # so MXFP8 1x32 and blockscale 1x128/128x128 are not conflated.
        if "weight_block_size" in cfg:
            wbs = cfg.get("weight_block_size")
            if wbs is None:
                return QuantType.per_Tensor
            if isinstance(wbs, (list, tuple)) and len(wbs) >= 2:
                try:
                    m, n = int(wbs[0]), int(wbs[1])
                except (TypeError, ValueError):
                    m = n = None
                if (m, n) == (1, 128):
                    return QuantType.per_1x128
                if (m, n) == (128, 128):
                    # per_128x128 enum has no consumers in linear.py / GEMM dispatch yet;
                    # the per_1x128 path already allocates a (out//128, in//128)
                    # scale grid which is exactly the (128, 128) block layout.
                    return QuantType.per_1x128
                if (m, n) == (1, 32):
                    return QuantType.per_1x32
                return QuantType.per_1x128
        # Check explicit fields
        for key in ("quant_type", "quantization_type", "scheme"):
            val = cfg.get(key)
            if val and isinstance(val, str):
                for pattern, qtype in self._QTYPE_PATTERNS.items():
                    if re.search(pattern, val.lower()):
                        return qtype
        # Check compressed-tensors config_groups for weight strategy
        config_groups = cfg.get("config_groups")
        if isinstance(config_groups, dict):
            for group in config_groups.values():
                if not isinstance(group, dict):
                    continue
                weights = group.get("weights") or {}
                strategy = weights.get("strategy", "")
                if strategy:
                    mapped = _QSCHEME_TO_QUANT_TYPE.get(strategy)
                    if mapped is None:
                        mapped = _QSCHEME_TO_QUANT_TYPE.get(f"per_{strategy}")
                    if mapped is not None:
                        return mapped
        # Fall back to regex heuristics on full config string
        for pattern, qtype in self._QTYPE_PATTERNS.items():
            if re.search(pattern, config_str):
                return qtype
        return QuantType.No
