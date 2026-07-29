# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from copy import copy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

if TYPE_CHECKING:
    from ..auto_gptq import AutoGPTQConfig
else:
    AutoGPTQConfig = object


def _clone_config(config: AutoGPTQConfig) -> AutoGPTQConfig:
    # Per-layer overrides only mutate scalar fields; large rule tables are shared.
    cloned_config = copy(config)
    cloned_config.full_config = config.full_config.copy()
    return cloned_config


# Match dynamic rules with module name (prefix) and override quantize
# config if module (prefix) matches a rule
def override_config(config: AutoGPTQConfig, prefix: str):
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act

    config.pack_factor = 32 // config.weight_bits  # packed into int32
    assert isinstance(config, AutoGPTQConfig)
    is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)
    if isinstance(is_sym, bool):
        config.is_sym = is_sym

    if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:
        raise ValueError(
            "Unsupported quantization config: "
            f"bits={config.weight_bits}, sym={config.is_sym}"
        )

    config.quant_type = config.TYPE_MAP[(config.weight_bits, config.is_sym)]


_REGEX_META_CHARACTERS = frozenset(r".^$*+?{}[]()|")


def _get_literal_pattern(pattern: str) -> str | None:
    """Return the literal matched by an anchored regex, when it has one."""
    if not pattern.startswith("^") or not pattern.endswith("$"):
        return None

    literal = []
    index = 1
    end = len(pattern) - 1
    while index < end:
        character = pattern[index]
        if character == "\\":
            index += 1
            if index >= end or pattern[index].isalnum():
                return None
            literal.append(pattern[index])
        elif character in _REGEX_META_CHARACTERS:
            return None
        else:
            literal.append(character)
        index += 1
    return "".join(literal)


def _get_dynamic_rule_matcher(config: AutoGPTQConfig) -> dict[str, Any]:
    cache = getattr(config, "_dynamic_rule_matcher", None)
    if (
        cache is not None
        and cache["dynamic"] is config.dynamic
        and cache["size"] == len(config.dynamic)
    ):
        return cache

    exact_rules: dict[str, tuple[int, bool, dict]] = {}
    regex_rules: list[tuple[int, Any, bool, dict]] = []
    for index, (pattern, pattern_dict) in enumerate(config.dynamic.items()):
        # GPTQModel often emits thousands of exact rules. Index those rules
        # while preserving the original first-match order for real regexes.
        is_negative = pattern.startswith("-:")
        regex_pattern = pattern.removeprefix("-:").removeprefix("+:")
        literal_pattern = _get_literal_pattern(regex_pattern)
        if literal_pattern is not None:
            exact_rules.setdefault(
                literal_pattern,
                (index, is_negative, pattern_dict),
            )
        else:
            regex_rules.append(
                (index, re.compile(regex_pattern), is_negative, pattern_dict)
            )

    cache = {
        "dynamic": config.dynamic,
        "size": len(config.dynamic),
        "exact_rules": exact_rules,
        "regex_rules": regex_rules,
    }
    config._dynamic_rule_matcher = cache
    return cache


def _find_dynamic_rule(
    config: AutoGPTQConfig,
    layer_name: str,
) -> tuple[bool, bool, dict]:
    matcher = _get_dynamic_rule_matcher(config)
    exact_rule = matcher["exact_rules"].get(layer_name)
    exact_rule_index = exact_rule[0] if exact_rule is not None else None

    for index, pattern, is_negative, pattern_dict in matcher["regex_rules"]:
        if exact_rule_index is not None and index > exact_rule_index:
            break
        if pattern.match(layer_name):
            return True, is_negative, pattern_dict

    if exact_rule is not None:
        _, is_negative, pattern_dict = exact_rule
        return True, is_negative, pattern_dict
    return False, False, {}


def _match_dynamic_override(
    config: AutoGPTQConfig,
    layer_name: str,
    key: str | None,
    default_value: int | bool | None,
) -> tuple[bool, dict | int | bool | None]:
    matched, is_negative, pattern_dict = _find_dynamic_rule(config, layer_name)
    if not matched:
        return False, default_value
    # Negative match: matched modules are excluded from quantized init
    if is_negative:
        return True, False
    # Positive match: matched modules have quant properties overrides
    # base quant config
    if key is None:
        return True, pattern_dict
    return True, pattern_dict.get(key, default_value)


def _format_shard_values(
    shard_proj_names: list[str],
    shard_values: list[dict | int | bool | None],
) -> dict[str, dict | int | bool | None]:
    return dict(zip(shard_proj_names, shard_values, strict=True))


def _get_fused_shards(
    layer_name: str,
    fused_mapping: Mapping[str, list[str]],
) -> tuple[str, list[str]] | None:
    def is_module_suffix(module_name: str) -> bool:
        suffix = module_name if module_name.startswith(".") else f".{module_name}"
        return layer_name == module_name or layer_name.endswith(suffix)

    fused_name = max(
        (name for name in fused_mapping if is_module_suffix(name)),
        key=len,
        default=None,
    )
    if fused_name is None:
        return None
    return (fused_name, fused_mapping[fused_name])


def get_dynamic_override(
    config: AutoGPTQConfig,
    layer_name: str,
    key: str | None = None,
    default_value: int | bool | None = None,
) -> dict | int | bool | None:
    matched, value = _match_dynamic_override(config, layer_name, key, default_value)
    if matched:
        return value

    fused_shards = _get_fused_shards(layer_name, config.packed_modules_mapping)
    if fused_shards is None:
        return default_value

    # Dynamic rules use unfused checkpoint names, so retry each logical shard.
    fused_name, shard_proj_names = fused_shards
    layer_prefix = layer_name.removesuffix(fused_name)
    shard_matches = [
        _match_dynamic_override(
            config,
            f"{layer_prefix}{shard_proj_name}",
            key,
            default_value,
        )
        for shard_proj_name in shard_proj_names
    ]

    if key is None:
        negative_matches = [
            matched and value is False for matched, value in shard_matches
        ]
        if any(negative_matches):
            if all(negative_matches):
                return False
            shard_values = [
                value if matched else default_value for matched, value in shard_matches
            ]
            raise ValueError(
                f"Dynamic quantization config for fused layer {layer_name} "
                "does not match across shards: "
                f"{_format_shard_values(shard_proj_names, shard_values)}"
            )

        for matched, value in shard_matches:
            if matched:
                return value
        return default_value

    shard_values = [
        value if matched else default_value for matched, value in shard_matches
    ]
    if any(value != shard_values[0] for value in shard_values[1:]):
        raise ValueError(
            f"Dynamic quantization config for fused layer {layer_name} "
            f"does not match across shards for {key}: "
            f"{_format_shard_values(shard_proj_names, shard_values)}"
        )
    return shard_values[0]


def flatten_list(lst: list[Any]) -> list[Any]:
    output = []

    def _flatten(lst: list[Any]):
        for i in lst:
            if isinstance(i, list):
                _flatten(i)
            else:
                output.append(i)

    _flatten(lst)
    return output


def is_layer_gptq_quantized(
    prefix: str,
    quantized_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    # prefix: model.layers.0.self_attn.q_proj
    # proj_name: q_proj

    # GPTQ's `modules_in_block_to_quantize`:
    # Substr: ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]
    # Full prefix ["model.layers.0.self_attn.q_proj"]

    quantized_layers = flatten_list(quantized_layers)

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    fused_shards = _get_fused_shards(prefix, fused_mapping)
    if fused_shards is not None:
        fused_name, shard_names = fused_shards
        layer_prefix = prefix.removesuffix(fused_name)
        shard_prefixes = [f"{layer_prefix}{shard_name}" for shard_name in shard_names]

        is_quantized = None
        for shard_prefix in shard_prefixes:
            is_shard_quantized = any(
                layer in shard_prefix for layer in quantized_layers
            )

            if is_quantized is None:
                is_quantized = is_shard_quantized
            elif is_shard_quantized != is_quantized:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_quantized = any(layer in prefix for layer in quantized_layers)

    assert is_quantized is not None
    return is_quantized


def get_linear_quant_method(
    config: AutoGPTQConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
):
    cloned_config = _clone_config(config)
    parallel_lm_head_quantized = (
        isinstance(layer, ParallelLMHead) and cloned_config.lm_head_quantized
    )
    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
        is_layer_quantized = is_layer_gptq_quantized(
            prefix=prefix,
            quantized_layers=cloned_config.modules_in_block_to_quantize,
            fused_mapping=cloned_config.packed_modules_mapping,
        )
        # False = skip module, None = no override, else = Positive match
        if get_dynamic_override(  # noqa: E712
            cloned_config,  # noqa: E712
            layer_name=prefix,
        ) == False or (not is_layer_quantized):  # noqa: E712
            if parallel_lm_head_quantized:
                return UnquantizedEmbeddingMethod()
            return UnquantizedLinearMethod()

        if prefix:
            # Dynamic per module/layer rules may override base config
            override_config(cloned_config, prefix=prefix)

        return linear_method_cls(cloned_config)
    return None
