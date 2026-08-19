# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping
from fnmatch import fnmatch
from types import MappingProxyType
from typing import TYPE_CHECKING

import regex as re

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def is_shared_expert_quant_fse_compatible(
    quant_config: "QuantizationConfig | None",
    expert_prefix: str,
    shared_expert_prefix: str,
    projection_names: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Check whether quantization permits fused shared-expert execution.

    Args:
        quant_config: Model quantization configuration.
        expert_prefix: Routed-expert module prefix.
        shared_expert_prefix: Shared-expert module prefix.
        projection_names: Shared-expert projection names.

    Returns:
        A compatibility flag and, when incompatible, the reason.
    """
    if projection_names is None:
        projection_names = ["gate_up_proj", "down_proj"]

    if quant_config is None:
        return True, None

    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

    if isinstance(quant_config, DeepseekV4FP8Config):
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.models.utils import extract_layer_index

        if quant_config.expert_dtype != "fp4":
            return False, "DeepSeek-V4 routed experts are not MXFP4"

        hf_config = get_current_vllm_config().model_config.hf_config

        # TODO: This is adapted from former `_shared_experts_are_fp4`, and
        # needs to be cleaned up this . There should not be Quark-specific
        # logic in DeepseekV4FP8Config.
        quantization_config = getattr(hf_config, "quantization_config", None)
        if quantization_config is None:
            return False, "DeepSeek-V4 has no quantization configuration"

        if not quant_config._is_quark_mxfp4_ocp(quantization_config):
            return False, "DeepSeek-v4 FSE is only implemented/tested with Quark MXFP4"

        layer_idx = extract_layer_index(shared_expert_prefix)
        if layer_idx >= hf_config.num_hidden_layers:
            shared_expert_prefix = (
                f"mtp.{layer_idx - hf_config.num_hidden_layers}.ffn.shared_experts"
            )
        else:
            shared_expert_prefix = f"layers.{layer_idx}.ffn.shared_experts"

        if any(
            entry.startswith(shared_expert_prefix)
            for entry in quantization_config.get("exclude") or []
            if isinstance(entry, str)
        ):
            return (
                False,
                f"DeepSeek-V4 excludes shared experts at {shared_expert_prefix}",
            )

        shared_expert_weight_name = f"{shared_expert_prefix}.w1"
        layer_quant_config = quantization_config.get("layer_quant_config") or {}
        layer_config = layer_quant_config.get(shared_expert_weight_name)
        if layer_config is None:
            layer_config = next(
                (
                    config
                    for pattern, config in layer_quant_config.items()
                    if isinstance(pattern, str)
                    and pattern.startswith("re:")
                    and re.fullmatch(
                        pattern.removeprefix("re:"), shared_expert_weight_name
                    )
                ),
                None,
            )
        shared_weight_config = (
            layer_config or quantization_config.get("global_quant_config") or {}
        ).get("weight") or {}
        if shared_weight_config.get("dtype") == "fp4":
            return True, None
        return (
            False,
            f"DeepSeek-V4 shared experts at {shared_expert_prefix} are not MXFP4",
        )

    if isinstance(quant_config, QuarkConfig):
        # TODO: layer_type_quant_config is not taken into account here.
        assert "exclude" in quant_config.quant_config
        assert "global_quant_config" in quant_config.quant_config

        is_compatible = not any(
            "shared_expert" in str(entry)
            for entry in quant_config.quant_config["exclude"]
        )
        if not is_compatible:
            return False, f"Quark excludes shared experts at {shared_expert_prefix}"

        global_quant_config = quant_config.quant_config["global_quant_config"]

        def get_projection_quant_configs(layer_name: str) -> list[object]:
            module_prefix, _, projection_name = layer_name.rpartition(".")
            packed_projection_names = quant_config.packed_modules_mapping.get(
                projection_name, [projection_name]
            )
            return [
                quant_config.get_layer_quant_config_from_name(
                    f"{module_prefix}.{packed_projection_name}"
                )
                or global_quant_config
                for packed_projection_name in packed_projection_names
            ]

        expert_quant_config = (
            quant_config.get_layer_quant_config_from_name(expert_prefix)
            or global_quant_config
        )
        shared_expert_quant_configs = [
            config
            for projection_name in projection_names
            for config in get_projection_quant_configs(
                f"{shared_expert_prefix}.{projection_name}"
            )
        ]
        if all(config == expert_quant_config for config in shared_expert_quant_configs):
            return True, None
        return (
            False,
            "Quark uses different quantization configurations for routed and "
            f"shared experts at {shared_expert_prefix}",
        )

    # TODO: Extend FSE support detection to other quantization methods. Typically,
    # one would check that the experts and shared_experts use the same
    # quantization config. This may be refactored as part of QuantizationConfig later.

    return (
        False,
        "shared-expert FSE quantization compatibility is not implemented for "
        f"{type(quant_config).__name__}",
    )


def find_matching_patterns(
    layer_name: str,
    patterns: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
    use_fnmatch: bool = False,
) -> list[set[str]]:
    """Return matching patterns for a layer or each shard of a fused layer.

    A pattern matching the fused layer directly takes precedence. Otherwise,
    return one set of matching patterns for every shard.
    """
    patterns = list(patterns)
    matches = [
        pattern
        for pattern in patterns
        if is_equal_or_regex_match(layer_name, pattern, use_fnmatch=use_fnmatch)
    ]
    if matches:
        return [set(matches)]

    proj_name = layer_name.split(".")[-1]
    if proj_name not in fused_mapping:
        return [set()]

    shard_names = [
        layer_name.replace(proj_name, shard_proj_name)
        for shard_proj_name in fused_mapping[proj_name]
    ]
    per_shard_matches = [
        {
            pattern
            for pattern in patterns
            if is_equal_or_regex_match(shard_name, pattern, use_fnmatch=use_fnmatch)
        }
        for shard_name in shard_names
    ]
    return per_shard_matches


def get_layer_name_after_index(layer_name: str) -> str:
    """Return the suffix following the final numeric component of a layer name."""
    parts = layer_name.split(".")
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].isdigit():
            return ".".join(parts[index + 1 :])
    return layer_name


def is_equal_or_regex_match(
    value: str,
    target: str,
    check_contains: bool = False,
    use_fnmatch: bool = False,
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    If use_fnmatch is set, supports shell-style patterns in target.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value or use_fnmatch and fnmatch(value, target):
        return True
    return False
