# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING

import regex as re
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

if TYPE_CHECKING:
    from ..gptq import GPTQConfig
    from ..gptq_marlin import GPTQMarlinConfig
else:
    GPTQConfig = object
    GPTQMarlinConfig = object


# Match dynamic rules with module name (prefix) and override quantize
# config if module (prefix) matches a rule
def override_config(config: GPTQConfig | GPTQMarlinConfig, prefix: str):
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act

    config.pack_factor = Fraction(32, config.weight_bits)  # packed into int32
    if config.get_name() == "gptq_marlin":
        assert isinstance(config, GPTQMarlinConfig)
        is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)
        if isinstance(is_sym, bool):
            config.is_sym = is_sym

        if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:
            raise ValueError(
                "Unsupported quantization config: "
                f"bits={config.weight_bits}, sym={config.is_sym}"
            )

        config.quant_type = config.TYPE_MAP[(config.weight_bits, config.is_sym)]
    elif config.get_name() == "gptq":
        assert isinstance(config, GPTQConfig)
        if config.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {config.weight_bits} bits."
            )


def get_dynamic_override(
    config: GPTQConfig | GPTQMarlinConfig,
    layer_name: str,
    key: str | None = None,
    default_value: int | bool | None = None,
) -> dict | int | bool | None:
    for pattern, pattern_dict in config.dynamic.items():
        # Negative match: matched modules are excluded from quantized init
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), layer_name):
                return False
        # Positive match: matched modules have quant properties overrides
        # base quant config
        elif re.match(pattern.removeprefix("+:"), layer_name):
            if key is None:
                return pattern_dict
            else:
                return pattern_dict.get(key, default_value)
    return default_value


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

    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

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
    config: GPTQConfig | GPTQMarlinConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
):
    cloned_config = deepcopy(config)
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
