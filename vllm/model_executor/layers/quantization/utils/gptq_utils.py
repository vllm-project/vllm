# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from copy import deepcopy
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


def gptq_qzeros_to_standard_format(
    qzeros: torch.Tensor, size_bits: int, stores_zero_minus_one: bool
) -> torch.Tensor:
    """Convert GPTQ qzeros to the layout the WNA16 kernels take for zero points.

    GPTQ stores qzeros as (G, N // pack) int32, packed along the output dim.
    The v1 checkpoint format (AutoGPTQ, GPTQModel "gptq") stores z - 1; the
    "gptq_v2" format stores z. Returns raw zero points as (N // pack, G),
    still packed along the output dim (the compressed-tensors layout, which
    is also what the AWQ path converts to). The +1 is applied per element
    modulo 2**size_bits, so it never carries into the neighbouring value.
    """
    mask = (1 << size_bits) - 1
    shifts = torch.arange(0, 32, size_bits, dtype=torch.int32, device=qzeros.device)
    unpacked = (qzeros.unsqueeze(-1) >> shifts) & mask  # (G, N // pack, pack)
    if stores_zero_minus_one:
        unpacked = (unpacked + 1) & mask
    repacked = (unpacked.to(torch.int32) << shifts).sum(dim=-1, dtype=torch.int32)
    return repacked.t().contiguous()


def normalize_and_validate_gptq_desc_act(
    desc_act: bool,
    group_size: int | None,
    dynamic: Mapping[str, Any],
) -> bool:
    base_desc_act = desc_act
    if desc_act and group_size == -1:
        # With one group per output channel, activation ordering is a no-op.
        desc_act = False
    if desc_act:
        raise ValueError(
            "GPTQ group activation ordering (desc_act=True) is no longer "
            "supported. Use a checkpoint with static activation ordering "
            "or desc_act=False."
        )

    for pattern, overrides in dynamic.items():
        if pattern.startswith("-:") or not isinstance(overrides, Mapping):
            continue
        dynamic_desc_act = overrides.get("desc_act", base_desc_act)
        dynamic_group_size = overrides.get("group_size", group_size)
        if dynamic_desc_act and dynamic_group_size != -1:
            raise ValueError(
                "GPTQ group activation ordering (desc_act=True) is no longer "
                f"supported, but is enabled by dynamic rule {pattern!r}. Use "
                "a checkpoint with static activation ordering or desc_act=False."
            )
    return desc_act


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

    if config.desc_act and config.group_size == -1:
        # Activation ordering is a no-op for channelwise quantization.
        config.desc_act = False

    config.pack_factor = 32 // config.weight_bits  # packed into int32
    assert isinstance(config, AutoGPTQConfig)
    is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)
    if isinstance(is_sym, bool):
        config.is_sym = is_sym

    config.quant_type = config.get_quant_type(config.weight_bits, config.is_sym)


def get_dynamic_override(
    config: AutoGPTQConfig,
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

    proj_name = prefix.split(".")[-1]

    quantized_layers = flatten_list(quantized_layers)

    # Empty means no metadata could be read at all, not that the checkpoint
    # holds no quantized layer, and loading GPTQ weights unquantized cannot work.
    if not quantized_layers:
        return True

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
    config: AutoGPTQConfig,
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
