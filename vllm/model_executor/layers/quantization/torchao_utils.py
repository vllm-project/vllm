# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import copy

import regex as re
from compressed_tensors.quantization import QuantizationArgs

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)


def maybe_get_torchao_config_for_moe_layer(
    fqn_to_config,
    moe_prefix: str,
):
    """
    Given an `fqn_to_config` and the prefix FQN of a vLLM model's MoE module,
    returns either a single `TorchAOBaseConfig` to apply to all of the expert
    weights, or `None` if the expert weights are not quantized.

    Throws an exception if multiple configs are found.

    Example moe_prefix: `model.layers.1.mlp.experts`, from
      `Qwen/Qwen1.5-MoE-A2.7B`
    """

    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

    found_configs = []
    for k, cur_config in fqn_to_config.fqn_to_config.items():
        if k.startswith("re:"):
            # regex match

            # 're:pattern' -> 'pattern'
            regex_pattern = k[3:]

            # It's tricky to match regexes defined on HuggingFace's model
            # definition against FQNs from the vLLM's model definion,
            # as the two definitions often have mismatches.
            #
            # For now, do an approximation:
            # 1. transform vLLM's prefix to include
            #    `gate_proj`, `up_proj`, `gate_up_proj`, `down_proj` to cover
            #    common cases.
            # 2. match regular expressions defined on HuggingFace's model
            #    definition against these transformed prefixes.

            transformed_vllm_prefixes = [
                f"{moe_prefix}.gate_proj",
                f"{moe_prefix}.up_proj",
                f"{moe_prefix}.gate_up_proj",
                f"{moe_prefix}.down_proj",
            ]

            for transformed_vllm_prefix in transformed_vllm_prefixes:
                is_match = re.fullmatch(regex_pattern, transformed_vllm_prefix)
                if is_match:
                    found_configs.append(cur_config)

        else:
            # direct match by module or parameter FQN
            if k.startswith(moe_prefix):
                found_configs.append(cur_config)

    # all entries in found_configs have to match
    for config_idx in range(len(found_configs)):
        if config_idx == 0:
            continue
        first_config = found_configs[0]
        cur_config = found_configs[config_idx]
        if cur_config != first_config:
            raise AssertionError(
                f"inconsistent configs {first_config} and "
                f"{cur_config} in a single MoE module, this is "
                "not supported"
            )

    first_config = None
    if len(found_configs):
        first_config = found_configs[0]
    if first_config is None:
        first_config = fqn_to_config.fqn_to_config.get("_default", None)

    if isinstance(
        first_config, Float8DynamicActivationFloat8WeightConfig
    ) and first_config.granularity[1] == PerRow(1):
        # For some models, HuggingFace stores expert weights in MKN layout
        # and vLLM loads them in MNK layout. If we detect that the
        # Hugging weight was quantized along the K dimension, change the
        # quantization config for vLLM to adjust for the fact that `K`'s index
        # is different in the vLLM definition.

        # deepcopy to prevent affecting other users of this config
        first_config = copy.deepcopy(first_config)

        # Change the granularity (this adjusts the quantization axis to match
        # vLLM's weight layout).
        first_config.granularity[-1] = PerRow()

    return first_config


def torchao_config_to_compressed_tensors_config(
    torchao_config,
) -> CompressedTensorsConfig:
    """
    Map from torchao config format to compressed-tensors config format
    """

    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerRow,
    )

    is_float8_rowwise = isinstance(
        torchao_config, Float8DynamicActivationFloat8WeightConfig
    ) and torchao_config.granularity == [PerRow(), PerRow()]
    # Special case of float8 rowwise where the HuggingFace weight is stored
    # in MKN layout and quantized rowwise along the K dimension
    is_float8_rowwise_for_3d_weight_mkn_layout = isinstance(
        torchao_config, Float8DynamicActivationFloat8WeightConfig
    ) and torchao_config.granularity == [PerRow(), PerRow(1)]

    if is_float8_rowwise or is_float8_rowwise_for_3d_weight_mkn_layout:
        # activations: float8 token-wise
        # weights: float8 channel-wise

        # source: https://gist.github.com/vkuzo/d30bcf8a2e4a1a3f3d122922f31e3572
        target_scheme_map = {
            "Linear": {
                "weights": QuantizationArgs(
                    num_bits=8,
                    type="float",
                    symmetric=True,
                    group_size=None,
                    strategy="channel",
                    block_structure=None,
                    dynamic=False,
                    actorder=None,
                    observer="minmax",
                    observer_kwargs={},
                ),
                "input_activations": QuantizationArgs(
                    num_bits=8,
                    type="float",
                    symmetric=True,
                    group_size=None,
                    strategy="token",
                    block_structure=None,
                    dynamic=True,
                    actorder=None,
                    observer="minmax",
                    observer_kwargs={},
                ),
            },
        }
        compressed_tensors_quant_config = CompressedTensorsConfig(
            target_scheme_map=target_scheme_map,
            # TODO(future): fill out the ignore map / audit why it matters
            ignore=[],
            quant_format="float-quantized",
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )
        return compressed_tensors_quant_config

    raise AssertionError(f"unsupported {torchao_config=}")
