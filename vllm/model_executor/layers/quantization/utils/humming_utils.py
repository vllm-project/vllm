# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import regex as re
import torch
from humming.layer import HummingInputSchema, HummingMethod
from humming.schema import BaseWeightSchema

from vllm import envs
from vllm.model_executor.layers.fused_moe.oracle.humming import (
    _process_single_sublayer,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase


def humming_is_layer_skipped(config: dict[str, Any], prefix: str):
    if not config:
        return True

    keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
    ignored_layers: list[str] = []
    for key in keys:
        ignored_layers = config.get(key, []) or []
        if not ignored_layers:
            break

    if any(module_name in prefix for module_name in ignored_layers):
        return True
    if "lm_head" in prefix:
        return True

    for regex in config.get("dynamic", {}):
        if regex[:1] != "-":
            continue
        if re.match(regex[2:], prefix):
            return True

    return False


def prepare_humming_layer(layer: LinearBase, quant_config: dict):
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_schema = HummingInputSchema()

    shape_k_stacks = [layer.input_size_per_partition]
    shape_n_stacks = layer.output_partition_sizes

    # Step 1: convert weight to humming standard format
    weight_schema, tensors = weight_schema.convert_humming(
        tensors=layer.named_parameters(),
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )

    layer.weight_schema = weight_schema

    for name, _ in list(layer.named_parameters()):
        delattr(layer, name)

    for name, tensor in tensors.items():
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    # Step 2: transform weight (humming standard format) for forwarding
    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=layer.output_partition_sizes_sum,
        shape_k=layer.input_size_per_partition,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=layer.has_bias,
        torch_dtype=layer.param_dtype,
    )

    HummingMethod.transform_humming_layer(layer)


def prepare_humming_moe_layer(layer: RoutedExperts, quant_config: dict):
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_quant_config = envs.VLLM_HUMMING_INPUT_QUANT_CONFIG or {}
    if humming_is_layer_skipped(input_quant_config, layer.layer_name):
        input_schema = HummingInputSchema()
    else:
        # TODO: read input_quant_config from quant_config
        input_schema = HummingInputSchema.from_config(input_quant_config)

    is_gated = layer.moe_config.activation.is_gated
    shape_config = {
        "w13": (
            layer.moe_config.intermediate_size_per_partition * 2,
            layer.moe_config.hidden_dim,
        ),
        "w2": (
            layer.moe_config.hidden_dim,
            layer.moe_config.intermediate_size_per_partition * (1 if is_gated else 2),
        ),
    }

    layer.weight_schemas = {}
    layer.input_schemas = {}

    for sublayer_name in shape_config:
        shape_n, shape_k = shape_config[sublayer_name]

        final_weight_schema, final_input_schema = _process_single_sublayer(
            layer=layer,
            sublayer_name=sublayer_name,
            shape_n=shape_n,
            shape_k=shape_k,
            weight_schema=weight_schema,
            input_schema=input_schema,
            has_bias=layer.moe_config.has_bias,
            num_experts=layer.local_num_experts,
            param_dtype=layer.params_dtype,
            force_weight_schema=None,  # No force requant in this code path
        )

        layer.weight_schemas[sublayer_name] = final_weight_schema
        layer.input_schemas[sublayer_name] = final_input_schema

    if not hasattr(layer, "locks"):
        device = layer.w13_weight.device
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        layer.register_buffer("locks", locks)
