# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any

import regex as re
import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.utils.humming import BaseWeightSchema, HummingInputSchema, HummingMethod


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


def convert_linear_layer_to_humming_standard(
    layer: LinearBase, name_map: dict[str, str]
):
    """Rename/reshape a linear layer's quantized params (the canonical MPLinear
    layout: ``weight_packed`` int32 + ``weight_scale``) into the parameter names
    and layout humming's weight schema expects (``weight`` / ``weight_scale``)."""
    for name, checkpoint_name in name_map.items():
        tensor = getattr(layer, checkpoint_name)
        delattr(layer, checkpoint_name)

        if name == "weight":
            input_dim = getattr(tensor, "input_dim", 1)
            output_dim = getattr(tensor, "output_dim", 0)

            if input_dim == 0 and output_dim == 1:
                tensor = tensor.transpose(1, 0).contiguous()
            else:
                assert output_dim == 0 and input_dim == 1

            tensor = tensor.view(tensor.size(0), -1).view(torch.int32)
        elif name in ["weight_scale", "zero_point"]:
            if getattr(tensor, "output_dim", 0) == 1:
                tensor = tensor.transpose(0, 1).contiguous()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)

            tensor = tensor.view(torch.int32) if name == "zero_point" else tensor

        if isinstance(tensor, torch.nn.Parameter):
            param = tensor
        else:
            param = torch.nn.Parameter(tensor, requires_grad=False)

        setattr(layer, name, param)


def prepare_humming_layer(layer: LinearBase, quant_config: dict):
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_schema = HummingInputSchema()

    # ReplicatedLinear has no TP partitioning and so does not set
    # input_size_per_partition; for it that is just input_size. Use hasattr
    # rather than getattr's default arg, which is evaluated eagerly and would
    # raise on layers lacking input_size (e.g. ParallelLMHead).
    if hasattr(layer, "input_size_per_partition"):
        input_size_per_partition = layer.input_size_per_partition
    else:
        input_size_per_partition = layer.input_size
    shape_k_stacks = [input_size_per_partition]
    shape_n_stacks = layer.output_partition_sizes

    # Step 1: convert weight to humming standard format
    weight_schema, tensors = weight_schema.convert_humming(
        tensors=dict(layer.named_parameters()),
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )

    layer.weight_schema = weight_schema

    for name, _ in list(layer.named_parameters()):
        delattr(layer, name)

    for name, tensor in tensors.items():
        if isinstance(tensor, torch.nn.Parameter):
            tensor = tensor.data
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    # Step 2: transform weight (humming standard format) for forwarding
    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=sum(layer.output_partition_sizes),
        shape_k=input_size_per_partition,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=layer.has_bias,
        torch_dtype=layer.params_dtype,
    )

    HummingMethod.transform_humming_layer(layer)
    if not hasattr(layer, "locks"):
        device = layer.weight.device
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        layer.register_buffer("locks", locks)

    compute_config = {
        "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
        "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
        "gemm_type": "dense",
    }

    layer.compute_config = json.dumps(compute_config)


def prepare_humming_moe_layer(layer: RoutedExperts, quant_config: dict):
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_quant_config = envs.VLLM_HUMMING_INPUT_QUANT_CONFIG or {}
    if humming_is_layer_skipped(input_quant_config, layer.layer_name):
        input_schema = HummingInputSchema()
    else:
        # TODO: read input_quant_config from quant_config
        input_schema = HummingInputSchema.from_config(input_quant_config)

    is_gated = layer.activation.is_gated
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
        # Step 1: convert weight to humming standard format
        tensors: dict[str, torch.Tensor] = dict(
            (key.removeprefix(sublayer_name + "_"), value)
            for key, value in layer.state_dict().items()
            if key.startswith(sublayer_name + "_")
        )

        shape_n, shape_k = shape_config[sublayer_name]
        shape_n_stacks = [shape_n]
        shape_k_stacks = [shape_k]
        if sublayer_name == "w13":
            shape_n_stacks = [shape_n // 2] * 2

        weight_schema_new, tensors = weight_schema.convert_humming(
            tensors=tensors,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            num_experts=layer.local_num_experts,
            param_dtype=layer.params_dtype,
        )

        layer.weight_schemas[sublayer_name] = weight_schema_new
        layer.input_schemas[sublayer_name] = input_schema

        for name, _ in list(layer.named_parameters()):
            if not name.startswith(sublayer_name + "_"):
                continue
            delattr(layer, name)

        for name, tensor in tensors.items():
            name = f"{sublayer_name}_{name}"
            param = torch.nn.Parameter(tensor, requires_grad=False)
            setattr(layer, name, param)

        # Step 2: transform weight (humming standard format) for forwarding
        HummingMethod.prepare_layer_meta(
            layer=layer,
            shape_n=shape_n,
            shape_k=shape_k,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            input_schema=input_schema,
            weight_schema=weight_schema_new,
            has_bias=layer.moe_config.has_bias,
            num_experts=layer.num_experts,
            torch_dtype=layer.params_dtype,
            sublayer_name=sublayer_name,
        )

        HummingMethod.transform_humming_layer(layer, sublayer_name=sublayer_name)

    if not hasattr(layer, "locks"):
        device = layer.w13_weight.device
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        layer.register_buffer("locks", locks)


def get_humming_moe_quant_config(
    layer: RoutedExperts,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
):
    input_schema = layer.input_schemas["w13"]
    weight_schema = layer.weight_schemas["w13"]

    a_dtype = input_schema.a_dtype
    if a_dtype is None or a_dtype.num_bits == 16:
        a_quant_desc = FusedMoEQuantDesc(dtype=None)
    else:
        shape = GroupShape(row=1, col=-1)
        a_quant_desc = FusedMoEQuantDesc(dtype=str(a_dtype), shape=shape)

    weight_scale_group_size = weight_schema.weight_scale_group_size
    weight_scale_group_size_n = weight_schema.weight_scale_group_size_n
    weight_group_shape: tuple[int, ...] = ()
    if weight_scale_group_size_n > 1:
        weight_group_shape = GroupShape(
            row=weight_scale_group_size,
            col=weight_scale_group_size_n,
        )
    elif weight_scale_group_size == 0:
        weight_group_shape = GroupShape(row=-1, col=1)
    else:
        weight_group_shape = GroupShape(row=weight_scale_group_size, col=1)

    w1_quant_desc = FusedMoEQuantDesc(
        dtype=str(weight_schema.b_dtype),
        shape=weight_group_shape,
        scale=getattr(layer, "w13_weight_scale", None),
        alpha_or_gscale=getattr(layer, "w13_global_scale", None),
        zp=getattr(layer, "w13_zero_point", None),
        bias=getattr(layer, "w13_bias", None),
    )

    w2_quant_desc = FusedMoEQuantDesc(
        dtype=str(weight_schema.b_dtype),
        shape=weight_group_shape,
        scale=getattr(layer, "w2_weight_scale", None),
        alpha_or_gscale=getattr(layer, "w2_global_scale", None),
        zp=getattr(layer, "w2_zero_point", None),
        bias=getattr(layer, "w2_bias", None),
    )

    return FusedMoEQuantConfig(
        _a1=a_quant_desc,
        _a2=a_quant_desc,
        _w1=w1_quant_desc,
        _w2=w2_quant_desc,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
