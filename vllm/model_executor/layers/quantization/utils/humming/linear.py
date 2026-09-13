# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and execute Humming linear layers."""

import json
from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.humming.schema import (
    check_and_fallback_input_schema,
)

if TYPE_CHECKING:
    from vllm.utils.humming import LayerConfig


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


def prepare_humming_linear_layer_config(
    layer: LinearBase,
    quant_config: dict,
    input_quant_config: dict | None = None,
) -> "LayerConfig":
    from vllm.utils.humming import (
        BaseInputSchema,
        BaseWeightSchema,
        HummingInputSchema,
        prepare_layer_config,
        transform_humming_tensors,
    )

    weight_schema = BaseWeightSchema.from_config(quant_config)
    if input_quant_config is not None:
        input_schema = BaseInputSchema.from_config(input_quant_config)
    else:
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

    # Step 1: convert weight and input schemas to humming standard format
    source_tensors = dict(layer.named_parameters())
    weight_schema, tensors = weight_schema.convert_humming(
        tensors=source_tensors,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )
    input_schema, input_tensors = input_schema.convert_humming(
        tensors=source_tensors,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )

    tensors.update(input_tensors)
    layer.weight_schema = weight_schema
    input_schema = check_and_fallback_input_schema(
        weight_schema=weight_schema,
        input_schema=input_schema,
        param_dtype=layer.params_dtype,
    )

    # Step 2: transform weight (humming standard format) for forwarding.
    config = prepare_layer_config(
        shape_n=sum(layer.output_partition_sizes),
        shape_k=input_size_per_partition,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=layer.has_bias,
        torch_dtype=layer.params_dtype,
        device=tensors["weight"].device,
    )
    tensors = transform_humming_tensors(config, tensors)
    tensors.update(input_tensors)
    for name, _ in list(layer.named_parameters()):
        delattr(layer, name)
    for name, tensor in tensors.items():
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    return config


def get_humming_linear_compute_config() -> str:
    return json.dumps(
        {
            "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": "dense",
        }
    )


def apply_humming_linear(
    layer: LinearBase,
    x: torch.Tensor,
    *,
    layer_config: "LayerConfig",
    compute_config: str,
    locks: torch.Tensor,
) -> torch.Tensor:
    from vllm.utils.humming import humming_forward

    flatten_inputs = x.reshape(-1, x.size(-1))
    output = humming_forward(
        layer_config,
        inputs=flatten_inputs,
        weight=layer.weight,
        weight_scale=getattr(layer, "weight_scale", None),
        zero_point=getattr(layer, "zero_point", None),
        bias=getattr(layer, "bias", None),
        weight_scale_2=getattr(layer, "weight_scale_2", None),
        input_scale=getattr(layer, "input_scale", None),
        input_scale_2=getattr(layer, "input_scale_2", None),
        hadamard_block_size=layer.weight_schema.hadamard_block_size,
        locks=locks,
        compute_config=compute_config,
    )
    return output.view(*x.shape[:-1], output.size(-1))
