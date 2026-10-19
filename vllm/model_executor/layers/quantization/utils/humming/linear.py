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
from vllm.model_executor.utils import (
    get_live_param,
    kernel_parameters,
    move_parameter,
    replace_parameter,
    retire_parameter,
    set_kernel_parameter,
)

if TYPE_CHECKING:
    from vllm.utils.humming import BaseInputSchema, LayerConfig

# A param the conversion leaves unused keeps its loaded data when it is at most
# this fraction of the largest converted tensor (scalars, per-tensor and
# per-channel scales); a larger one is emptied (its data is in the new tensors).
_KEEP_UNUSED_FRACTION = 1 / 256


def replace_layer_parameters(
    layer: torch.nn.Module, tensors: dict[str, torch.Tensor], names: list[str]
) -> None:
    """Write ``tensors`` into ``layer``, keeping existing parameter objects.

    A param in ``names`` that gets no new tensor stays registered: an alias of a
    replaced param follows it, a small one keeps its data (retired), and a large
    one is emptied."""
    replaced = {
        id(param)
        for name in tensors
        if (param := layer._parameters.get(name)) is not None
    }
    limit = _KEEP_UNUSED_FRACTION * max(
        (t.numel() * t.element_size() for t in tensors.values()), default=0
    )
    for name in names:
        param = layer._parameters.get(name)
        if name in tensors or param is None or id(param) in replaced:
            continue
        if param.numel() * param.element_size() <= limit:
            retire_parameter(layer, name)
        else:
            replace_parameter(layer, name, param.data.new_empty(0))
    for name, tensor in tensors.items():
        set_kernel_parameter(layer, name, tensor)


def convert_linear_layer_to_humming_standard(
    layer: LinearBase, name_map: dict[str, str]
):
    """Rename/reshape a linear layer's quantized params (the canonical MPLinear
    layout: ``weight_packed`` int32 + ``weight_scale``) into the parameter names
    and layout humming's weight schema expects (``weight`` / ``weight_scale``)."""
    for name, checkpoint_name in name_map.items():
        param = getattr(layer, checkpoint_name)
        tensor = param.data

        if name == "weight":
            # (input_dim, output_dim) after PWAL, if it transposed the weight.
            input_dim, output_dim = getattr(
                layer,
                "processed_weight_dims",
                (getattr(param, "input_dim", 1), getattr(param, "output_dim", 0)),
            )

            if input_dim == 0 and output_dim == 1:
                tensor = tensor.transpose(1, 0).contiguous()
            else:
                assert output_dim == 0 and input_dim == 1

            tensor = tensor.view(tensor.size(0), -1).view(torch.int32)
        elif name in ["weight_scale", "zero_point"]:
            if getattr(param, "output_dim", 0) == 1:
                tensor = tensor.transpose(0, 1).contiguous()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)

            tensor = tensor.view(torch.int32) if name == "zero_point" else tensor

        if name != checkpoint_name:
            # One param under both names: the loader keeps its name.
            move_parameter(layer, checkpoint_name, name)
        replace_parameter(layer, name, tensor)


def prepare_humming_linear_layer_config(
    layer: LinearBase,
    quant_config: dict,
    input_quant_config: dict | None = None,
    *,
    input_schema: "BaseInputSchema | None" = None,
    allow_input_schema_fallback: bool = True,
) -> "LayerConfig":
    from vllm.utils.humming import (
        BaseInputSchema,
        BaseWeightSchema,
        HummingInputSchema,
        prepare_and_transform_humming_tensors,
    )

    weight_schema = BaseWeightSchema.from_config(quant_config)
    if input_schema is None:
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
    source_tensors = kernel_parameters(layer)
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

    if (
        weight_schema.has_zero_point
        and input_schema.a_dtype is not None
        and weight_schema.b_dtype.num_bits == input_schema.a_dtype.num_bits
    ):
        raise ValueError(
            "Humming does not support asymmetric (zero_point) weights "
            "when weight and activation bit widths are equal "
            f"(W{weight_schema.b_dtype.num_bits}"
            f"A{input_schema.a_dtype.num_bits}). "
            "Use symmetric weights or a different activation bit width."
        )

    tensors.update(input_tensors)
    layer.weight_schema = weight_schema
    input_schema = check_and_fallback_input_schema(
        weight_schema=weight_schema,
        input_schema=input_schema,
        param_dtype=layer.params_dtype,
        allow_fallback=allow_input_schema_fallback,
    )
    for name in ("input_scale", "input_scale_2"):
        if name != input_schema.static_tensor_scale_name:
            tensors.pop(name, None)
            input_tensors.pop(name, None)

    # Step 2: transform weight (humming standard format) for forwarding.
    config, tensors = prepare_and_transform_humming_tensors(
        tensors=tensors,
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
    tensors.update(input_tensors)
    if "bias" in tensors:
        zero_bias = torch.zeros_like(tensors["bias"])
        layer.register_buffer("zero_bias", zero_bias)
    replace_layer_parameters(layer, tensors, list(layer._parameters))

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
    skip_bias_add: bool = False,
    layer_config: "LayerConfig",
    compute_config: str,
    locks: torch.Tensor,
) -> torch.Tensor:
    from vllm.utils.humming import humming_forward

    mode = layer_config.input_quant_mode
    input_scale = get_live_param(layer, "input_scale") if mode.should_quantize else None
    input_scale_2 = (
        get_live_param(layer, "input_scale_2") if mode.has_secondary_scale else None
    )
    flatten_inputs = x.reshape(-1, x.size(-1))
    output = humming_forward(
        layer_config,
        inputs=flatten_inputs,
        weight=layer.weight,
        weight_scale=get_live_param(layer, "weight_scale"),
        zero_point=get_live_param(layer, "zero_point"),
        bias=get_live_param(layer, "zero_bias" if skip_bias_add else "bias"),
        weight_scale_2=get_live_param(layer, "weight_scale_2"),
        input_scale=input_scale,
        input_scale_2=input_scale_2,
        hadamard_block_size=layer.weight_schema.hadamard_block_size,
        locks=locks,
        compute_config=compute_config,
    )
    return output.view(*x.shape[:-1], output.size(-1))
