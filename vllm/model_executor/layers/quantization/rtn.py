# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright © 2025, Oracle and/or its affiliates.

import os
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_rtn_marlin_linear,
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)
"""By default, use 8 bit as target precision, but it can be 
overridden by setting the RTN_NUM_BITS envvar
"""
NUM_BITS = os.getenv("RTN_NUM_BITS", "8")
"""By default, use group size of 128 parameters, but it can be 
overridden by setting the RTN_GROUP_SIZE envvar
"""
GROUP_SIZE = os.getenv("RTN_GROUP_SIZE", "128")
"""Global Marlin workspace shared by all modules
"""
workspace = None


class RTNConfig(QuantizationConfig):
    """Config class for RTN."""

    def __init__(
        self,
        weight_bits: int = int(NUM_BITS),
        group_size: int = int(GROUP_SIZE),
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size

        if self.weight_bits != 4 and self.weight_bits != 8:
            raise ValueError(
                "Currently, only 4-bit or 8-bit weight quantization is "
                f"supported for RTN, but got {self.weight_bits} bits."
            )

        self.quant_type = (
            scalar_types.uint8b128 if self.weight_bits == 8 else scalar_types.uint4b8
        )

    def __repr__(self) -> str:
        return (
            f"RTNConfig(weight_bits={self.weight_bits}, group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "rtn"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RTNConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return RTNLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return RTNMoEMethod(self, layer.moe_config)
        return None


class RTNTensor:
    """A wrapper over Tensor that enables quantization on-the-fly by
    overloading the copy_ method.
    """

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.data = data
        self.scale = scale
        self.quant_config = quant_config

    def narrow(self, dim, start, length):
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        return RTNTensor(
            self.data.narrow(dim, start // factor, length // factor),
            self.scale.narrow(dim, start, length),
            self.quant_config,
        )

    def __getitem__(self, key):
        return RTNTensor(self.data[key], self.scale[key], self.quant_config)

    @property
    def shape(self):
        shape = self.data.shape
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        batch_present = len(shape) == 3
        if batch_present:
            return torch.Size((shape[0], shape[1] * factor, shape[2]))
        else:
            return torch.Size((shape[0] * factor, shape[1]))

    def copy_(self, loaded_weight: torch.Tensor) -> None:
        qweight, weight_scale = rtn_quantize(
            loaded_weight.cuda(),
            self.quant_config.weight_bits,
            self.quant_config.group_size,
        )

        self.data.copy_(qweight)
        self.scale.data.copy_(weight_scale)


class RTNParameter(Parameter):
    """A wrapper over Parameter that returns RTNTensor (a wrapper over Tensor)
    when its data is accessed. We need this wrapper for the data loading phase
    only, so we can intercept a weight copying function (torch.Tensor.copy_)
    and apply quantization on-the-fly.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.scale = scale
        self.quant_config = quant_config

    @property
    def data(self):
        return RTNTensor(super().data, self.scale, self.quant_config)


class RTNLinearMethod(LinearMethodBase):
    """Linear method for RTN.

    Args:
        quant_config: The RTN quantization config.
    """

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        num_groups_per_col = (
            input_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )

        scale = Parameter(
            torch.empty(
                output_size_per_partition, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        factor = 1 if self.quant_config.weight_bits == 8 else 2

        weight = RTNParameter(
            data=torch.empty(
                output_size_per_partition // factor,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=scale,
            quant_config=self.quant_config,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("scale", scale)
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack weights and scales for Marlin kernels."""
        weight_bits = self.quant_config.weight_bits

        weight, scale = repack_weights(layer.weight, layer.scale, weight_bits)

        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "scale", scale)

        init_workspace(layer.weight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_rtn_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.scale,
            workspace=workspace,
            quant_type=self.quant_config.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            bias=bias,
        )


class RTNMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: RTNConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        factor = 1 if self.quant_config.weight_bits == 8 else 2

        # Fused gate_up_proj (column parallel)
        num_groups_per_col = (
            hidden_size // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w13_scale = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                num_groups_per_col,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale", w13_scale)

        w13_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition // factor,
                hidden_size,
                dtype=torch.uint8,
            ),
            scale=w13_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        num_groups_per_col = (
            intermediate_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w2_scale = Parameter(
            torch.zeros(
                num_experts, hidden_size, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale", w2_scale)

        w2_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                hidden_size // factor,
                intermediate_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=w2_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack weights and scales for Marlin kernels."""
        weight_bits = self.quant_config.weight_bits

        w13_weight, w13_scale = repack_weights(
            layer.w13_weight, layer.w13_scale, weight_bits
        )
        replace_parameter(layer, "w13_weight", w13_weight)
        replace_parameter(layer, "w13_scale", w13_scale)

        w2_weight, w2_scale = repack_weights(
            layer.w2_weight, layer.w2_scale, weight_bits
        )
        replace_parameter(layer, "w2_weight", w2_weight)
        replace_parameter(layer, "w2_scale", w2_scale)

        init_workspace(layer.w13_weight.device)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, _ = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        return fused_marlin_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
            layer.w13_scale,
            layer.w2_scale,
            router_logits,
            topk_weights,
            topk_ids,
            quant_type_id=self.quant_config.quant_type.id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            workspace=workspace,
        )


def rtn_quantize(
    tensor: torch.Tensor, num_bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor using per-group static scaling factor.

    Args:
        tensor: The input tensor.
        num_bits: Target precision for the result (supported values are
                  8 or 4).
        group_size: Quantization granularity.
                    If equal to -1, each row in the input tensor is treated
                    as one group.
    """
    batch_present = len(tensor.shape) == 3
    if not batch_present:
        tensor = tensor.unsqueeze(0)

    q_range = 2**num_bits
    num_groups = (
        tensor.shape[1] * tensor.shape[2] // group_size
        if group_size != -1
        else tensor.shape[1]
    )
    """Calculate a scaling factor per input group.
    """
    input_flat = tensor.reshape(tensor.shape[0], num_groups, -1)
    input_min = torch.min(input_flat, dim=2, keepdim=True)[0]
    input_max = torch.max(input_flat, dim=2, keepdim=True)[0]
    input_max_abs = torch.max(input_min.abs(), input_max.abs())
    scale = input_max_abs * 2.0 / (q_range - 1)
    """Scale each input group, round to the nearest integer, shift 
    the range and truncate.
    """
    scaled_input = input_flat / scale
    scaled_input = scaled_input.round()
    scaled_input += q_range // 2
    scaled_input = scaled_input.clamp(0, q_range - 1)

    scale = scale.reshape(tensor.shape[0], tensor.shape[1], -1).contiguous()
    inputs_q = scaled_input.reshape(tensor.shape).to(torch.uint8)
    inputs_q = inputs_q.contiguous()

    if num_bits == 4:
        """Pack two 4-bit values into each byte.
        """
        inputs_q = (inputs_q[:, :, 1::2] << 4) | (inputs_q[:, :, ::2] & 0xF)
        inputs_q = inputs_q.reshape(
            tensor.shape[0], tensor.shape[1] // 2, tensor.shape[2]
        )
        inputs_q = inputs_q.contiguous()

    if not batch_present:
        inputs_q = inputs_q.squeeze(0)
        scale = scale.squeeze(0)

    return inputs_q, scale


def rtn_dequantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a tensor using per-group static scaling factors.

    Args:
        tensor: The input tensor.
        scale: The tensor with per-group scale factors.
    """
    batch_present = len(tensor.shape) == 3
    if not batch_present:
        tensor = tensor.unsqueeze(0)
        scale = scale.unsqueeze(0)

    num_groups = scale.size(1) * scale.size(2)
    batch, input_dim, output_dim = tensor.shape

    num_bits = 8 if input_dim == scale.size(1) else 4
    q_range = 2**num_bits
    if num_bits == 4:
        input_dim *= 2

    data = torch.empty(
        (batch, input_dim, output_dim), dtype=scale.dtype, device=tensor.device
    )

    if num_bits == 8:
        data.copy_(tensor)
        data -= q_range // 2
    else:
        """Unpack two 4-bit values from each byte.
        """
        tensor = tensor.reshape(batch, input_dim, output_dim // 2)
        for i in range(2):
            data[:, :, i::2] = ((tensor << 4 * (1 - i)) >> 4).to(
                torch.int8
            ) - q_range // 2
    """Scale each input group with its scaling factor.
    """
    scale = scale.reshape(batch, num_groups, -1)
    data = data.reshape(batch, num_groups, -1)
    data = torch.mul(data, scale)

    input_deq = data.reshape((batch, input_dim, output_dim)).contiguous()
    if not batch_present:
        input_deq = input_deq.squeeze(0)

    return input_deq


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm_arr = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm_arr = perm_arr.reshape((-1, 8))[:, interleave].ravel()
    perm_tensor = torch.from_numpy(perm_arr)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm_tensor, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


def pack_for_marlin(weight, scale, qbits):
    batch = weight.shape[0]

    n = weight.size(1)
    k = weight.size(2)
    groupsize = k // scale.size(2)

    tile = 16
    s = scale.permute(0, 2, 1)  # transpose
    w = weight.permute(0, 2, 1)  # transpose
    if groupsize != k:
        w = w.reshape((batch, -1, groupsize, n))
        w = w.permute(0, 2, 1, 3)
        w = w.reshape((batch, groupsize, -1))
        s = s.reshape((batch, 1, -1))

    if groupsize != k:
        w = w.reshape((batch, groupsize, -1, n))
        w = w.permute(0, 2, 1, 3)
        w = w.reshape((batch, k, n)).contiguous()
        s = s.reshape((batch, -1, len(_scale_perm)))[:, :, _scale_perm]
    else:
        s = s.reshape((batch, -1, len(_scale_perm_single)))[:, :, _scale_perm_single]
    s = s.reshape((batch, -1, n)).contiguous()
    w = w.reshape((batch, k // tile, tile, n // tile, tile))
    w = w.permute((0, 1, 3, 2, 4))
    w = w.reshape((batch, k // tile, n * tile))
    res = w
    res = res.reshape((batch, -1, _perm.numel()))[:, :, _perm].reshape(res.shape)
    if qbits == 4:
        q = torch.zeros(
            (batch, res.shape[1], res.shape[2] // 2), dtype=torch.int8, device=w.device
        )
        for i in range(2):
            q |= res[:, :, i::2] << 4 * i
        q = q.reshape(batch, -1, n).contiguous()
    else:
        q = res.clone()
        q[:, :, 2::8] = res[:, :, 4::8]
        q[:, :, 3::8] = res[:, :, 5::8]
        q[:, :, 4::8] = res[:, :, 2::8]
        q[:, :, 5::8] = res[:, :, 3::8]
        q = q.reshape(batch, -1, n).to(torch.int8).contiguous()

    return q, s


def repack_8bit_into_32bit(input):
    output = torch.zeros(
        (input.shape[0], input.shape[1], input.shape[2] // 4),
        dtype=torch.int32,
        device=input.device,
    )
    for i in range(4):
        output |= (input[:, :, i::4] & 0xFF).to(torch.int32) << 8 * i

    return output


def repack_weights(qweight, scale, weight_bits):
    batch_present = len(qweight.shape) == 3
    if not batch_present:
        qweight = qweight.unsqueeze(0)
        scale = scale.unsqueeze(0)

    if weight_bits == 4:
        """Unpack two 4-bit values from each byte.
        """
        qweight_unpacked = torch.empty(
            (qweight.shape[0], qweight.shape[1] * 2, qweight.shape[2]),
            dtype=torch.uint8,
            device=qweight.device,
        )
        for i in range(2):
            qweight_unpacked[:, :, i::2] = ((qweight << 4 * (1 - i)) >> 4).reshape(
                qweight.shape[0], qweight.shape[1] * 2, qweight.shape[2] // 2
            )
    else:
        qweight_unpacked = qweight

    qweight_packed, scale_packed = pack_for_marlin(qweight_unpacked, scale, weight_bits)
    """Marlin kernels expect tensors in int32 format in a certain shape
    """
    qweight_repacked = repack_8bit_into_32bit(qweight_packed.to(torch.uint8))
    qweight_reshaped = qweight_repacked.reshape(
        qweight.shape[0], qweight.shape[2] // 16, -1
    )
    if not batch_present:
        qweight_reshaped = qweight_reshaped.squeeze(0)
        scale_packed = scale_packed.squeeze(0)

    return qweight_reshaped, scale_packed


def init_workspace(device):
    global workspace
    if workspace is None:
        workspace = marlin_make_workspace_new(device, 4)
