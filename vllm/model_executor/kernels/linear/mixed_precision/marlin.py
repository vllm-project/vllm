# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import dataclasses
from fractions import Fraction
from typing import cast

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MIN_THREAD_N,
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_act_int8_process_scales,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_sort_g_idx,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.math_utils import round_up

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)


def _pad_tensor_dim(x: torch.Tensor, dim: int, pad: int) -> torch.Tensor:
    if pad == 0:
        return x

    dim = dim if dim >= 0 else x.dim() + dim
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)


def _get_param_output_dim_padding(
    param: BasevLLMParameter,
    output_dim_pad: int,
) -> tuple[int, int]:
    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        raise ValueError(
            "Marlin output-dim padding requires vLLM parameter output_dim metadata."
        )

    pad = output_dim_pad
    if getattr(param, "packed_dim", None) == output_dim:
        packed_factor = getattr(param, "packed_factor", None)
        if packed_factor is None:
            raise ValueError(
                "Marlin packed output-dim padding requires packed_factor metadata."
            )

        packed_pad = Fraction(output_dim_pad, 1) / Fraction(packed_factor)
        if packed_pad.denominator != 1:
            raise ValueError(
                "Marlin output padding is not divisible by packed_factor: "
                f"pad={output_dim_pad}, packed_factor={packed_factor}."
            )
        pad = packed_pad.numerator

    return output_dim, pad


def _pad_parameter_output_dim(
    param: BasevLLMParameter,
    output_dim_pad: int,
) -> None:
    output_dim, pad = _get_param_output_dim_padding(param, output_dim_pad)
    param.data = _pad_tensor_dim(param.data, output_dim, pad)


class MarlinLinearKernel(MPLinearKernel):
    config: MPLinearLayerConfig
    w_q_name: str
    w_s_name: str
    w_zp_name: str | None
    w_gidx_name: str | None
    orig_output_size_per_partition: int

    def __init__(
        self,
        c: MPLinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
        w_zp_param_name: str | None = None,
        w_gidx_param_name: str | None = None,
    ) -> None:
        super().__init__(
            c,
            w_q_param_name,
            w_s_param_name,
            w_zp_param_name,
            w_gidx_param_name,
        )
        self.orig_output_size_per_partition = self.config.partition_weight_shape[1]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        # Pad sub-tile output dims at load time; see _maybe_pad_n().
        padded_n = round_up(c.partition_weight_shape[1], GPTQ_MARLIN_MIN_THREAD_N)
        return check_marlin_supports_shape(
            padded_n,  # out_features (possibly padded up to tile multiple)
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

    def _maybe_pad_n(self, layer: torch.nn.Module) -> None:
        """Pad output dim to a Marlin tile multiple when needed."""
        c = self.config
        orig_n = c.partition_weight_shape[1]
        padded_n = round_up(orig_n, GPTQ_MARLIN_MIN_THREAD_N)
        self.orig_output_size_per_partition = orig_n
        if padded_n == orig_n:
            return

        pad = padded_n - orig_n

        q = cast(BasevLLMParameter, getattr(layer, self.w_q_name))
        _pad_parameter_output_dim(q, pad)

        s = cast(BasevLLMParameter, getattr(layer, self.w_s_name))
        _pad_parameter_output_dim(s, pad)

        if c.zero_points and self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                _pad_parameter_output_dim(cast(BasevLLMParameter, zp), pad)

        self.config = dataclasses.replace(
            c,
            partition_weight_shape=(c.partition_weight_shape[0], padded_n),
        )
        logger.info_once(
            "Marlin: padded output dim %d -> %d to satisfy tile_n=%d",
            orig_n,
            padded_n,
            GPTQ_MARLIN_MIN_THREAD_N,
        )

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._maybe_pad_n(layer)

        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        is_output_padded = (
            self.orig_output_size_per_partition != self.config.partition_weight_shape[1]
        )
        if hasattr(layer, "bias") and layer.bias is not None and not is_output_padded:
            layer.bias.data = marlin_permute_bias(layer.bias)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin

        padded_n = c.partition_weight_shape[1]
        orig_n = self.orig_output_size_per_partition

        if bias is not None and orig_n != padded_n:
            if bias.shape[-1] == orig_n:
                bias = _pad_tensor_dim(bias, -1, padded_n - orig_n)
                bias = marlin_permute_bias(bias)
            elif bias.shape[-1] == padded_n:
                bias = marlin_permute_bias(bias)
            else:
                raise ValueError(
                    "Marlin bias shape does not match original or padded output dim: "
                    f"bias={bias.shape[-1]}, orig_n={orig_n}, padded_n={padded_n}."
                )

        out = apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=padded_n,
            is_k_full=self.is_k_full,
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )

        if orig_n != padded_n:
            out = out[..., :orig_n].contiguous()
        return out
