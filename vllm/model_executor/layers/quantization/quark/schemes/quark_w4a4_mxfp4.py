# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    OCP_MX_BLOCK_SIZE, dequant_mxfp4, quant_dequant_mxfp4)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform


@cache
def is_rocm_aiter_fp4_asm_gemm_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM \
        and envs.VLLM_ROCM_USE_AITER


try:
    import triton
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
    from aiter.ops.triton.fused_mxfp4_quant import _fused_rms_mxfp4_quant_kernel

    from vllm.utils import direct_register_custom_op
    if is_rocm_aiter_fp4_asm_gemm_enabled():
        from aiter import gemm_a4w4, per_1x32_f4_quant_hip

    def gemm_with_dynamic_quant(
        result: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scales: Optional[torch.Tensor] = None,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        if rocm_use_aiter_fp4_asm_gemm:
            M = x.shape[0]
            if x_scales is None:
                # use hip quant kernel for performance
                x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
            else:
                x_q = x
                x_s = x_scales

            # 32 alignment is enough for dim0 padding of output for
            # gemm_a4w4 kernel
            y = torch.empty((M + 31) // 32 * 32,
                            weight.shape[0],
                            device=x_q.device,
                            dtype=out_dtype)

            gemm_a4w4(x_q,
                      weight,
                      x_s,
                      weight_scale.view(x_s.dtype),
                      y,
                      bpreshuffle=True)
            result.copy_(y[:M])
        else:
            if x_scales is None:
                x_q, x_s = dynamic_mxfp4_quant(x)
            else:
                x_q = x
                x_s = x_scales
            gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, out_dtype, result)

    def gemm_with_dynamic_quant_fake(
        result: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scales: torch.Tensor = None,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> torch.Tensor:
        return

    direct_register_custom_op(
        op_name="gemm_with_dynamic_quant",
        op_func=gemm_with_dynamic_quant,
        mutates_args=['result'],
        fake_impl=gemm_with_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    def silu_and_mul_mxfp4_gemm(
        result: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: Optional[torch.dtype] = torch.bfloat16
    ) -> None:
        x_fp4, blockscale_e8m0 = act_mul_and_mxfp4_quant(x, 'silu')
        gemm_with_dynamic_quant(result, x_fp4, weight, weight_scale, blockscale_e8m0, rocm_use_aiter_fp4_asm_gemm, out_dtype)

    def silu_and_mul_mxfp4_gemm_fake(
        result: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: Optional[torch.dtype] = torch.bfloat16
    ) -> None:
        return

    direct_register_custom_op(
        op_name="silu_and_mul_mxfp4_gemm",
        op_func=silu_and_mul_mxfp4_gemm,
        mutates_args=['result'],
        fake_impl=silu_and_mul_mxfp4_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    def add_rmsnorm_mxfp4_gemm(
        result: torch.Tensor, input: torch.Tensor, residual_out: torch.Tensor,
        residual: torch.Tensor, weight_rms: torch.Tensor, 
        weight_gemm: torch.Tensor, scale: torch.Tensor, epsilon: float,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: Optional[torch.dtype] = torch.bfloat16
    ) -> None:
        MXFP4_QUANT_BLOCK_SIZE = 32
        M, N1 = input.shape
        BLOCK_SIZE = max(triton.next_power_of_2(N1), MXFP4_QUANT_BLOCK_SIZE)
        BLOCK_SIZE = max(BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE)
        res_row_stride = residual.stride(0)
        out_res_row_stride = residual_out.stride(0)
        rms_out_fp4 = torch.empty((M, N1 // 2), dtype=torch.uint8, device=input.device)
        rms_out_bs = torch.empty(
            ((N1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
            dtype=torch.uint8,
            device=input.device,
        ).T
        _fused_rms_mxfp4_quant_kernel[(M,)](
            input,
            weight_rms,
            None,
            None,
            residual,
            rms_out_fp4,
            rms_out_bs,
            None,
            residual_out,
            epsilon,
            0.0,
            M,
            N1,
            0,
            input.stride(0),
            0,
            res_row_stride,
            rms_out_fp4.stride(0),
            *rms_out_bs.stride(),
            0,
            out_res_row_stride,
            BLOCK_SIZE=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
            SKIP_SECOND_INPUT=True,
            FIRST_INPUT_RES=True,
        )
        gemm_with_dynamic_quant(result, rms_out_fp4, weight_gemm, scale, rms_out_bs, rocm_use_aiter_fp4_asm_gemm, out_dtype)

    def add_rmsnorm_mxfp4_gemm_fake(
        result: torch.Tensor, input: torch.Tensor, residual_out: torch.Tensor,
        residual: torch.Tensor, weight_rms: torch.Tensor, 
        weight_gemm: torch.Tensor, scale: torch.Tensor, epsilon: float,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: Optional[torch.dtype] = torch.bfloat16
    ) -> None:
        return

    direct_register_custom_op(
        op_name="add_rmsnorm_mxfp4_gemm",
        op_func=add_rmsnorm_mxfp4_gemm,
        mutates_args=['result', 'residual_out'],
        fake_impl=add_rmsnorm_mxfp4_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )

except ImportError:
    dynamic_mxfp4_quant = gemm_afp4wfp4 = None

__all__ = ["QuarkW4A4MXFP4"]


class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(self, weight_quant_spec: dict[str, Any],
                 input_quant_spec: dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.emulate = not current_platform.supports_mx()
        self.rocm_use_aiter_fp4_asm_gemm = is_rocm_aiter_fp4_asm_gemm_enabled()
        if not self.emulate and (dynamic_mxfp4_quant is None
                                 or gemm_afp4wfp4 is None):
            # Currently need these kernels if not emulating
            raise NotImplementedError(
                f"{self.__class__.__name__} requires AITER to be installed "
                "for non-emulation mode! Please refer to "
                "https://github.com/ROCm/aiter for installation details.")

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)

        if self.emulate:
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data,
                                                    requires_grad=False)
            try:
                from quark.torch.export.nn.modules import realquantizer
                from quark.torch.quantization.config.config import (
                    QuantizationSpec)
            except ImportError as err:
                raise ImportError(
                    "The package `amd-quark` is required to use AMD Quark "
                    "MX-FP4 models. Please install it with `pip install "
                    "amd-quark`.") from err

            weight_quant_spec = QuantizationSpec.from_dict(
                self.weight_quant_spec)

            weight_quantizer = realquantizer.get_real_quantizer(
                qspec=weight_quant_spec,
                quantizer=None,
                real_quantized=True,
                reorder=False,
                float_dtype=self.out_dtype,
                scale_shape=layer.weight_scale.shape,
                zero_point_shape=None,
            )
            weight_quantizer.scale.data = layer.weight_scale.data

            layer.weight = torch.nn.Parameter(
                weight_quantizer(layer.weight.data).to(self.out_dtype),
                requires_grad=False,
            )
            layer.weight_scale = None

            # This call is necessary to release the scales memory.
            torch.cuda.empty_cache()
        else:
            if self.rocm_use_aiter_fp4_asm_gemm:
                # shuffle weight scale
                weight_scale_shuffle = layer.weight_scale.data
                sm, sn = weight_scale_shuffle.shape
                weight_scale_shuffle = weight_scale_shuffle.view(
                    sm // 32, 2, 16, sn // 8, 2, 4, 1)
                weight_scale_shuffle = weight_scale_shuffle.permute(
                    0, 3, 5, 2, 4, 1, 6).contiguous()
                weight_scale_shuffle = weight_scale_shuffle.view(sm, sn)
                layer.weight_scale = torch.nn.Parameter(weight_scale_shuffle,
                                                        requires_grad=False)

                # shuffle weight
                weight_shuffle = layer.weight.data
                weight_shuffle = shuffle_weight(weight_shuffle,
                                                layout=(16, 16))
                layer.weight = torch.nn.Parameter(weight_shuffle,
                                                  requires_grad=False)
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data.T.contiguous(),
                    requires_grad=False)

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.emulate:
            dq_w = dequant_mxfp4(layer.weight, layer.weight_scale, x.dtype)
            x = quant_dequant_mxfp4(x)
            return F.linear(x, dq_w, bias)
        else:
            result = torch.empty((*x.shape[:-1], layer.weight.shape[0]), dtype=self.out_dtype, device=x.device)
            torch.ops.vllm.gemm_with_dynamic_quant(
                result, x, layer.weight, layer.weight_scale, None, self.rocm_use_aiter_fp4_asm_gemm, self.out_dtype)
            return result
