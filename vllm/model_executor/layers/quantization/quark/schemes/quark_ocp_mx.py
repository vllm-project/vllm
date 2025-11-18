# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction
from functools import cache, partial
from typing import Any

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    dequant_mxfp6,
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)


# TODO: move registration of custom op to aiter_ops.py
# `from vllm._aiter_ops import rocm_aiter_ops`
# use `rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()`
# for envs checks which does not require @cache anymore.
# triton kernel is torch compile compatible.
# does not require direct registeration.
# use `rocm_aiter_ops.triton_fp4_gemm_dynamic_qaunt`.
@cache
def is_rocm_aiter_fp4_asm_gemm_enabled() -> bool:
    return (
        current_platform.is_rocm()
        and envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM
        and envs.VLLM_ROCM_USE_AITER
    )


try:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.utils.torch_utils import direct_register_custom_op

    if is_rocm_aiter_fp4_asm_gemm_enabled():
        from aiter import gemm_a4w4, per_1x32_f4_quant_hip

    def gemm_with_dynamic_quant(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
        x_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M = x.shape[0]
        if rocm_use_aiter_fp4_asm_gemm:
            if x_scales is None:
                # use hip quant kernel for performance
                x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
            else:
                x_q = x
                x_s = x_scales

            # 32 alignment is enough for dim0 padding of output for
            # gemm_a4w4 kernel
            y = torch.empty(
                (M + 31) // 32 * 32, weight.shape[0], device=x_q.device, dtype=out_dtype
            )

            gemm_a4w4(
                x_q, weight, x_s, weight_scale.view(x_s.dtype), y, bpreshuffle=True
            )
            return y[:M]
        else:
            if x_scales is None:
                x_q, x_s = dynamic_mxfp4_quant(x)
            else:
                x_q = x
                x_s = x_scales
            y = torch.empty(
                x_q.shape[0], weight.shape[0], device=x_q.device, dtype=out_dtype
            )

            gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, out_dtype, y)
            return y

    def gemm_with_dynamic_quant_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scales: torch.Tensor = None,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
    ) -> torch.Tensor:
        return torch.empty(
            (*x.shape[:-1], weight.shape[0]), dtype=out_dtype, device=x.device
        )

    direct_register_custom_op(
        op_name="gemm_with_dynamic_quant",
        op_func=gemm_with_dynamic_quant,
        mutates_args=[],
        fake_impl=gemm_with_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )
except (ImportError, AttributeError):
    dynamic_mxfp4_quant = gemm_afp4wfp4 = None


class QuarkOCP_MX(QuarkScheme):
    def __init__(
        self, weight_quant_spec: dict[str, Any], input_quant_spec: dict[str, Any]
    ):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec

        self.weight_dtype = weight_quant_spec["dtype"].replace("fp", "mxfp")
        self.input_dtype = input_quant_spec["dtype"].replace("fp", "mxfp")

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.weight_dtype == "mxfp4":
            self.packed_factor: int | Fraction = 2
            self.dequant_func = dequant_mxfp4
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)
            self.dequant_func = partial(
                dequant_mxfp6, quant_dtype=self.weight_dtype.replace("mx", "")
            )

        if self.input_dtype == "mxfp4":
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = partial(
                quant_dequant_mxfp6, quant_dtype=self.input_dtype.replace("mx", "")
            )

        self.static_input_scales = not input_quant_spec.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX with static input scales is currently not "
                "implemented. Please open an issue."
            )

        # TODO: integrate (or test) mixed-precision kernel.
        self.emulate = not current_platform.supports_mx() or (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        )

        self.rocm_use_aiter_fp4_asm_gemm = is_rocm_aiter_fp4_asm_gemm_enabled()

        if not self.emulate and (dynamic_mxfp4_quant is None or gemm_afp4wfp4 is None):
            # Currently need these kernels if not emulating
            raise NotImplementedError(
                f"{self.__class__.__name__} requires AITER to be installed "
                "for non-emulation mode! Please refer to "
                "https://github.com/ROCm/aiter for installation details."
            )

        if not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

        if current_platform.supports_mx() and (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                f"computation, but kernels for input_dtype={self.input_dtype} "
                f"and weight_dtype={self.weight_dtype} are not yet integrated "
                "in vLLM. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"mxfp6_e3m2", "mxfp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        if self.emulate:
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )
        else:
            if self.rocm_use_aiter_fp4_asm_gemm:
                # shuffle weight scale
                weight_scale_shuffle = layer.weight_scale.data
                sm, sn = weight_scale_shuffle.shape
                weight_scale_shuffle = weight_scale_shuffle.view(
                    sm // 32, 2, 16, sn // 8, 2, 4, 1
                )
                weight_scale_shuffle = weight_scale_shuffle.permute(
                    0, 3, 5, 2, 4, 1, 6
                ).contiguous()
                weight_scale_shuffle = weight_scale_shuffle.view(sm, sn)
                layer.weight_scale = torch.nn.Parameter(
                    weight_scale_shuffle, requires_grad=False
                )

                # shuffle weight
                weight_shuffle = layer.weight.data
                weight_shuffle = shuffle_weight(weight_shuffle, layout=(16, 16))
                layer.weight = torch.nn.Parameter(weight_shuffle, requires_grad=False)
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data.T.contiguous(), requires_grad=False
                )

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                self.get_packed_dim(input_size_per_partition, self.weight_dtype),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.packed_factor,
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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.emulate:
            dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
            qdq_x = self.quant_dequant_func(x)
            return F.linear(qdq_x, dq_w, bias)
        else:
            return torch.ops.vllm.gemm_with_dynamic_quant(
                x,
                layer.weight,
                layer.weight_scale,
                self.rocm_use_aiter_fp4_asm_gemm,
                self.out_dtype,
            )
