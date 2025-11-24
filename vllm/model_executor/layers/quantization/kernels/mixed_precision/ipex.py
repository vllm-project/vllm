# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class IPEXwNA16LinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # TODO: add zero point support in the future
        if c.zero_points:
            return False, "Zero points not supported for Now"

        if not (current_platform.is_xpu() or current_platform.is_cpu()):
            return False, "IPEX wNa16 only supported on XPU/CPU devices"
        return True, None

        if not current_platform.is_device_capability(90):
            return False, "CUTLASS W4A8 requires compute capability of 90 (Hopper)"

        if c.act_type != torch.float8_e4m3fn:
            return False, "CUTLASS W4A8 only supports FP8 (e4m3) activations"

        if c.has_g_idx:
            return False, "Act reordering not supported by CUTLASS W4A8"

        if c.weight_type != scalar_types.int4:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "CUTLASS W4A8, only supported int4",
            )

        # TODO(czhu): support -1 (column-wise)
        if c.group_size != 128:
            return False, "Only group_size 128 is supported"

        in_features, out_features = c.partition_weight_shape
        if in_features % 128 or out_features % 128:
            return (
                False,
                f"K and N must be divisible by 128, got {c.partition_weight_shape}",
            )

        if c.out_type != torch.bfloat16:
            return (
                False,
                f"Only bfloat16 output type currently supportedgot {c.out_type=}",
            )

        return True, None


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from packaging import version
        MIN_IPEX_VERSION = "2.6.0"
        bias = layer.bias if not layer.skip_bias_add else None

        try:
            import intel_extension_for_pytorch as ipex

            if version.parse(ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    "intel_extension_for_pytorch version is "
                    "wrong. Please install "
                    f"intel_extension_for_pytorch>={MIN_IPEX_VERSION}."
                )
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method."
            ) from err
        # Using the compute dtype (lowp_mode) as INT8 to leverage instructions
        # with better performance.
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        # The weight will be de-packed from INT4 to INT8.
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        # The float activation will be quantized (dynamic, per-token) to INT8.
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH
        # act_quant_mode = ipex.quantization.WoqActQuantMode.NONE

        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size= self.config.group_size,
            weight_qscheme=ipex.quantization.WoqWeightQScheme.SYMMETRIC
        )
        qweight = layer.weight_packed
        layer.ipex_output_size = qweight.shape[-1]
        g_idx = layer.weight_g_idx if self.config.has_g_idx else None
        scales = layer.weight_scale
        qzeros = layer.weight_zero_point if self.config.zero_points else None
        ipex_output_size = qweight.shape[0]
        qweight = qweight.t()
        scales = scales.t()
        ipex_in_size = qweight.size(0)
        layer.ipex_output_size = ipex_output_size
        layer.ipex_qlinear = (
            ipex.llm.quantization.woq_linear.IPEXWeightOnlyQuantizedLinear.from_weight(
                qweight,
                scales,
                qzeros,
                in_features=ipex_in_size,
                out_features=ipex_output_size,
                qconfig=qconfig,
                g_idx=g_idx,
                bias=bias,
                group_size= self.config.group_size,
                quant_method=0, # `0` stands for the IPEX GPTQ
            )
        )
        

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size,))
