# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from .inc_wna16_linear import INCXPULinearMethod


class INCXPUW4A8LinearMethod(INCXPULinearMethod):
    """XPU linear method for INC int4 weights with dynamic int8 activations.

    Uses the same GPTQ-packed "NT" qweight layout as ``INCXPULinearMethod`` —
    ``int4_gemm_w4a8`` and ``int4_gemm_w4a16`` accept bit-identical weights, so
    no extra repacking is needed. Activations are dynamically quantized
    per-token to symmetric int8, which keeps the GEMM on the int8 datapath
    instead of upconverting the weights to the activation dtype. That is a win
    for compute-bound shapes (large token counts, e.g. diffusion), where w4a16
    is dominated by dequantization rather than by weight bandwidth.

    Both scale tensors must be float16: the kernel reads them as fp16
    regardless of the activation dtype, so bf16 scales are silently
    misinterpreted. The activations themselves may be bf16 or fp16.

    Opt in with ``VLLM_XPU_INC_WNA16_BACKEND=w4a8``. Whether this beats ARK is
    device-dependent, so it is not the default: it was measured faster on B70
    (Xe2), where ARK cannot use its XMX int8 path and falls back to fp16.
    """

    # int8 activations only pay off once the GEMM is compute-bound. Below this
    # token count the per-token quantization overhead dominates and w4a16 is
    # faster, so fall back to it per-call.
    _MIN_TOKENS_FOR_INT8 = 512

    # int4_gemm_w4a8 requires both GEMM dims to be multiples of 8. Tensor
    # parallelism shards these, so an aligned model dimension can still yield an
    # unaligned partition.
    _DIM_ALIGNMENT = 8

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        unaligned = [
            (name, size)
            for name, size in (
                ("input", input_size_per_partition),
                ("output", output_size_per_partition),
            )
            if size % self._DIM_ALIGNMENT != 0
        ]
        if unaligned:
            raise NotImplementedError(
                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires partitioned in/out "
                f"sizes that are multiples of {self._DIM_ALIGNMENT}, got "
                + ", ".join(f"{name}={size}" for name, size in unaligned)
                + f". Partition shape: ({input_size_per_partition}, "
                f"{output_size_per_partition})."
            )
        super().create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # The kernel reads both scale tensors as fp16; keep an fp16 copy for the
        # w4a8 path and leave ``scales`` untouched for the w4a16 fallback.
        layer.scales_fp16 = Parameter(
            layer.scales.data.to(torch.float16), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        if reshaped_x.shape[0] < self._MIN_TOKENS_FOR_INT8:
            return super().apply_weights(layer, x, bias)

        from vllm._xpu_ops import xpu_ops as ops

        out_shape = x.shape[:-1] + (layer.qweight.shape[1],)
        quant_x, x_scale, x_zero = ops.dynamic_per_token_int8_quant_ref(
            reshaped_x, True, 8
        )
        out = torch.ops._xpu_C.int4_gemm_w4a8(
            quant_x,
            x_scale.to(torch.float16),
            x_zero,
            layer.qweight,
            layer.scales_fp16,
            layer.qzeros,
            self.group_size,
            None,
            bias,
        )
        return out.to(x.dtype).reshape(out_shape)
