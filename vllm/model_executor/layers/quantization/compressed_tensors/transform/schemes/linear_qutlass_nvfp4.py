# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm._custom_ops import fusedQuantizeNv
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsScheme,
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    TransformTuple,
)
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    slice_nvfp4_output,
)
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
)

__all__ = ["is_qutlass_fp4_scheme", "QutlassNvFP4LinearMethod"]

NVFP4_MAX = 6.0


# QUTLASS supports block sizes (16, 32, 64, 128) for NVFP4
# https://github.com/IST-DASLab/qutlass/blob/v0.2.0/qutlass/csrc/bindings.cpp#L413-L414
def is_qutlass_fp4_scheme(
    quant_scheme: CompressedTensorsScheme | None,
    input_tfms: dict[int, TransformTuple],
) -> bool:
    return (
        isinstance(quant_scheme, CompressedTensorsW4A4Fp4)
        and len(input_tfms) >= 1
        and all(
            input_tfm.scheme.head_dim in (16, 32, 64, 128)
            for input_tfm in input_tfms.values()
        )
    )


class QutlassNvFP4LinearMethod(CompressedTensorsLinearTransformMethod):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        # initializes fp4 qparams
        assert isinstance(layer.scheme, (CompressedTensorsW4A4Fp4,))
        ret = super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        assert self.input_transform is not None
        assert len(self.input_transform.weight.partitions) >= 1

        return ret

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)

        assert self.input_transform is not None
        h = self.input_transform.weight.partitions[0].data
        h_normalized = (h * self.input_transform.scales[0]).to(torch.bfloat16)
        layer.hadamard_matrix = Parameter(h_normalized, requires_grad=False)

        # fusedQuantizeNv stores raw absmax as block scales (sf = absmax),
        # while CT weights use sf = absmax * SFScaleVal / 6.0. The GEMM
        # computes alpha * sum(fp4_a * sf_a * fp4_w * sf_w), so alpha must
        # compensate: alpha = weight_global_scale / 6.0
        layer.fused_alpha = Parameter(
            layer.weight_global_scale / NVFP4_MAX, requires_grad=False
        )

        layer.fused_global_scale = Parameter(
            torch.tensor([NVFP4_MAX], dtype=torch.float32, device=h.device),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None
        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]

        x_flat = x.contiguous().flatten(end_dim=-2)

        x_fp4, x_scales = fusedQuantizeNv(
            x_flat, layer.hadamard_matrix, layer.fused_global_scale
        )

        x_scales_blocked = to_blocked(x_scales, backend="triton").view(x_scales.shape)

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_scales_blocked,
            layer.weight_scale,
            layer.fused_alpha,
            x.dtype,
            backend="cutlass",
        )

        out = slice_nvfp4_output(out, output_size)

        if self.output_transform is not None:
            for part_id, (start, length) in enumerate(self.partition_ranges):
                out[:, start : start + length] = self.output_transform(
                    out[:, start : start + length].clone(), part_id=part_id
                )

        return out.view(*output_shape)
