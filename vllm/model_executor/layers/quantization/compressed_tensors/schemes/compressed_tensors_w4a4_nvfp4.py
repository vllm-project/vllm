# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
    run_nvfp4_emulations,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported,
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.backend = "none"
        if envs.VLLM_NVFP4_GEMM_BACKEND is None:
            if has_flashinfer():
                self.backend = "flashinfer-cutlass"
            elif cutlass_fp4_supported():
                self.backend = "cutlass"
        elif envs.VLLM_USE_FBGEMM:
            self.backend = "fbgemm"
            try:
                import fbgemm_gpu  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Backend fbgemm requires fbgemm.f4f4bf16 operator, "
                    "Please install with: pip install fbgemm-gpu-genai"
                ) from exc
        elif envs.VLLM_NVFP4_GEMM_BACKEND.startswith("flashinfer-"):
            self.backend = envs.VLLM_NVFP4_GEMM_BACKEND
            assert has_flashinfer(), f"FlashInfer is required for {self.backend}"
        elif envs.VLLM_NVFP4_GEMM_BACKEND == "cutlass":
            self.backend = "cutlass"
            assert cutlass_fp4_supported(), f"Cutlass is required for {self.backend}"

        if self.backend == "none":
            raise ValueError(
                "No valid NVFP4 GEMM backend found. "
                "Please check your platform capability."
            )

        logger.info_once(f"Using {self.backend} for NVFP4 GEMM")
        self.group_size = 16

    @classmethod
    def get_min_capability(cls, weight_quant: QuantizationArgs) -> int:
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            return 80
        return 100

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
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        global_input_scale = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        if self.backend == "flashinfer-trtllm":
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            if self.backend == "fbgemm":
                swizzled_weight_scale = swizzled_weight_scale.view(-1).view(torch.uint8)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)

            # Pad weights for CUTLASS/FlashInfer kernel alignment (K and N
            # divisible by 32). fbgemm has its own layout requirements.
            if self.backend in ("cutlass", "flashinfer-cutlass"):
                weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
                    layer.weight_packed.data
                )
                layer.weights_padding_cols = weights_padding_cols
                layer.weight_packed = Parameter(weight, requires_grad=False)
            else:
                layer.weights_padding_cols = 0
                layer.weight_packed = Parameter(
                    layer.weight_packed.data, requires_grad=False
                )

        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            out = run_nvfp4_emulations(
                x=x,
                input_global_scale=layer.input_global_scale,
                weight=layer.weight_packed,
                weight_scale_swizzled=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
            )
            if bias is not None:
                out = out + bias
            return out

        output_dtype = x.dtype
        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale,
            is_sf_swizzled_layout=True,
            backend=self.backend,
        )

        # Pad activations to match weight K-dimension padding
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

        mm_args = (
            x_fp4,
            layer.weight_packed,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
        )
        if self.backend.startswith("flashinfer-"):
            backend_name = self.backend[len("flashinfer-") :]
            out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
        elif self.backend == "fbgemm":
            out = torch.ops.fbgemm.f4f4bf16(
                x_fp4,
                layer.weight_packed,
                x_blockscale.view(-1).view(torch.uint8),
                layer.weight_scale,
                layer.alpha,
                use_mx=False,
            ).to(output_dtype)
        else:
            assert self.backend == "cutlass"
            out = cutlass_scaled_fp4_mm(*mm_args)

        # Slice output to remove N-dimension padding
        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
