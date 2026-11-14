# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_prepare_bf16_fp4_weights,
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
    has_flashinfer_b12x_gemm,
    has_flashinfer_bf16_fp4,
    has_flashinfer_sm12x_cutedsl_fp4_gemm,
)
from vllm.utils.torch_utils import direct_register_custom_op, eager_bypass_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

# On SM120, FlashInfer cute-dsl beats FlashInfer CUTLASS up to this many tokens
# and loses above it (RTX PRO 6000, Qwen3.8-27B MLP and lm_head shapes). Its
# SM12x runner only writes BF16 output; other dtypes use CUTLASS.
SM120_CUTEDSL_MAX_TOKENS = 64


def _use_sm120_cutedsl(num_tokens: int, out_dtype: torch.dtype) -> bool:
    return out_dtype == torch.bfloat16 and num_tokens <= SM120_CUTEDSL_MAX_TOKENS


def _sm120_cutedsl_nvfp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    # Both backends take the same swizzled + padded operands. The token count
    # is only concrete inside the op, so the choice is made here.
    backend = "cute-dsl" if _use_sm120_cutedsl(a.shape[0], out_dtype) else "cutlass"
    return flashinfer_scaled_fp4_mm(
        a, b, block_scale_a, block_scale_b, alpha, out_dtype, backend=backend
    )


def _sm120_cutedsl_nvfp4_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[0]), dtype=out_dtype)


if current_platform.is_cuda():
    direct_register_custom_op(
        op_name="sm120_cutedsl_nvfp4_mm",
        op_func=_sm120_cutedsl_nvfp4_mm,
        fake_impl=_sm120_cutedsl_nvfp4_mm_fake,
    )
    sm120_cutedsl_nvfp4_mm = eager_bypass_custom_op(
        "sm120_cutedsl_nvfp4_mm", _sm120_cutedsl_nvfp4_mm
    )

# Packed weight shapes whose SM120 cute-dsl kernels are already compiled.
_sm120_cutedsl_compiled_shapes: set[tuple[int, ...]] = set()


def _compile_sm120_cutedsl(layer: torch.nn.Module) -> None:
    """Compile the SM120 cute-dsl kernels for every token count they serve.

    cute-dsl compiles a kernel per token-count bucket on first use (0.3-0.5 s
    each). CUDA graph capture covers layers inside the graph, but layers that
    run eagerly, such as lm_head on the sampled tokens, would compile while
    serving requests.
    """
    dtype = torch.get_default_dtype()
    shape = tuple(layer.weight.shape)
    if dtype != torch.bfloat16 or shape in _sm120_cutedsl_compiled_shapes:
        return
    _sm120_cutedsl_compiled_shapes.add(shape)
    k = layer.weight.shape[1] * 2
    for m in range(1, SM120_CUTEDSL_MAX_TOKENS + 1):
        x = torch.zeros(m, k, dtype=dtype, device=layer.weight.device)
        data, scale = scaled_fp4_quant(x, layer.input_global_scale_inv)
        _sm120_cutedsl_nvfp4_mm(
            data, layer.weight, scale, layer.weight_scale, layer.alpha, dtype
        )


class FlashInferCuteDslNvFp4W4A16LinearKernel(NvFp4LinearKernel):
    """BF16 x NVFP4 GEMM via FlashInfer's CuTe-DSL backend."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            if not current_platform.is_cuda():
                return False, "FlashInfer CuTe-DSL W4A16 requires CUDA"
            capability = current_platform.get_device_capability()
            if capability is None:
                return False, "CUDA compute capability is unavailable"
            compute_capability = capability.to_int()

        if compute_capability not in (100, 103) and not (
            compute_capability >= 120 and compute_capability < 130
        ):
            return False, "FlashInfer CuTe-DSL W4A16 requires sm_100 or sm_12x"
        if not has_flashinfer_bf16_fp4():
            return False, "FlashInfer CuTe-DSL BF16 x FP4 GEMM is unavailable"
        return True, None

    @classmethod
    def executed_activation_key(
        cls, activation_quant_key: QuantKey | None
    ) -> QuantKey | None:
        return None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        padded_weight, weights_padding_bytes = pad_nvfp4_weight_for_cutlass(
            layer.weight.data, alignment=64
        )
        swizzled_scale = swizzle_blockscale(layer.weight_scale.data)
        weight, weight_scale, global_scale = flashinfer_prepare_bf16_fp4_weights(
            padded_weight,
            swizzled_scale,
            layer.weight_global_scale.data.reshape(1),
            backend="cute-dsl",
        )
        assert global_scale is not None

        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale", weight_scale)
        replace_parameter(layer, "weight_global_scale", global_scale)
        layer.weights_padding_cols = weights_padding_bytes

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(
                f"FlashInfer CuTe-DSL W4A16 requires BF16 input, got {x.dtype}"
            )

        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]
        x_2d = x.reshape(-1, x.shape[-1])
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)
        if weights_padding_bytes:
            x_2d = torch.nn.functional.pad(x_2d, (0, weights_padding_bytes * 2))
        x_2d = x_2d.contiguous()

        from vllm.utils.flashinfer import flashinfer_mm_bf16_fp4

        out = flashinfer_mm_bf16_fp4(
            x_2d,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
        )
        out = slice_nvfp4_output(out, output_size)
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferCuteDslNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's cutedsl backend."""

    def input_quant_key(self) -> QuantKey | None:
        return kNvfp4Dynamic

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_device_capability(120):
            # Runs cute-dsl for decode-sized inputs and CUTLASS above
            # SM120_CUTEDSL_MAX_TOKENS. SM121 (GB10) is not measured yet.
            if not has_flashinfer_sm12x_cutedsl_fp4_gemm():
                return False, "FlashInfer cutedsl on SM120 requires its SM12x runner"
        elif not current_platform.is_device_capability_family(100):
            return False, "FlashInfer cutedsl requires sm_10x or SM120"
        if not has_flashinfer():
            return False, "FlashInfer required"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # cutedsl uses the same swizzled + padded layout as cutlass.
        replace_parameter(
            layer,
            "weight_scale",
            swizzle_blockscale(layer.weight_scale.data),
        )
        padded_weight, _ = pad_nvfp4_weight_for_cutlass(layer.weight.data)
        replace_parameter(layer, "weight", padded_weight)
        if current_platform.is_device_capability(120):
            _compile_sm120_cutedsl(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        qa = self.prepare_input(layer, x)

        if current_platform.is_device_capability(120):
            out = sm120_cutedsl_nvfp4_mm(
                qa.data,
                layer.weight,
                qa.scale,
                layer.weight_scale,
                layer.alpha,
                qa.orig_dtype,
            )
        else:
            out = flashinfer_scaled_fp4_mm(
                qa.data,
                layer.weight,
                qa.scale,
                layer.weight_scale,
                layer.alpha,
                qa.orig_dtype,
                backend="cute-dsl",
            )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*qa.orig_shape[:-1], output_size)


class FlashInferCutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's CUTLASS wrapper."""

    def input_quant_key(self) -> QuantKey | None:
        """This kernel supports dynamic quantization of the input. By
        convention, pre-quantized blockscales must use the swizzled layout."""
        return kNvfp4Dynamic

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            cutlass_fp4_supported,
        )

        if (
            cutlass_fp4_supported()
            and current_platform.has_device_capability(100)
            and has_flashinfer()
        ):
            return True, None
        return False, "FlashInfer + >=sm_100 required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(
            layer,
            "weight_scale",
            swizzle_blockscale(layer.weight_scale.data),
        )
        padded_weight, _ = pad_nvfp4_weight_for_cutlass(layer.weight.data)
        replace_parameter(layer, "weight", padded_weight)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        qa = self.prepare_input(layer, x)

        out = flashinfer_scaled_fp4_mm(
            qa.data,
            layer.weight,
            qa.scale,
            layer.weight_scale,
            layer.alpha,
            qa.orig_dtype,
            backend="cutlass",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*qa.orig_shape[:-1], output_size)


class FlashInferTrtllmNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # FlashInfer's TRT-LLM FP4 GEMM only supports the SM100 family.
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "FlashInfer TRT-LLM NVFP4 GEMM requires CUDA sm_10x"
        if has_flashinfer():
            return True, None
        return False, "FlashInfer required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        epilogue_tile_m = 128

        replace_parameter(
            layer,
            "weight",
            shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m),
        )
        replace_parameter(
            layer,
            "weight_scale",
            shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
            .reshape(weight_scale.shape)
            .view(torch.float8_e4m3fn),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-trtllm",
        )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="trtllm",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferCudnnNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's cuDNN wrapper."""

    def input_quant_key(self) -> QuantKey:
        return kNvfp4Dynamic

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda() and current_platform.has_device_capability(100)
        ):
            return False, "NVFP4 activation quantization requires CUDA sm_100+"
        if has_flashinfer():
            return True, None
        return False, "FlashInfer required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # cuDNN uses the same swizzled + padded layout as CUTLASS
        replace_parameter(
            layer,
            "weight_scale",
            swizzle_blockscale(layer.weight_scale.data),
        )
        padded_weight, _ = pad_nvfp4_weight_for_cutlass(layer.weight.data)
        replace_parameter(layer, "weight", padded_weight)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        qa = self.prepare_input(layer, x)

        out = flashinfer_scaled_fp4_mm(
            qa.data,
            layer.weight,
            qa.scale,
            layer.weight_scale,
            layer.alpha,
            qa.orig_dtype,
            backend="cudnn",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*qa.orig_shape[:-1], output_size)


class FlashInferB12xNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's b12x CuTe DSL warp-level MMA kernel (SM12x)."""

    def input_quant_key(self) -> QuantKey:
        return kNvfp4Dynamic

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
            and has_flashinfer_b12x_gemm()
        ):
            return True, None
        return (
            False,
            (
                "FlashInfer b12x requires CUDA sm_12x and FlashInfer "
                "with Sm120BlockScaledDenseGemmKernel"
            ),
        )

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(
            layer, "weight_scale", swizzle_blockscale(layer.weight_scale.data)
        )
        padded_weight, _ = pad_nvfp4_weight_for_cutlass(layer.weight.data)
        replace_parameter(layer, "weight", padded_weight)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        qa = self.prepare_input(layer, x)

        out = flashinfer_scaled_fp4_mm(
            qa.data,
            layer.weight,
            qa.scale,
            layer.weight_scale,
            layer.alpha,
            qa.orig_dtype,
            backend="b12x",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*qa.orig_shape[:-1], output_size)
