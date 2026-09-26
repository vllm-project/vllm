# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental CuTe readers sharing one canonical NVFP4 representation."""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


def cutedsl_dynamic_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    max_a16_tokens: int,
    fuse_silu: bool,
) -> torch.Tensor:
    import flashinfer
    from flashinfer.quantization.fp4_quantization import silu_and_mul_nvfp4_quantize

    if x.ndim != 2 or x.dtype != torch.bfloat16 or not 0 <= max_a16_tokens <= 16:
        raise ValueError(
            "Dynamic CuTe NVFP4 requires BF16 matrices and a cutoff of 0..16"
        )
    x = x.contiguous()
    if 0 < x.shape[0] <= max_a16_tokens:
        if fuse_silu:
            from vllm.model_executor.kernels.linear.nvfp4.cutedsl_silu import run

            x = run(x, block=256, vector=1, enable_pdl=True)
        return flashinfer.mm_bf16_fp4(
            x,
            weight,
            weight_scale,
            weight_global_scale.reshape(1),
            backend="cute-dsl-native",
            enable_pdl=True,
        )

    if fuse_silu:
        x_fp4, x_scale = silu_and_mul_nvfp4_quantize(
            x, input_global_scale_inv.reshape(1), enable_pdl=True
        )
    else:
        x_fp4, x_scale = flashinfer.nvfp4_quantize(
            x,
            input_global_scale_inv.reshape(1),
            sfLayout=flashinfer.SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cute-dsl",
            enable_pdl=True,
        )
    return flashinfer.mm_fp4(
        x_fp4,
        weight.T,
        x_scale.view(torch.uint8),
        weight_scale.view(torch.uint8).T,
        alpha.reshape(1),
        out_dtype=torch.bfloat16,
        backend="cute-dsl",
        use_nvfp4=True,
        use_8x4_sf_layout=False,
        enable_pdl=True,
    )


def _cutedsl_dynamic_nvfp4_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    max_a16_tokens: int,
    fuse_silu: bool,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    "cutedsl_dynamic_nvfp4",
    cutedsl_dynamic_nvfp4,
    fake_impl=_cutedsl_dynamic_nvfp4_fake,
    dispatch_key="CUDA",
)


class FlashInferCuTeDynamicNvFp4LinearKernel(NvFp4LinearKernel):
    """Select activation precision before quantization, without repacking weights."""

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        runtime = get_current_vllm_config()
        if runtime.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Experimental dynamic CuTe NVFP4 currently requires TP=1")
        if runtime.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "Experimental dynamic CuTe NVFP4 requires BF16 activations"
            )
        self.max_a16_tokens = runtime.kernel_config.nvfp4_dynamic_max_tokens
        silu_cutoff = runtime.kernel_config.nvfp4_dynamic_silu_max_tokens
        self.max_silu_a16_tokens = (
            self.max_a16_tokens if silu_cutoff is None else silu_cutoff
        )

    @classmethod
    def is_supported(cls, compute_capability=None):
        if compute_capability is None:
            if not current_platform.is_cuda():
                return False, "Dynamic CuTe NVFP4 requires CUDA"
            cc = current_platform.get_device_capability()
            compute_capability = cc.to_int() if cc is not None else None
        if compute_capability not in (120, 121):
            return False, "Dynamic CuTe NVFP4 currently requires SM120 or SM121"
        try:
            from flashinfer import mm_bf16_fp4, mm_fp4

            available = mm_fp4.is_backend_supported("cute-dsl", compute_capability)
            available &= mm_bf16_fp4.is_backend_supported(
                "cute-dsl-native", compute_capability
            )
        except (ImportError, AttributeError, KeyError, ValueError):
            available = False
        return (True, None) if available else (False, "Both CuTe readers are required")

    @classmethod
    def can_implement(cls, config):
        return True, None

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        n, packed_k = weight.shape
        k = packed_k * 2
        if weight.dtype != torch.uint8 or n % 128 or k % 256:
            raise ValueError("Dynamic CuTe NVFP4 requires uint8 weights, N%128=K%256=0")
        if tuple(layer.weight_scale.shape) != (n, k // 16):
            raise ValueError("Expected unprepared block-16 weight scales")
        if not all(
            hasattr(layer, name)
            for name in ("input_global_scale_inv", "weight_global_scale", "alpha")
        ):
            raise ValueError("Dynamic CuTe NVFP4 requires calibrated W4A4 scales")
        layer.weight = torch.nn.Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )

    def apply_weights(self, layer, x, bias=None):
        shape = (*x.shape[:-1], layer.output_size_per_partition)
        out = self.apply_silu_or_linear(layer, x.reshape(-1, x.shape[-1]), False)
        if bias is not None:
            out = out + bias
        return out.view(shape)

    def apply_silu_or_linear(self, layer, x, fuse_silu):
        return torch.ops.vllm.cutedsl_dynamic_nvfp4(
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.input_global_scale_inv,
            layer.alpha,
            self.max_silu_a16_tokens if fuse_silu else self.max_a16_tokens,
            fuse_silu,
        )
