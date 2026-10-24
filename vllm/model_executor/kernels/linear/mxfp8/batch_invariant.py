# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 visible math with a fixed K32 accumulation order for precision runs."""

import functools

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.batch_invariant import (
    block32_scaled_mm as batched_block32_mm,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def supports_ds41_mxfp8_bi() -> bool:
    """Limit recipe-specific K32 arithmetic to DS4.1 on SM100."""
    config = get_current_vllm_config_or_none()
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and config is not None
        and config.model_config is not None
        and config.model_config.hf_text_config.model_type == "deepseek_v41"
    )


@functools.cache
def _deep_gemm():
    from vllm.utils.deep_gemm import _import_deep_gemm

    return _import_deep_gemm()


def _deep_gemm_weight_scale(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[N, K/32] E8M0 bytes -> DeepGEMM's packed UE8M0 weight-scale layout."""
    sf = torch.exp2(scale.view(torch.uint8).float() - 127)
    return _deep_gemm().transform_sf_into_required_layout(
        sf, n, k, recipe=(1, 1, 32), is_sfa=False
    )


def block32_scaled_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Consume E4M3 codes/E8M0 bytes using FP8 products and FP32 accumulation.

    Each K32 product is accumulated before the next block, independently of
    token count, backend autotuning and the physical scale layout.
    """
    return batched_block32_mm(a, a_scale, weight, weight_scale, output_dtype)


@functools.cache
def _cute_dsl_mxfp8_runner(out_dtype: torch.dtype):
    """FlashInfer's cute-dsl MXFP8 runner with only its non-split-K tactics."""
    from flashinfer.gemm.gemm_base import _cute_dsl_gemm_mxfp8_runner

    base = type(_cute_dsl_gemm_mxfp8_runner(10, 0, True, out_dtype))

    class NoSplitKRunner(base):  # type: ignore[valid-type,misc]
        def get_valid_tactics(self, inputs, profile):
            tactics = super().get_valid_tactics(inputs, profile)
            return [t for t in tactics if t[-1] == 1]

    return NoSplitKRunner()


def _pinned_mxfp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """FlashInfer cute-dsl MXFP8 GEMM autotuned over non-split-K tactics only.

    Every non-split-K tactic gives a row the same bits at any M, so the tuned
    choice (or the untuned fallback, also non-split-K) only changes speed.
    """
    from flashinfer.autotuner import AutoTuner
    from flashinfer.gemm.gemm_base import (
        _MM_MXFP8_CUTE_DSL_TUNING_CONFIG,
        DEFAULT_WORKSPACE_SIZE,
        _get_cache_buf,
    )

    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
    if a.shape[0] == 0:
        return out
    workspace = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, a.device)
    runner = _cute_dsl_mxfp8_runner(out_dtype)
    inputs = [a, b, a_scale, b_scale, out_dtype, out, workspace]
    _, tactic = AutoTuner.get().choose_one(
        custom_op="bi_mxfp8_gemm",
        runners=[runner],
        tuning_config=_MM_MXFP8_CUTE_DSL_TUNING_CONFIG,
        inputs=inputs,
    )
    runner.forward(inputs, tactic=tactic)
    return out


def _pinned_mxfp8_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)


direct_register_custom_op(
    op_name="bi_pinned_mxfp8_mm",
    op_func=_pinned_mxfp8_mm,
    fake_impl=_pinned_mxfp8_mm_fake,
)


def _uses_tensor_core(weight_shape: tuple[int, int]) -> bool:
    n, k = weight_shape
    return envs.VLLM_BI_MXFP8_GEMM == "tensor_core" and n >= 128 and k >= 128


class BatchInvariantMxfp8LinearKernel(Mxfp8LinearKernel):
    """Fixed-block FP8 GEMM; retain quantized weights through reload."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        supported = supports_ds41_mxfp8_bi() and (
            compute_capability is None or 100 <= compute_capability < 110
        )
        return supported, "requires DS4.1 on CUDA SM100"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        n, k = c.weight_shape
        supported = c.bmm_batch_size is None and n > 0 and k > 0 and k % 32 == 0
        return supported, "requires an ungrouped positive shape with K divisible by 32"

    def input_quant_key(self) -> QuantKey | None:
        # Fused producers quantize per row, so their MXFP8 input keeps a row's
        # bits independent of M; the GEMM then skips its own quantize.
        if envs.VLLM_BI_MXFP8_PREQUANT and _uses_tensor_core(self.config.weight_shape):
            return kMxfp8Dynamic
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        n, k = weight.shape
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        scale = layer.weight_scale.data[:n, : k // 32].contiguous()
        if envs.VLLM_BI_WO_FP8_EPILOGUE and _uses_tensor_core((n, k)):
            # DeepGEMM's packed-UE8M0 weight scales for a pre-quantized input
            # in DeepGEMM's layout (wo_a's FP8 epilogue). A Parameter: layerwise
            # reload copies parameters back in place but skips non-persistent
            # buffers, which would leave stale scales after a weight update.
            layer.weight_scale_dg = Parameter(
                _deep_gemm_weight_scale(scale, n, k), requires_grad=False
            )
        if _uses_tensor_core((n, k)):
            scale = swizzle_mxfp8_scale(scale, M=n, K=k)
        layer.weight_scale = Parameter(scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = layer.weight.shape[-1]
        tensor_core = _uses_tensor_core(tuple(layer.weight.shape))
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            dtype, shape = qa.orig_dtype, qa.orig_shape
            codes, scales = qa.data.reshape(-1, k), qa.scale
        else:
            assert isinstance(x, torch.Tensor)
            dtype, shape = x.dtype, x.shape
            codes, scales = mxfp8_e4m3_quantize(
                x.reshape(-1, k), is_sf_swizzled_layout=tensor_core
            )
        if tensor_core and scales.dtype == torch.int32:
            # DeepGEMM-layout scales (packed UE8M0, MN-major): same bits as the
            # pinned FlashInfer GEMM on the same codes.
            output = torch.empty(
                (codes.shape[0], layer.weight.shape[0]),
                dtype=dtype,
                device=codes.device,
            )
            _deep_gemm().fp8_fp4_gemm_nt(
                (codes, scales),
                (layer.weight, layer.weight_scale_dg),
                output,
                recipe=(1, 1, 32),
            )
        elif tensor_core:
            output = torch.ops.vllm.bi_pinned_mxfp8_mm(
                codes, layer.weight.t(), scales, layer.weight_scale, dtype
            )
        else:
            output = block32_scaled_mm(
                codes, scales, layer.weight, layer.weight_scale, dtype
            )
        if bias is not None:
            output = output + bias
        return output.reshape(*shape[:-1], layer.weight.shape[0])
