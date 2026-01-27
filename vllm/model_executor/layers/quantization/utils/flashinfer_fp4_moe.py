# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for NVFP4 + FlashInfer fused-MoE path"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    has_flashinfer_cutlass_fused_moe,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
    )

logger = init_logger(__name__)


__all__ = [
    "is_flashinfer_fp4_cutlass_moe_available",
    "is_flashinfer_fp4_cutedsl_moe_available",
    "reorder_w1w3_to_w3w1",
]

#
# Methods used by the oracle for kernel selection.
#


def _supports_current_device() -> bool:
    """Supports only Blackwell-family GPUs."""
    p = current_platform
    return p.is_cuda() and p.is_device_capability_family(100)


def _supports_no_act_and_mul() -> bool:
    """Does not support non-gated MoE (i.e. Nemotron-Nano)."""
    return False


def _supports_quant_scheme(
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> bool:
    """Supports Nvfp4 quantization."""
    SUPPORTED_W_A = [
        (kNvfp4Static, kNvfp4Dynamic),
    ]
    return (weight_key, activation_key) in SUPPORTED_W_A


def _supports_activation(activation: str) -> bool:
    """Supports silu activation only."""
    return activation in ["silu"]


def _supports_routing_method(
    routing_method: RoutingMethodType,
) -> bool:
    """Monolithic kernels need to express router support."""
    # NOTE(rob): potentially allow others here. This is a conservative list.
    return routing_method in [
        RoutingMethodType.DeepSeekV3,
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
        RoutingMethodType.Llama4,
    ]


def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
    """Supports EP."""
    return True


def is_supported_config_trtllm(
    moe_config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    activation_format: mk.FusedMoEActivationFormat,
) -> tuple[bool, str | None]:
    """
    This method mirrors mk.FusedMoEPermuteExpertsUnpermute.is_supported_config
    """

    def _make_reason(reason: str) -> str:
        return f"kernel does not support {reason}"

    if not _supports_current_device():
        return False, _make_reason("current device")
    elif not (moe_config.is_act_and_mul or _supports_no_act_and_mul()):
        return False, _make_reason("no act_and_mul MLP layer")
    elif not _supports_activation(moe_config.activation):
        return False, _make_reason(f"{moe_config.activation} activation")
    elif not _supports_quant_scheme(weight_key, activation_key):
        return False, _make_reason("quantization scheme")
    elif not _supports_parallel_config(moe_config.moe_parallel_config):
        return False, _make_reason("parallel config")
    elif not _supports_routing_method(moe_config.routing_method):
        return False, _make_reason("routing method")
    elif activation_format != mk.FusedMoEActivationFormat.Standard:
        return False, _make_reason("activation format")

    return True, None


def is_flashinfer_fp4_cutlass_moe_available() -> bool:
    """Return `True` when FlashInfer CUTLASS NV-FP4 kernels can be used."""
    return (
        envs.VLLM_USE_FLASHINFER_MOE_FP4
        and has_flashinfer_cutlass_fused_moe()
        and current_platform.is_cuda()
        and current_platform.has_device_capability(100)
    )


def is_flashinfer_fp4_cutedsl_moe_available() -> bool:
    """Return ``True`` when FlashInfer CUTEDSL NV-FP4 kernels can be used."""
    return (
        envs.VLLM_USE_FLASHINFER_MOE_FP4
        and has_flashinfer_cutedsl_grouped_gemm_nt_masked()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    )


def reorder_w1w3_to_w3w1(
    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order the concatenated `[w1, w3]` tensors to `[w3, w1]`"""
    size = weight.size(dim)
    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
    half = size // 2

    w1, w3 = weight.split(half, dim=dim)
    s1, s3 = scale.split(half, dim=dim)

    return (
        torch.cat([w3, w1], dim=dim).contiguous(),
        torch.cat([s3, s1], dim=dim).contiguous(),
    )


def prepare_static_weights_for_trtllm_fp4_moe(
    # args_dequant,
    # args,
    gemm1_weights,
    gemm2_weights,
    gemm1_scales_linear_fp4_bytes,
    gemm2_scales_linear_fp4_bytes,
    hidden_size,
    intermediate_size,
    num_experts,
):
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    """Prepare quantized weights for kernel (done offline with weights)."""
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

    # Convert quantized weights to proper formats
    gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )  # packed fp4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 16
    )  # fp8 scaling factors

    gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(num_experts, hidden_size, intermediate_size // 16)  # fp8 scaling factors

    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    for i in range(num_experts):
        # Calculate the permute indices for the following:
        # 1. Reorder rows of W1 and scales for fused gated activation
        # 2. Shuffle weights and scaling factors for transposed mma output
        # for both w3_w1 and w2 weights and scale factors
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm1_weights_fp4_shuffled.append(
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm1_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm1_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

        permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm2_weights_fp4_shuffled.append(
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm2_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

    # Stack weights for all experts
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    )


def prepare_nvfp4_moe_layer_for_fi_or_cutlass(
    backend: "NvFp4MoeBackend",
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor,
    is_act_and_mul: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Delayed import for circular dependency avoidance.
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        is_global_sf_supported_for_nvfp4_backend,
    )

    assert backend in [
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
    ]

    # Reorder [w1, w3] to [w3, w1] for FI NVFP4 MoE kernels.
    if is_act_and_mul and backend in [
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
    ]:
        w13, w13_scale = reorder_w1w3_to_w3w1(w13, w13_scale)

    # For some FI kernels, the input scales are shared by all experts.
    if is_global_sf_supported_for_nvfp4_backend(backend):
        num_experts = w13.shape[0]
        a13_scale = a13_scale.max().to(torch.float32).expand(num_experts)
        a2_scale = a2_scale.max().to(torch.float32).expand(num_experts)
    else:
        a13_scale = a13_scale.max(dim=1).values.to(torch.float32)

    # Shuffle weights and scales for FI TRTLLM NVFP4 MoE kernels.
    if backend == NvFp4MoeBackend.FLASHINFER_TRTLLM:
        w13, w13_scale, w2, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
            w13,
            w2,
            w13_scale,
            w2_scale,
            w2.size(-2),  # hidden_size
            w13.size(-2) // 2,  # intermediate_size
            w13.size(0),  # num_experts
        )

        # We do not need to make this a parameter, because
        # it is not used during the weight (re)-loading process.
        layer.g1_scale_c = a13_scale * w13_scale_2 / a2_scale
        layer.a1_gscale = 1.0 / a13_scale
        layer.g1_alphas = a13_scale * w13_scale_2
        layer.g2_alphas = a2_scale * w2_scale_2
    else:
        # Swizzle the block scales for other FI NVFP4 MoE kernels.
        w13_scale = swizzle_blockscale(w13_scale)

        # Apply padding if needed.
        pad_size = w13_scale.size(1) - w13.size(1)
        if pad_size > 0:
            if is_act_and_mul:
                raise NotImplementedError(
                    "Intermediate size padding for w1 and w3, for %s "
                    "NvFp4 backend, but this is not currently supported",
                    backend.value,
                )
            w13 = torch.nn.functional.pad(w13, (0, 0, 0, pad_size))
            w2 = torch.nn.functional.pad(w2, (0, pad_size // 2, 0, 0))
            w2_scale = torch.nn.functional.pad(w2_scale, (0, pad_size // 16))

        w2_scale = swizzle_blockscale(w2_scale)

    return w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale
