# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Union

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    mxfp4_mxfp8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        from triton_kernels.matmul_ogs import PrecisionConfig
    except (ImportError, AttributeError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


class Mxfp4MoeBackend(Enum):
    NONE = "None"
    # FlashInfer TRTLLM backends
    FLASHINFER_TRTLLM_MXFP4_MXFP8 = "FLASHINFER_TRTLLM_MXFP4_MXFP8"
    FLASHINFER_TRTLLM_MXFP4_BF16 = "FLASHINFER_TRTLLM_MXFP4_BF16"
    # FlashInfer CUTLASS backends
    FLASHINFER_CUTLASS_MXFP4_MXFP8 = "FLASHINFER_CUTLASS_MXFP4_MXFP8"
    FLASHINFER_CUTLASS_MXFP4_BF16 = "FLASHINFER_CUTLASS_MXFP4_BF16"
    # Marlin
    BATCHED_MARLIN = "BATCHED_MARLIN"
    MARLIN = "MARLIN"
    # ROCm AITER
    AITER = "AITER"
    # Triton
    TRITON = "TRITON"
    TRITON_UNFUSED = "TRITON_UNFUSED"
    # XPU
    XPU = "XPU"
    # Emulation
    EMULATION = "EMULATION"


# Backends that share the same TRTLLM weight format
TRTLLM_BACKENDS = (
    Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
    Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
)

TRITON_BACKENDS = (
    Mxfp4MoeBackend.TRITON,
    Mxfp4MoeBackend.TRITON_UNFUSED,
)


def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
    ):
        from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
            TrtLlmMxfp4ExpertsModular,
            TrtLlmMxfp4ExpertsMonolithic,
        )

        # NOTE: prefer Monolithic > Modular, so return Monolithic first.
        return [TrtLlmMxfp4ExpertsMonolithic, TrtLlmMxfp4ExpertsModular]

    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        return [FlashInferExperts]

    elif backend == Mxfp4MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
            OAITritonExperts,
            OAITritonMxfp4ExpertsMonolithic,
        )

        # NOTE: prefer Monolithic > Modular, so return Monolithic first.
        return [OAITritonMxfp4ExpertsMonolithic, OAITritonExperts]

    elif backend == Mxfp4MoeBackend.TRITON_UNFUSED:
        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
            UnfusedOAITritonExperts,
        )

        return [UnfusedOAITritonExperts]

    elif backend == Mxfp4MoeBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            MarlinExperts,
        )

        return [MarlinExperts]

    elif backend == Mxfp4MoeBackend.BATCHED_MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            BatchedMarlinExperts,
        )

        return [BatchedMarlinExperts]

    elif backend == Mxfp4MoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        return [AiterExperts]

    elif backend == Mxfp4MoeBackend.XPU:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsMXFp4

        return [XPUExpertsMXFp4]

    elif backend == Mxfp4MoeBackend.EMULATION:
        from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
            OCP_MXQuantizationEmulationTritonExperts,
        )

        return [OCP_MXQuantizationEmulationTritonExperts]

    else:
        raise ValueError(f"Unknown MXFP4 MoE backend: {backend.value}")


def map_mxfp4_backend(runner_backend: MoEBackend) -> Mxfp4MoeBackend:
    """Map user's moe_backend string to Mxfp4MoeBackend."""
    mapping: dict[str, Mxfp4MoeBackend] = {
        "flashinfer_trtllm": Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        "flashinfer_trtllm_afp8": Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        "flashinfer_cutlass": Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        "flashinfer_cutlass_afp8": Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        "triton": Mxfp4MoeBackend.TRITON,
        "marlin": Mxfp4MoeBackend.MARLIN,
        "aiter": Mxfp4MoeBackend.AITER,
        "xpu": Mxfp4MoeBackend.XPU,
        "emulation": Mxfp4MoeBackend.EMULATION,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for MXFP4 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def _get_priority_backends() -> list[Mxfp4MoeBackend]:
    """
    Get available backends in priority order based on platform and config.
    Only includes BF16 backends. MXFP8 backends are selected via env vars.
    """
    _AVAILABLE_BACKENDS = [
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.AITER,
        Mxfp4MoeBackend.TRITON,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.TRITON_UNFUSED,
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
        Mxfp4MoeBackend.XPU,
        Mxfp4MoeBackend.EMULATION,
    ]
    return _AVAILABLE_BACKENDS


def _backend_activation_key(backend: Mxfp4MoeBackend) -> QuantKey | None:
    """Map backend to its activation key (MXFP8 or None for BF16)."""
    if backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        return kMxfp8Dynamic
    return None


def select_gpt_oss_mxfp4_moe_backend(
    config: FusedMoEConfig,
) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]:
    """
    Select the primary MXFP4 MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    device_capability = current_platform.get_device_capability()
    triton_kernels_supported = (
        has_triton_kernels()
        and device_capability is not None
        and (9, 0) <= device_capability < (11, 0)
    )

    # LoRA: separate experts backend path
    if config.is_lora_enabled:
        if not current_platform.is_cuda():
            # ROCm: Triton mxfp4 LoRA hits GPU memory faults due to
            # triton_kernels.tensor.Tensor / HIP read-only page issues
            # during weight swizzle and LoRA forward. Needs work from
            # the triton_kernels/aiter side.
            raise NotImplementedError("Mxfp4 LoRA is currently only supported on CUDA.")
        if envs.VLLM_MXFP4_USE_MARLIN is False and triton_kernels_supported:
            logger.info_once("Using Triton backend for mxfp4 lora")
            return Mxfp4MoeBackend.TRITON_UNFUSED, backend_to_kernel_cls(
                Mxfp4MoeBackend.TRITON_UNFUSED
            )[0]
        logger.info_once("Using Marlin backend for mxfp4 lora")
        return Mxfp4MoeBackend.MARLIN, backend_to_kernel_cls(Mxfp4MoeBackend.MARLIN)[0]

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Mxfp4MoeBackend):
        return f"Using '{backend.value}' Mxfp4 MoE backend."

    def _make_log_unsupported(backend: Mxfp4MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"Mxfp4 MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"Mxfp4 MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: Mxfp4MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts]]:
        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_mxfp4_backend(runner_backend)
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == Mxfp4MoeBackend.MARLIN
        ):
            requested_backend = Mxfp4MoeBackend.BATCHED_MARLIN
        return _return_or_raise(
            requested_backend,
            config,
            kMxfp4Static,
            _backend_activation_key(requested_backend),
            activation_format,
        )

    # Select kernels in order of backend.
    AVAILABLE_BACKENDS = _get_priority_backends()

    # Handle explicit FlashInfer MXFP4 BF16 configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"):
        if not envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16)
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16)
        else:
            if current_platform.is_device_capability(90):
                return _return_or_raise(
                    Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
                    config,
                    kMxfp4Static,
                    None,
                    activation_format,
                )
            if current_platform.is_device_capability_family(100):
                return _return_or_raise(
                    Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
                    config,
                    kMxfp4Static,
                    None,
                    activation_format,
                )
            raise ValueError(
                "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 is set but the "
                "current device capability is not supported. "
                "Only SM90 (CUTLASS) and SM100+ (TRTLLM) are supported."
            )

    # Handle explicit FlashInfer MXFP4 MXFP8 TRTLLM configuration.
    if (
        envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8")
        and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8
    ):
        return _return_or_raise(
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            config,
            kMxfp4Static,
            kMxfp8Dynamic,
            activation_format,
        )

    # Handle explicit FlashInfer MXFP4 MXFP8 CUTLASS configuration.
    if (
        envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS")
        and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS
    ):
        return _return_or_raise(
            Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
            config,
            kMxfp4Static,
            kMxfp8Dynamic,
            activation_format,
        )

    # Handle explicit Marlin MXFP4 configuration.
    if envs.is_set("VLLM_MXFP4_USE_MARLIN") and envs.VLLM_MXFP4_USE_MARLIN:
        return _return_or_raise(
            Mxfp4MoeBackend.MARLIN,
            config,
            kMxfp4Static,
            None,
            activation_format,
        )

    for backend in AVAILABLE_BACKENDS:
        activation_key = _backend_activation_key(backend)
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, kMxfp4Static, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason))

    if current_platform.is_xpu():
        backend = Mxfp4MoeBackend.XPU
        logger.info_once(_make_log_backend(backend))
        return _return_or_raise(
            Mxfp4MoeBackend.XPU,
            config,
            kMxfp4Static,
            None,
            activation_format,
        )

    if current_platform.is_cuda() or current_platform.is_rocm():
        raise NotImplementedError(
            "No MXFP4 MoE backend supports the deployment configuration."
        )

    return Mxfp4MoeBackend.NONE, None


def mxfp4_round_up_hidden_size_and_intermediate_size(
    backend: Mxfp4MoeBackend, hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    """Round up hidden_size and intermediate_size based on backend requirements."""
    if backend in (Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.BATCHED_MARLIN):
        intermediate_size = round_up(intermediate_size, 128)
        if current_platform.is_xpu():
            hidden_size = round_up(hidden_size, 128)
        else:
            hidden_size = round_up(hidden_size, 256)
    elif backend in TRTLLM_BACKENDS:
        intermediate_size = round_up(intermediate_size, 256)
        hidden_size = round_up(hidden_size, 256)
    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        intermediate_size = round_up(intermediate_size, 128)
        hidden_size = round_up(hidden_size, 128)
    elif current_platform.is_rocm():
        intermediate_size = round_up(intermediate_size, 256)
        hidden_size = round_up(hidden_size, 256)
    else:
        intermediate_size = round_up(intermediate_size, 64)
    return hidden_size, intermediate_size


def convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
    mxfp4_backend: Mxfp4MoeBackend,
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    _cache_permute_indices: dict[torch.Size, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Union[torch.Tensor, "PrecisionConfig"],
    Union[torch.Tensor, "PrecisionConfig"],
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Convert loaded weights into backend-specific kernel format."""

    num_experts = w13_weight.shape[0]
    intermediate_size = w13_weight.shape[1] // 2
    hidden_size = w13_weight.shape[2] * 2

    sf_block_size = 32  # mxfp4 block size

    if mxfp4_backend in (
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
    ):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        return prepare_moe_mxfp4_layer_for_marlin(
            layer,
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    elif mxfp4_backend in TRTLLM_BACKENDS:
        assert _cache_permute_indices is not None
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

        # gemm1_alpha/beta/clamp_limit are created by the expert class
        # (TrtLlmMxfp4ExpertsBase), not on the layer.

        w13_weight = w13_weight.data
        w2_weight = w2_weight.data
        w13_weight_scale = w13_weight_scale.data
        w2_weight_scale = w2_weight_scale.data
        assert w13_bias is not None and w2_bias is not None
        w13_bias = w13_bias.data.to(torch.float32)
        w2_bias = w2_bias.data.to(torch.float32)

        # Swap w1 and w3 as the definition of swiglu is different in trtllm-gen
        def swap_every_two_rows(x, axis=-1):
            shape = x.shape
            if axis < 0:
                axis = len(shape) + axis
            new_shape = list(shape)
            new_shape[axis] = shape[axis] // 2
            new_shape.insert(axis + 1, 2)
            x = x.reshape(*new_shape)
            x = x.flip(axis + 1)
            new_shape = list(shape)
            return x.reshape(*new_shape)

        w13_weight_scale = swap_every_two_rows(w13_weight_scale, -2)
        w13_weight = swap_every_two_rows(w13_weight, -2)
        w13_bias = swap_every_two_rows(w13_bias, -1)

        # Shuffle weights and scaling factors for transposed mma output
        gemm1_weights_shuffled = []
        gemm1_scales_shuffled = []
        gemm2_weights_shuffled = []
        gemm2_scales_shuffled = []
        gemm1_bias_shuffled = []
        gemm2_bias_shuffled = []
        epilogue_tile_m = 128
        for i in range(num_experts):
            # w13 weight
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_shuffled.append(
                w13_weight[i]
                .view(torch.uint8)[permute_indices.to(w13_weight.device)]
                .contiguous()
            )
            # w13 scale
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w13_weight_scale.device)]
                    .contiguous()
                )
            )
            # w13 bias
            permute_bias_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_bias[i].clone().reshape(-1, 1),
                epilogue_tile_m,
            )
            gemm1_bias_shuffled.append(
                w13_bias[i]
                .clone()
                .reshape(-1, 1)[permute_bias_indices.to(w13_bias.device)]
                .contiguous()
            )
            # w2 weight
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_shuffled.append(
                w2_weight[i]
                .view(torch.uint8)[permute_indices.to(w2_weight.device)]
                .contiguous()
            )
            # w2 scale
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w2_weight_scale.device)]
                    .contiguous()
                )
            )
            # w2 bias
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_bias[i].clone().reshape(-1, 1),
                epilogue_tile_m,
            )
            gemm2_bias_shuffled.append(
                w2_bias[i]
                .clone()
                .reshape(-1, 1)[permute_indices.to(w2_bias.device)]
                .contiguous()
            )

        w13_weight = torch.stack(gemm1_weights_shuffled)
        w13_weight_scale = (
            torch.stack(gemm1_scales_shuffled)
            .reshape(num_experts, 2 * intermediate_size, hidden_size // sf_block_size)
            .view(torch.float8_e4m3fn)
        )
        w2_weight = torch.stack(gemm2_weights_shuffled)
        w2_weight_scale = (
            torch.stack(gemm2_scales_shuffled)
            .reshape(num_experts, hidden_size, intermediate_size // sf_block_size)
            .view(torch.float8_e4m3fn)
        )
        w13_bias = torch.stack(gemm1_bias_shuffled).reshape(num_experts, -1)
        w2_bias = torch.stack(gemm2_bias_shuffled).reshape(num_experts, -1)

        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    elif mxfp4_backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        # De-interleave and swap for w13 weight, bias, and scales
        w13_w = w13_weight.data
        gate_w, up_w = w13_w[:, ::2, :], w13_w[:, 1::2, :]
        deinterleaved_w13_w = torch.cat([gate_w, up_w], dim=1)
        w1_w, w3_w = torch.chunk(deinterleaved_w13_w, 2, dim=1)
        w13_weight_swapped = torch.cat([w3_w, w1_w], dim=1)

        assert w13_bias is not None and w2_bias is not None
        w13_b = w13_bias.data.to(torch.float32)
        gate_b, up_b = w13_b[:, ::2], w13_b[:, 1::2]
        deinterleaved_w13_b = torch.cat([gate_b, up_b], dim=1)
        b1, b3 = torch.chunk(deinterleaved_w13_b, 2, dim=-1)
        w13_bias_swapped = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

        w13_s = w13_weight_scale.data
        gate_s, up_s = w13_s[:, ::2, :], w13_s[:, 1::2, :]
        deinterleaved_w13_s = torch.cat([gate_s, up_s], dim=1)
        s1, s3 = torch.chunk(deinterleaved_w13_s, 2, dim=1)
        w13_scale_swapped = torch.cat([s3, s1], dim=1)

        if mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8:
            from flashinfer import block_scale_interleave

            orig_shape = w13_scale_swapped.shape
            w13_scale_interleaved = block_scale_interleave(
                w13_scale_swapped.view(torch.uint8)
            ).reshape(orig_shape)

            w2_s = w2_weight_scale.data
            orig_shape = w2_s.shape
            w2_scale_interleaved = block_scale_interleave(
                w2_s.view(torch.uint8)
            ).reshape(orig_shape)

            return (
                w13_weight_swapped,
                w2_weight,
                w13_scale_interleaved,
                w2_scale_interleaved,
                w13_bias_swapped,
                w2_bias,
            )

        else:
            assert mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16

            def _interleave_mxfp4_cutlass_sm90(w):
                w_shape = w.shape
                w_interleaved = w.reshape(w_shape[0], w_shape[1], (w_shape[2] // 4), 4)
                w_interleaved = w_interleaved.permute(0, 2, 1, 3)
                w_interleaved = w_interleaved.reshape(
                    w_shape[0], w_shape[2] // 4, w_shape[1] * 4
                )
                return w_interleaved

            w31_scales = w13_scale_swapped.to(torch.uint8)
            w31_scales_interleaved = _interleave_mxfp4_cutlass_sm90(w31_scales)

            w2_scale = w2_weight_scale.data.to(torch.uint8)
            w2_scale_interleaved = _interleave_mxfp4_cutlass_sm90(w2_scale)

            return (
                w13_weight_swapped,
                w2_weight,
                w31_scales_interleaved,
                w2_scale_interleaved,
                w13_bias_swapped,
                w2_bias,
            )

    elif mxfp4_backend == Mxfp4MoeBackend.AITER:
        from vllm._aiter_ops import rocm_aiter_ops

        if w13_bias is not None:
            w13_bias = w13_bias.data.to(torch.float32)
        if w2_bias is not None:
            w2_bias = w2_bias.data.to(torch.float32)

        e, n, k = w13_weight.shape

        # De-interleave w13 rows: gate/up pairs -> contiguous gate, up blocks
        w13_weight.view(torch.uint8).copy_(
            w13_weight.data.view(torch.uint8)
            .view(e, n // 2, 2, k)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, k)
        )
        w13_weight_scale.data = (
            w13_weight_scale.data.view(e, n // 2, 2, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(e, n, -1)
        )

        # View as native FP4 dtype for AITER shuffle
        w13_weight.data = w13_weight.data.view(torch.float4_e2m1fn_x2)
        w2_weight.data = w2_weight.data.view(torch.float4_e2m1fn_x2)

        # Shuffle weights and scales for AITER CK kernel layout
        w13_weight.data = rocm_aiter_ops.shuffle_weight_a16w4(w13_weight, 16, True)
        shuffled_w13_scale = rocm_aiter_ops.shuffle_scale_a16w4(
            w13_weight_scale.view(-1, w13_weight_scale.shape[-1]),
            num_experts,
            True,
        )

        w2_weight.data = rocm_aiter_ops.shuffle_weight_a16w4(w2_weight, 16, False)
        shuffled_w2_scale = rocm_aiter_ops.shuffle_scale_a16w4(
            w2_weight_scale.view(-1, w2_weight_scale.shape[-1]),
            num_experts,
            False,
        )

        # Permute bias to match de-interleaved weight layout
        if w13_bias is not None:
            w13_bias = (
                w13_bias.data.view(-1, n // 2, 2)
                .permute(0, 2, 1)
                .contiguous()
                .view(-1, n)
            )

        return (
            w13_weight,
            w2_weight,
            shuffled_w13_scale,
            shuffled_w2_scale,
            w13_bias,
            w2_bias,
        )

    elif mxfp4_backend in TRITON_BACKENDS:
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

        assert w13_bias is not None and w2_bias is not None
        w13_bias = w13_bias.to(torch.float32)
        w2_bias = w2_bias.to(torch.float32)

        w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
            w13_weight,
            w13_weight_scale,
        )
        w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
            w2_weight,
            w2_weight_scale,
        )

        w13_precision_config = PrecisionConfig(
            weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
        )
        w2_precision_config = PrecisionConfig(
            weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
        )

        del layer.w13_weight
        del layer.w2_weight

        return (
            w13_weight,
            w2_weight,
            w13_precision_config,
            w2_precision_config,
            w13_bias,
            w2_bias,
        )
    elif mxfp4_backend == Mxfp4MoeBackend.XPU:
        # No additional transformation needed for XPU backend
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )
    elif mxfp4_backend == Mxfp4MoeBackend.EMULATION:
        # No additional transformation needed for emulation backend,
        # weights are dequantized on the fly in the experts class.
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )
    else:
        raise ValueError(
            f"Unsupported mxfp4_backend: {mxfp4_backend}: "
            f"should be one of: {list(Mxfp4MoeBackend)}."
        )


def make_mxfp4_moe_quant_config(
    mxfp4_backend: Mxfp4MoeBackend,
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig | None:
    """Create a FusedMoEQuantConfig for the given MXFP4 backend."""
    if mxfp4_backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        return mxfp4_mxfp8_moe_quant_config(
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
    elif mxfp4_backend in (
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
        Mxfp4MoeBackend.TRITON,
        Mxfp4MoeBackend.TRITON_UNFUSED,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.AITER,
    ):
        return mxfp4_w4a16_moe_quant_config(
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
    else:
        return ocp_mx_moe_quant_config(
            quant_dtype="mxfp4",
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )


def make_mxfp4_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    mxfp4_backend: Mxfp4MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel:
    """Create a FusedMoEKernel for the given MXFP4 backend."""
    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)

    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=(
            shared_experts
            if moe_config.moe_parallel_config.use_deepep_ll_kernels
            else None
        ),
        inplace=(
            not moe_config.disable_inplace and mxfp4_backend not in TRTLLM_BACKENDS
        ),
    )

    return kernel
