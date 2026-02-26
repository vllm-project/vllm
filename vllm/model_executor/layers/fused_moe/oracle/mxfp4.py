# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Union

import torch
from torch.nn.parameter import Parameter

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
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
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    _swizzle_mxfp4,
    get_padding_alignment,
)
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
    # FIXME(zyongye) we temporarily treat monolithic and modular into 2 backend
    # pending unifying them after https://github.com/vllm-project/vllm/pull/32564
    NONE = "None"
    FLASHINFER_TRTLLM_MXFP4_MXFP8 = "FLASHINFER_TRTLLM_MXFP4_MXFP8"
    FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC = (
        "FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC"
    )
    FLASHINFER_CUTLASS_MXFP4_MXFP8 = "FLASHINFER_CUTLASS_MXFP4_MXFP8"
    FLASHINFER_TRTLLM_MXFP4_BF16 = "FLASHINFER_TRTLLM_MXFP4_BF16"
    FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC = "FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC"
    FLASHINFER_CUTLASS_MXFP4_BF16 = "FLASHINFER_CUTLASS_MXFP4_BF16"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    TRITON_MONOLITHIC = "TRITON_MONOLITHIC"
    TRITON_UNFUSED = "TRITON_UNFUSED"
    XPU = "XPU"


def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> type[mk.FusedMoEPermuteExpertsUnpermute]:
    if backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
        Mxfp4MoeBackend.TRITON_MONOLITHIC,
        Mxfp4MoeBackend.XPU,
    ):
        raise NotImplementedError
    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
    ):
        from vllm.model_executor.layers.fused_moe.trtllm_moe import (
            TrtLlmGenExperts,
        )

        return TrtLlmGenExperts
    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        return FlashInferExperts
    elif backend == Mxfp4MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
            OAITritonExperts,
        )

        return OAITritonExperts
    elif backend == Mxfp4MoeBackend.TRITON_UNFUSED:
        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
            UnfusedOAITritonExperts,
        )

        return UnfusedOAITritonExperts
    elif backend == Mxfp4MoeBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            MarlinExperts,
        )

        return MarlinExperts
    elif backend == Mxfp4MoeBackend.BATCHED_MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            BatchedMarlinExperts,
        )

        return BatchedMarlinExperts

    else:
        raise ValueError(f"Unknown MXFP4 MoE backend: {backend.value}")


def select_mxfp4_moe_backend(
    config: FusedMoEConfig,
) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEPermuteExpertsUnpermute] | None]:
    """
    Select the primary MXFP4 MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    # If FlashInfer is not available, try either Marlin or Triton
    triton_kernels_supported = (
        has_triton_kernels()
        # NOTE: triton_kernels are only confirmed to work on SM90 and SM100
        # SM110 fails with this error: https://github.com/vllm-project/vllm/issues/29317
        # SM120 needs this fix: https://github.com/triton-lang/triton/pull/8498
        and (9, 0) <= current_platform.get_device_capability() < (11, 0)
    )

    if config.is_lora_enabled:
        if not current_platform.is_cuda():
            raise NotImplementedError("Mxfp4 LoRA only supported on CUDA Platform.")

        if envs.VLLM_MXFP4_USE_MARLIN is False and triton_kernels_supported:
            logger.info_once("Using Triton backend for mxfp4 lora")
            return Mxfp4MoeBackend.TRITON_UNFUSED, backend_to_kernel_cls(
                Mxfp4MoeBackend.TRITON_UNFUSED
            )

        logger.info_once("Using Marlin backend for mxfp4 lora")
        return Mxfp4MoeBackend.MARLIN, backend_to_kernel_cls(Mxfp4MoeBackend.MARLIN)

    # FIXME(zyongye): we still need to fix kernel section
    # after monolithic kernel refactor PR is merged
    AVAILABLE_BACKENDS = [
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
        Mxfp4MoeBackend.TRITON,
        Mxfp4MoeBackend.TRITON_MONOLITHIC,
        Mxfp4MoeBackend.TRITON_UNFUSED,
        Mxfp4MoeBackend.XPU,
    ]

    # NOTE(zyongye): See similar comments in fp8.py
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Mxfp4MoeBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Mxfp4 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Mxfp4MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"Mxfp4 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
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
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEPermuteExpertsUnpermute]]:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, config, weight_key, activation_key, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend), scope="local")
            return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit FlashInfer MXFP4 BF16 configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"):
        if not envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16)
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16)
            AVAILABLE_BACKENDS.remove(
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC
            )
        else:
            if current_platform.is_device_capability(90):
                backend = Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16
                return _return_or_raise(
                    backend,
                    config,
                    kMxfp4Static,
                    None,
                    activation_format,
                )
            if current_platform.is_device_capability_family(100):
                # Using modular interface
                # unifying them after #32564 is merged
                if config.dp_size > 1 and config.use_ep:
                    backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16
                    return _return_or_raise(
                        backend,
                        config,
                        kMxfp4Static,
                        None,
                        activation_format,
                    )
                else:
                    backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC
                    return backend, None

    # Handle explicit FlashInfer MXFP4 MXFP8 TRTLLM configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"):
        # same as BF16 case
        if not envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8)
            AVAILABLE_BACKENDS.remove(
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC
            )
        if config.dp_size > 1 and config.use_ep:
            backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8
            return _return_or_raise(
                backend,
                config,
                kMxfp4Static,
                kMxfp8Dynamic,
                activation_format,
            )
        else:
            backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC
            return backend, None

    # Handle explicit FlashInfer MXFP4 MXFP8 CUTLASS configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"):
        if not envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8)
        else:
            backend = Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8
            return _return_or_raise(
                backend,
                config,
                kMxfp4Static,
                kMxfp8Dynamic,
                activation_format,
            )

    # Handle explicit Marlin MXFP4 configuration.
    if envs.is_set("VLLM_MXFP4_USE_MARLIN"):
        if not envs.VLLM_MXFP4_USE_MARLIN:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.MARLIN)
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.BATCHED_MARLIN)
        else:
            backend = Mxfp4MoeBackend.MARLIN
            return _return_or_raise(
                backend,
                config,
                kMxfp4Static,
                None,
                activation_format,
            )

    # FIXME(zyongye): manually select default kernels
    # change to automatic after monolithic kernel PR is merged
    if (
        current_platform.is_device_capability_family(100)
        and Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16 in AVAILABLE_BACKENDS
    ):
        if config.dp_size > 1 and config.use_ep:
            backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16
            return _return_or_raise(
                backend, config, kMxfp4Static, None, activation_format
            )
        else:
            backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC
            logger.info_once(_make_log_backend(backend))
            return backend, None
    elif current_platform.has_device_capability(90):
        if config.dp_size > 1 and config.use_ep:
            backend = Mxfp4MoeBackend.TRITON
            return _return_or_raise(
                backend,
                config,
                kMxfp4Static,
                None,
                activation_format,
            )
        else:
            backend = Mxfp4MoeBackend.TRITON_MONOLITHIC
            logger.info_once(_make_log_backend(backend))
            return backend, None
    elif current_platform.has_device_capability(70):
        backend = (
            Mxfp4MoeBackend.MARLIN
            if activation_format == mk.FusedMoEActivationFormat.Standard
            else Mxfp4MoeBackend.BATCHED_MARLIN
        )
        return _return_or_raise(
            backend,
            config,
            kMxfp4Static,
            None,
            activation_format,
        )
    elif current_platform.is_xpu():
        backend = Mxfp4MoeBackend.XPU
        logger.info_once(_make_log_backend(backend))
        return backend, None

    if current_platform.is_cuda() or current_platform.is_rocm():
        raise NotImplementedError(
            "No MXFP4 MoE backend supports the deployment configuration."
        )

    return Mxfp4MoeBackend.NONE, None


def convert_to_mxfp4_moe_kernel_format(
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
    Union[type[torch.Tensor], "PrecisionConfig"],
    Union[type[torch.Tensor], "PrecisionConfig"],
    type[torch.Tensor] | None,
    type[torch.Tensor] | None,
]:
    assert _cache_permute_indices is not None

    num_experts = w13_weight.shape[0]
    intermediate_size = w13_weight.shape[1] // 2
    hidden_size = w13_weight.shape[2] * 2

    sf_block_size = 32  # mxfp4 block size
    assert w13_bias is not None and w2_bias is not None

    if mxfp4_backend in (Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.BATCHED_MARLIN):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        ) = prepare_moe_mxfp4_layer_for_marlin(
            layer,
            w13=w13_weight,
            w13_scale=w13_weight_scale,
            w13_bias=w13_bias,
            w2=w2_weight,
            w2_scale=w2_weight_scale,
            w2_bias=w2_bias,
        )
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )
    elif mxfp4_backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
    ):
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

        layer.gemm1_alpha = Parameter(
            torch.tensor([1.702] * num_experts, dtype=torch.float32).cuda(),
            requires_grad=False,
        )
        layer.gemm1_beta = Parameter(
            torch.tensor([1.0] * num_experts, dtype=torch.float32).cuda(),
            requires_grad=False,
        )
        layer.gemm1_clamp_limit = Parameter(
            torch.tensor([7.0] * num_experts, dtype=torch.float32).cuda(),
            requires_grad=False,
        )

        w13_weight = w13_weight.data
        w2_weight = w2_weight.data
        w13_weight_scale = w13_weight_scale.data
        w2_weight_scale = w2_weight_scale.data
        w13_bias = w13_bias.data.to(torch.float32)
        w2_bias = w2_bias.data.to(torch.float32)

        # Swap w1 and w3 as the definition of
        # swiglu is different in the trtllm-gen
        def swap_every_two_rows(x, axis=-1):
            shape = x.shape
            if axis < 0:
                axis = len(shape) + axis

            # Create a new shape with pairs swapped along specified axis
            new_shape = list(shape)
            new_shape[axis] = shape[axis] // 2
            new_shape.insert(axis + 1, 2)

            # Reshape to expose pairs, swap them, and reshape back
            x = x.reshape(*new_shape)
            x = x.flip(axis + 1)
            new_shape = list(shape)
            return x.reshape(*new_shape)

        w13_weight_scale = swap_every_two_rows(w13_weight_scale, -2)
        w13_weight = swap_every_two_rows(w13_weight, -2)
        w13_bias = swap_every_two_rows(w13_bias, -1)

        # Do not interleave as the checkpoint is already interleaved

        # Shuffle weights and scaling factors for transposed mma output
        gemm1_weights_mxfp4_shuffled = []
        gemm1_scales_mxfp4_shuffled = []
        gemm2_weights_mxfp4_shuffled = []
        gemm2_scales_mxfp4_shuffled = []
        gemm1_bias_shuffled = []
        gemm2_bias_shuffled = []
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
        for i in range(num_experts):
            # w13 weight shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_mxfp4_shuffled.append(
                w13_weight[i]
                .view(torch.uint8)[permute_indices.to(w13_weight.device)]
                .contiguous()
            )
            # w13 scale shuffling
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_mxfp4_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w13_weight_scale.device)]
                    .contiguous()
                )
            )
            # w13 bias shuffling
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
            # w2 weight shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_mxfp4_shuffled.append(
                w2_weight[i]
                .view(torch.uint8)[permute_indices.to(w2_weight.device)]
                .contiguous()
            )
            # w2 scale shuffling
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_mxfp4_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w2_weight_scale.device)]
                    .contiguous()
                )
            )
            # w2 bias shuffling
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

        w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
        w13_weight_scale = (
            torch.stack(gemm1_scales_mxfp4_shuffled)
            .reshape(
                num_experts,
                2 * intermediate_size,
                hidden_size // sf_block_size,
            )
            .view(torch.float8_e4m3fn)
        )

        w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
        w2_weight_scale = (
            torch.stack(gemm2_scales_mxfp4_shuffled)
            .reshape(
                num_experts,
                hidden_size,
                intermediate_size // sf_block_size,
            )
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

            w13_weight = w13_weight_swapped
            w13_weight_scale = w13_scale_interleaved
            w13_bias = w13_bias_swapped
            w2_weight_scale = w2_scale_interleaved

            return (
                w13_weight,
                w2_weight,
                w13_weight_scale,
                w2_weight_scale,
                w13_bias,
                w2_bias,
            )

        elif mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16:

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

            w2_weight_scale = layer.w2_weight_scale.data
            w2_scale = w2_weight_scale.to(torch.uint8)
            w2_scale_interleaved = _interleave_mxfp4_cutlass_sm90(w2_scale)

            w13_weight = w13_weight_swapped
            w13_bias = w13_bias_swapped
            w13_weight_scale = w31_scales_interleaved
            w2_weight_scale = w2_scale_interleaved

            return (
                w13_weight,
                w2_weight,
                w13_weight_scale,
                w2_weight_scale,
                w13_bias,
                w2_bias,
            )

    elif mxfp4_backend in (
        Mxfp4MoeBackend.TRITON,
        Mxfp4MoeBackend.TRITON_MONOLITHIC,
        Mxfp4MoeBackend.TRITON_UNFUSED,
    ):
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

        w13_bias = layer.w13_bias.to(torch.float32)
        w2_bias = layer.w2_bias.to(torch.float32)

        w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
            layer.w13_weight,
            layer.w13_weight_scale,
        )
        w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
            layer.w2_weight,
            layer.w2_weight_scale,
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
    else:
        raise ValueError(
            f"Unsupported mxfp4_backend: {mxfp4_backend}: "
            f"should be one of: {list(Mxfp4MoeBackend)}."
        )
    return (
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_bias,
        w2_bias,
    )


def mxfp4_round_up_hidden_size_and_intermediate_size(
    backend: Mxfp4MoeBackend, hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    if backend in (Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.BATCHED_MARLIN):
        # The moe marlin kernel requires that for each linear
        # n % 256 == 0 and k % 128 == 0.
        # In gate_up_proj:
        #    n = 2 * intermediate_size_per_partition_after_pad
        #    k = hidden_size
        # In down_proj
        #    n = hidden_size
        #    k = intermediate_size_per_partition_after_pad
        intermediate_size = round_up(intermediate_size, 128)
        if backend == Mxfp4MoeBackend.XPU:
            hidden_size = round_up(hidden_size, 128)
        else:
            hidden_size = round_up(hidden_size, 256)

    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
    ):
        # pad the intermediate size to be a multiple of 2 * mxfp4_block
        # for to hold non-uniform sharded tensor as well as swizzling
        # other padding to increase performance
        intermediate_size = round_up(intermediate_size, 256)
        hidden_size = round_up(hidden_size, 256)
    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        intermediate_size = round_up(intermediate_size, 128)
        hidden_size = round_up(hidden_size, 128)
    elif current_platform.is_rocm():
        pad_align = get_padding_alignment()
        intermediate_size = round_up(intermediate_size, pad_align)
        hidden_size = round_up(hidden_size, pad_align)
    else:
        intermediate_size = round_up(intermediate_size, 64)
    return hidden_size, intermediate_size


def make_mxfp4_moe_quant_config(
    mxfp4_backend: Mxfp4MoeBackend,
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig | None:
    if mxfp4_backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
    ):
        return None
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
        Mxfp4MoeBackend.TRITON_MONOLITHIC,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
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
    experts_cls: type[mk.FusedMoEPermuteExpertsUnpermute],
    mxfp4_backend: Mxfp4MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
):
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

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

    # NOTE(rob): we only want the mk to control the shared_expert
    # if using all2all (for SBO). bnell is making this explict in
    # the new MoE runner class.
    kernel = mk.FusedMoEModularKernel(
        prepare_finalize,
        experts,
        shared_experts=(
            shared_experts
            if moe_config.moe_parallel_config.use_all2all_kernels
            else None
        ),
        moe_parallel_config=moe_config.moe_parallel_config,
        inplace=(
            not moe_config.disable_inplace
            and mxfp4_backend
            not in (
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLITHIC,
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
            )
        ),
    )

    return kernel
