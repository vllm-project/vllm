# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    fp8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
    prepare_fp8_moe_layer_for_fi,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    prepare_fp8_moe_layer_for_deepgemm,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_fp8_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Fp8MoeBackend(Enum):
    NONE = "NONE"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    DEEPGEMM = "DEEPGEMM"
    BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    AITER = "AITER"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    BATCHED_VLLM_CUTLASS = "BATCHED_VLLM_CUTLASS"


def backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> type[mk.FusedMoEPermuteExpertsUnpermute]:
    if backend == Fp8MoeBackend.FLASHINFER_TRTLLM:
        from vllm.model_executor.layers.fused_moe.flashinfer_trtllm_fp8_moe import (
            FlashInferTrtLlmFp8Experts,
        )

        return FlashInferTrtLlmFp8Experts

    elif backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        return FlashInferExperts

    elif backend == Fp8MoeBackend.DEEPGEMM:
        from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
            TritonOrDeepGemmExperts,
        )

        return TritonOrDeepGemmExperts

    elif backend == Fp8MoeBackend.BATCHED_DEEPGEMM:
        from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )

        return BatchedDeepGemmExperts

    elif backend == Fp8MoeBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            MarlinExperts,
        )

        return MarlinExperts

    elif backend == Fp8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            TritonExperts,
        )

        return TritonExperts

    elif backend == Fp8MoeBackend.BATCHED_TRITON:
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return BatchedTritonExperts

    elif backend == Fp8MoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        return AiterExperts

    elif backend == Fp8MoeBackend.VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.triton_cutlass_moe import (
            TritonOrCutlassExperts,
        )

        return TritonOrCutlassExperts

    elif backend == Fp8MoeBackend.BATCHED_VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            CutlassBatchedExpertsFp8,
        )

        return CutlassBatchedExpertsFp8

    else:
        raise ValueError(f"Unknown FP8 MoE backend: {backend.value}")


def select_fp8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    allow_vllm_cutlass: bool = False,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEPermuteExpertsUnpermute] | None]:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    k_cls: type[mk.FusedMoEPermuteExpertsUnpermute] | None = None

    if config.is_lora_enabled:
        return Fp8MoeBackend.TRITON, backend_to_kernel_cls(Fp8MoeBackend.TRITON)

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = [
        Fp8MoeBackend.AITER,
        Fp8MoeBackend.FLASHINFER_TRTLLM,
        Fp8MoeBackend.FLASHINFER_CUTLASS,
        Fp8MoeBackend.DEEPGEMM,
        Fp8MoeBackend.BATCHED_DEEPGEMM,
        Fp8MoeBackend.VLLM_CUTLASS,
        Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        Fp8MoeBackend.TRITON,
        Fp8MoeBackend.BATCHED_TRITON,
        Fp8MoeBackend.MARLIN,
    ]

    # NOTE(rob): We need to peak into the P/F selection to determine
    # if we are using the batched or standard expert format, which
    # if not ideal. Once we unify TP + DP/EP, we can select P/F first.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Fp8MoeBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Fp8 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Fp8MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"FP8 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"FP8 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: Fp8MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Fp8MoeBackend, type[mk.FusedMoEPermuteExpertsUnpermute]]:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, config, weight_key, activation_key, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend), scope="local")
            return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit FlashInfer FP8 configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_FP8"):
        if not envs.VLLM_USE_FLASHINFER_MOE_FP8:
            # If the user rejects FlashInfer remove those backends.
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.FLASHINFER_TRTLLM)
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.FLASHINFER_CUTLASS)

        elif envs.is_set("VLLM_FLASHINFER_MOE_BACKEND"):
            # If user is explicit about backend, validate it.
            fi_backend = get_flashinfer_moe_backend()
            if fi_backend == FlashinferMoeBackend.CUTLASS:
                backend = Fp8MoeBackend.FLASHINFER_CUTLASS
            elif fi_backend == FlashinferMoeBackend.TENSORRT_LLM:
                backend = Fp8MoeBackend.FLASHINFER_TRTLLM
            else:
                raise ValueError(
                    f"FlashInfer MOE backend {fi_backend} does not support FP8 MoE."
                )
            k_cls = backend_to_kernel_cls(backend)
            return _return_or_raise(
                backend, config, weight_key, activation_key, activation_format
            )
        else:
            # If the user is not explicit about the backend, try both.
            for backend in [
                Fp8MoeBackend.FLASHINFER_TRTLLM,
                Fp8MoeBackend.FLASHINFER_CUTLASS,
            ]:
                k_cls = backend_to_kernel_cls(backend)
                supported, reason = k_cls.is_supported_config(
                    k_cls,
                    config,
                    weight_key,
                    activation_key,
                    activation_format,
                )

            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

            raise NotImplementedError(
                "Found VLLM_USE_FLASHINFER_MOE_FP8=1, but no "
                "FlashInfer FP8 MoE backend supports the configuration."
            )

    # Handle explicit DeepGEMM FP8 configuration.
    if envs.is_set("VLLM_USE_DEEP_GEMM") or envs.is_set("VLLM_MOE_USE_DEEP_GEMM"):
        if not envs.VLLM_USE_DEEP_GEMM or not envs.VLLM_MOE_USE_DEEP_GEMM:
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.DEEPGEMM)
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.BATCHED_DEEPGEMM)
        else:
            backend = (
                Fp8MoeBackend.DEEPGEMM
                if activation_format == mk.FusedMoEActivationFormat.Standard
                else Fp8MoeBackend.BATCHED_DEEPGEMM
            )
            return _return_or_raise(
                backend, config, weight_key, activation_key, activation_format
            )

    # Handle explicit MARLIN FP8 configuration.
    if envs.VLLM_TEST_FORCE_FP8_MARLIN:
        backend = Fp8MoeBackend.MARLIN
        return _return_or_raise(
            backend, config, weight_key, activation_key, activation_format
        )

    # Handle explicit AITER FP8 configuration.
    if envs.is_set("VLLM_ROCM_USE_AITER") or envs.is_set("VLLM_ROCM_USE_AITER_MOE"):
        if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.AITER)
        else:
            backend = Fp8MoeBackend.AITER
            return _return_or_raise(
                backend, config, weight_key, activation_key, activation_format
            )

    if not allow_vllm_cutlass:
        AVAILABLE_BACKENDS.remove(Fp8MoeBackend.VLLM_CUTLASS)
        AVAILABLE_BACKENDS.remove(Fp8MoeBackend.BATCHED_VLLM_CUTLASS)

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls,
            config,
            weight_key,
            activation_key,
            activation_format,
        )

        if supported:
            logger.info_once(_make_log_backend(backend), scope="local")
            return backend, k_cls
        else:
            logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    # TODO(rob): per discussion with TPU team, we need a way to register
    # MoE backends by OOT plugins, rather than having an explicit list
    # of AVAILBLE_BACKENDS. Enabling returning `Fp8MoeBackend.NONE` is
    # a temporary measure until these register APIs are complete.
    if current_platform.is_cuda() or current_platform.is_rocm():
        raise NotImplementedError(
            "No FP8 MoE backend supports the deployment configuration."
        )

    return Fp8MoeBackend.NONE, None


def convert_to_fp8_moe_kernel_format(
    fp8_backend: Fp8MoeBackend,
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_input_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_quant = hasattr(layer, "weight_block_size")
    if fp8_backend in [Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.BATCHED_DEEPGEMM]:
        assert block_quant
        w13, w2, w13_scale, w2_scale = prepare_fp8_moe_layer_for_deepgemm(
            w13,
            w2,
            w13_scale,
            w2_scale,
            tuple(layer.weight_block_size),
        )
    elif fp8_backend == Fp8MoeBackend.AITER:
        w13, w2 = rocm_aiter_ops.shuffle_weights(w13, w2)
    elif fp8_backend == Fp8MoeBackend.MARLIN:
        w13, w2, w13_scale, w2_scale = prepare_fp8_moe_layer_for_marlin(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
        )
    elif fp8_backend in [
        Fp8MoeBackend.FLASHINFER_CUTLASS,
        Fp8MoeBackend.FLASHINFER_TRTLLM,
    ]:
        w13, w2, w13_scale = prepare_fp8_moe_layer_for_fi(
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w13_input_scale=w13_input_scale,
            w2_scale=w2_scale,
            w2_input_scale=w2_input_scale,
            is_trtllm=(fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM),
        )
    else:
        if fp8_backend not in [
            Fp8MoeBackend.TRITON,
            Fp8MoeBackend.BATCHED_TRITON,
            Fp8MoeBackend.VLLM_CUTLASS,
            Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        ]:
            raise ValueError(f"Unsupported FP8 MoE backend: {fp8_backend.value}")

    return w13, w2, w13_scale, w2_scale


def make_fp8_moe_quant_config(
    fp8_backend: Fp8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
) -> FusedMoEQuantConfig:
    """
    Create FusedMoEQuantConfig for the specifed FP8 Backend.
    The FusedMoEQuantConfig holds the scales that are used
    at runtime by the Modular Kernel abstraction.

    Note that certain kernels (e.g. Flashinfer CUTLASS) need
    special Quant configs to handle non-standard inputs to
    their kernel interfaces.

    In a future PR, we will have this function should be
    a method of the modular kernel itself.
    """

    # MARLIN is mixed precision W8A16 config.
    if fp8_backend == Fp8MoeBackend.MARLIN:
        return fp8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )

    # Flashinfer CUTLASS per-tensor uses single dq scale
    # (alpha = w_scale * a_scale) and inverse a2 scale.
    if fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS and block_shape is None:
        assert a1_scale is not None and a2_scale is not None
        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            a1_gscale=(1.0 / a1_scale),
            a2_gscale=(1.0 / a2_scale),
            g1_alphas=(w1_scale * a1_scale).squeeze(),
            g2_alphas=(w2_scale * a2_scale).squeeze(),
        )
    # All other backends use normal config.
    return fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=per_out_ch_quant,
    )


def make_fp8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEPermuteExpertsUnpermute],
    fp8_backend: Fp8MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> tuple[mk.FusedMoEModularKernelBase, bool]:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
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

    # NOTE(rob): we only want the mk to control the shared_expert
    # if using all2all (for SBO). bnell is making this explict in
    # the new MoE runner class.
    kernel = mk.FusedMoEModularKernelBase.make_mk(
        prepare_finalize,
        experts,
        shared_experts=(
            shared_experts
            if moe_config.moe_parallel_config.use_all2all_kernels
            else None
        ),
        moe_parallel_config=moe_config.moe_parallel_config,
    )

    # TODO(rob): update inplace logic to be part of the kernel.
    inplace = fp8_backend != Fp8MoeBackend.FLASHINFER_CUTLASS
    return kernel, inplace
