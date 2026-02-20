# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Union

import torch

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
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels

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
    FLASHINFER_TRTLLM_MXFP4_MXFP8 = "FLASHINFER_TRTLLMMXFP4_MXFP8"
    FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC = (
        "FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC"
    )
    FLASHINFER_CUTLASS_MXFP4_MXFP8 = "FLASHINFER_MXFP4_MXFP8_CUTLASS"
    FLASHINFER_TRTLLM_MXFP4_BF16 = "FLASHINFER_MXFP4_BF16"
    FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC = "FLASHINFER_MXFP4_BF16_MONOLOTHIC"
    FLASHINFER_CUTLASS_MXFP4_BF16 = "FLASHINFER_MXFP4_BF16"
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
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
        Mxfp4MoeBackend.TRITON_MONOLITHIC,
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
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
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
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC,
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
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEPermuteExpertsUnpermute]]:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, config, None, None, activation_format
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
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC
            )
        else:
            if current_platform.is_device_capability(90):
                backend = Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16
                return _return_or_raise(backend, config, activation_format)
            if current_platform.is_device_capability_family(100):
                # Using modular interface
                # unifying them after #32564 is merged
                if config.dp_size > 1 and config.use_ep:
                    backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16
                    return _return_or_raise(backend, config, activation_format)
                else:
                    backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC
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
            return _return_or_raise(backend, config, activation_format)
        else:
            backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC
            return backend, None

    # Handle explicit FlashInfer MXFP4 MXFP8 CUTLASS configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"):
        if not envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8)
        else:
            backend = Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8
            return _return_or_raise(backend, config, activation_format)

    # Handle explicit Marlin MXFP4 configuration.
    if envs.is_set("VLLM_MXFP4_USE_MARLIN"):
        if not envs.VLLM_MXFP4_USE_MARLIN:
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.MARLIN)
            AVAILABLE_BACKENDS.remove(Mxfp4MoeBackend.BATCHED_MARLIN)
        else:
            backend = Mxfp4MoeBackend.MARLIN
            return _return_or_raise(backend, config, activation_format)

    # FIXME(zyongye): manually select default kernels
    # change to automatic after monolithic kernel PR is merged
    if current_platform.is_device_capability_family(100):
        if config.dp_size > 1 and config.use_ep:
            backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16
            return _return_or_raise(backend, config, activation_format)
        else:
            backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC
            return backend, None
    elif current_platform.is_device_capability(90):
        if config.dp_size > 1 and config.use_ep:
            backend = Mxfp4MoeBackend.TRITON
            return _return_or_raise(backend, config, activation_format)
        else:
            backend = Mxfp4MoeBackend.TRITON_MONOLITHIC
            return backend, None
    elif current_platform.has_device_capability(70):
        # TODO (zyongye): integrate XPU backend
        backend = (
            Mxfp4MoeBackend.MARLIN
            if activation_format == mk.FusedMoEActivationFormat.Standard
            else Mxfp4MoeBackend.BATCHED_MARLIN
        )
        return _return_or_raise(backend, config, activation_format)

    if current_platform.is_cuda() or current_platform.is_rocm():
        raise NotImplementedError(
            "No MXFP4 MoE backend supports the deployment configuration."
        )

    return Mxfp4MoeBackend.NONE, None


def convert_to_mxfp4_moe_kernel_format(): ...


def make_mxfp4_moe_quant_config(
    mxfp4_backend: Mxfp4MoeBackend,
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
):
    if mxfp4_backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
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
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC,
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
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16_MONOLOTHIC,
                Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8_MONOLITHIC,
            )
        ),
    )

    return kernel
