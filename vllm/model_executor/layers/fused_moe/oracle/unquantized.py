# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from torch.nn import Module

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    get_flashinfer_moe_backend,
    swap_w13_to_w31,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class UnquantizedMoeBackend(Enum):
    FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    AITER = "ROCm AITER"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    CPU = "CPU"
    XPU = "XPU"
    TPU = "TPU"
    OOT = "OOT"


def _get_priority_backends() -> list[UnquantizedMoeBackend]:
    """
    Get available backends in priority order based on platform and config.

    This function can be extended to become more complex as needed.
    """

    if current_platform.is_rocm():
        _AVAILABLE_BACKENDS = [
            UnquantizedMoeBackend.AITER,
            UnquantizedMoeBackend.TRITON,
            UnquantizedMoeBackend.BATCHED_TRITON,
        ]
    elif current_platform.is_cuda():
        _AVAILABLE_BACKENDS = [
            UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            UnquantizedMoeBackend.TRITON,
            UnquantizedMoeBackend.BATCHED_TRITON,
        ]
    elif current_platform.is_xpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.XPU]
    elif current_platform.is_cpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.CPU]
    return _AVAILABLE_BACKENDS


def backend_to_kernel_cls(
    backend: UnquantizedMoeBackend,
) -> type[mk.FusedMoEExperts]:
    if backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM:
        from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
            TrtLlmBf16Experts,
        )

        return TrtLlmBf16Experts

    elif backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        return FlashInferExperts

    elif backend == UnquantizedMoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        return AiterExperts

    elif backend == UnquantizedMoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

        return TritonExperts

    elif backend == UnquantizedMoeBackend.BATCHED_TRITON:
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return BatchedTritonExperts

    elif backend == UnquantizedMoeBackend.XPU:
        from vllm.model_executor.layers.fused_moe.xpu_fused_moe import XPUExperts

        return XPUExperts

    else:
        raise ValueError(f"Unknown unquantized MoE backend: {backend.value}")


def map_unquantized_backend(runner_backend: MoEBackend) -> UnquantizedMoeBackend:
    """Map user's MoEBackend to UnquantizedMoeBackend."""
    mapping = {
        "triton": UnquantizedMoeBackend.TRITON,
        "flashinfer_trtllm": UnquantizedMoeBackend.FLASHINFER_TRTLLM,
        "flashinfer_cutlass": UnquantizedMoeBackend.FLASHINFER_CUTLASS,
        "aiter": UnquantizedMoeBackend.AITER,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for unquantized MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
    """
    Select the primary Unquantized MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    if current_platform.is_cpu():
        # TODO: migrate to MK structure.
        return UnquantizedMoeBackend.CPU, None

    if current_platform.is_tpu():
        return UnquantizedMoeBackend.TPU, None

    if current_platform.is_out_of_tree():
        return UnquantizedMoeBackend.OOT, None

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = _get_priority_backends()

    # NOTE(rob): We need to peak into the P/F selection to determine
    # if we are using the batched or standard expert format, which
    # if not ideal. Once we unify TP + DP/EP, we can select P/F first.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if moe_config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: UnquantizedMoeBackend) -> str:
        available_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Unquantized MoE backend out "
            f"of potential backends: {available_strs}."
        )

    def _make_log_unsupported(
        backend: UnquantizedMoeBackend, reason: str | None
    ) -> str:
        if reason:
            return (
                f"Unquantized MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"Unquantized MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: UnquantizedMoeBackend,
        config: FusedMoEConfig,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, config, None, None, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend), scope="local")
            return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    runner_backend = moe_config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_unquantized_backend(runner_backend)
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == UnquantizedMoeBackend.TRITON
        ):
            requested_backend = UnquantizedMoeBackend.BATCHED_TRITON

        return _return_or_raise(requested_backend, moe_config, activation_format)

    # Handle explicit FlashInfer FP16 configuration.
    if envs.is_set("VLLM_USE_FLASHINFER_MOE_FP16"):
        if not envs.VLLM_USE_FLASHINFER_MOE_FP16:
            if UnquantizedMoeBackend.FLASHINFER_TRTLLM in AVAILABLE_BACKENDS:
                AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.FLASHINFER_TRTLLM)
            if UnquantizedMoeBackend.FLASHINFER_CUTLASS in AVAILABLE_BACKENDS:
                AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.FLASHINFER_CUTLASS)
        elif envs.is_set("VLLM_FLASHINFER_MOE_BACKEND"):
            # If user is explicit about backend, validate it.
            fi_backend = get_flashinfer_moe_backend()
            if fi_backend == FlashinferMoeBackend.CUTLASS:
                backend = UnquantizedMoeBackend.FLASHINFER_CUTLASS
            elif fi_backend == FlashinferMoeBackend.TENSORRT_LLM:
                backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM
            else:
                raise ValueError(
                    f"FlashInfer MOE backend {fi_backend} "
                    "does not support unquantized MoE."
                )
            k_cls = backend_to_kernel_cls(backend)
            return _return_or_raise(backend, moe_config, activation_format)
        else:
            # If the user is not explicit about the backend, try both.
            for backend in [
                UnquantizedMoeBackend.FLASHINFER_TRTLLM,
                UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            ]:
                k_cls = backend_to_kernel_cls(backend)
                supported, reason = k_cls.is_supported_config(
                    k_cls, moe_config, None, None, activation_format
                )
                if supported:
                    logger.info_once(_make_log_backend(backend), scope="local")
                    return backend, k_cls
                else:
                    logger.debug_once(
                        _make_log_unsupported(backend, reason), scope="local"
                    )

            raise NotImplementedError(
                "Found VLLM_USE_FLASHINFER_MOE_FP16=1, but no "
                "FlashInfer unquantized MoE backend supports the configuration."
            )

    for backend in AVAILABLE_BACKENDS:
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, moe_config, None, None, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend), scope="local")
            return backend, k_cls

        logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    raise NotImplementedError(
        "No unquantized MoE backend supports the deployment configuration."
    )


def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    layer: Module,
    w13_weight: torch.Tensor | None = None,
    w2_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unquantized_backend == UnquantizedMoeBackend.AITER:
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(w13_weight, w2_weight)

    elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        # Swap halves to arrange as [w3; w1] (kernel expectation)
        w13_weight = swap_w13_to_w31(w13_weight)

    elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM:
        # Swap halves to arrange as [w3; w1] (kernel expectation)
        w13_weight = swap_w13_to_w31(w13_weight)
        _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        w13_weight, w2_weight = convert_moe_weights_to_flashinfer_trtllm_block_layout(
            _cache_permute_indices,
            w13_weight,
            w2_weight,
        )

    return w13_weight, w2_weight


def make_unquantized_moe_kernel(
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    backend: UnquantizedMoeBackend,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize
    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

    # Create Experts
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=quant_config,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=(
            shared_experts
            if (
                moe_config.moe_parallel_config.use_deepep_ll_kernels
                and not is_monolithic
            )
            else None
        ),
        moe_parallel_config=moe_config.moe_parallel_config,
        inplace=(
            not moe_config.disable_inplace
            and backend != UnquantizedMoeBackend.FLASHINFER_CUTLASS
        ),
    )

    return kernel
