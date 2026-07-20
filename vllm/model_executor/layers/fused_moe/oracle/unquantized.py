# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    align_moe_weights_for_fi,
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    swap_w13_to_w31,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

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


def _get_priority_backends(moe_config: FusedMoEConfig) -> list[UnquantizedMoeBackend]:
    """
    Get available backends in priority order based on platform and config.

    This function can be extended to become more complex as needed.
    """

    def _move_to_back(
        backends: list[UnquantizedMoeBackend],
        backend: UnquantizedMoeBackend,
    ) -> None:
        backends.append(backends.pop(backends.index(backend)))

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

        # On Hopper (SM90), the FlashInfer unquantized MoE kernels are slower
        # than Triton, so prefer Triton by default.
        if current_platform.is_device_capability_family(90):
            _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_TRTLLM)
            _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)

        # HACK: Qwen3.5 has crash with FLASHINFER_CUTLASS BF16 if DEP.
        # Updating the oracle querying logic is out of the scope of this
        # PR. Need to fix the kernel or update structure in follow up.
        if moe_config.moe_parallel_config.dp_size > 1:
            _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)

        # HACK: unquantized FlashInfer aliases SWIGLUOAI to plain Swiglu
        # (swiglu_alpha/limit only set on the MXFP4 branch). Route to
        # Triton's swigluoai_and_mul until that's plumbed through. Same
        # demotion pattern as the Qwen3.5/dp_size hack above.
        if moe_config.activation == MoEActivation.SWIGLUOAI:
            _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_TRTLLM)
            _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)

    elif current_platform.is_xpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.XPU]
    elif current_platform.is_cpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.CPU]
    return _AVAILABLE_BACKENDS


def backend_to_kernel_cls(
    backend: UnquantizedMoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM:
        from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
            TrtLlmBf16ExpertsModular,
            TrtLlmBf16ExpertsMonolithic,
        )

        return [TrtLlmBf16ExpertsMonolithic, TrtLlmBf16ExpertsModular]

    elif backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (  # noqa: E501
            FlashInferExperts,
        )

        return [FlashInferExperts]

    elif backend == UnquantizedMoeBackend.AITER:
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            AiterExperts,
        )

        return [AiterExperts]

    elif backend == UnquantizedMoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    elif backend == UnquantizedMoeBackend.BATCHED_TRITON:
        from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return [BatchedTritonExperts]

    elif backend == UnquantizedMoeBackend.XPU:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExperts

        return [XPUExperts]

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


def _trtllm_bf16_lora_supported(moe_config: FusedMoEConfig) -> bool:
    """Gate for routing LoRA-enabled BF16 MoE to the FlashInfer TRT-LLM
    gemm1_lora_delta path (PR #3153). Conservative: device + routing method;
    the experts class's own _supports_* checks and the modular_kernel LoRA
    gate provide the final filtering.
    """
    from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
        TrtLlmBf16LoRAExperts,
    )

    if not TrtLlmBf16LoRAExperts._supports_current_device():
        return False
    if not TrtLlmBf16LoRAExperts._supports_routing_method(
        moe_config.routing_method, None, None
    ):
        return False
    if not TrtLlmBf16LoRAExperts._supports_parallel_config(
        moe_config.moe_parallel_config
    ):
        return False
    # The flashinfer trtllm fused-MoE kernel requires the per-partition
    # intermediate size to be a multiple of 128. Plain TP shards the MoE
    # intermediate dim (e.g. 768 -> 192 at tp=4), which would crash the kernel
    # at runtime; fall back to Triton in that case.
    return moe_config.intermediate_size_per_partition % 128 == 0


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

    if moe_config.is_lora_enabled:
        if _trtllm_bf16_lora_supported(moe_config):
            from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
                TrtLlmBf16LoRAExperts,
            )

            logger.info_once(
                "Using TrtLlmBf16LoRAExperts Unquantized MoE LoRA backend "
                "(TrtLlmBf16LoRAExperts)."
            )
            return UnquantizedMoeBackend.FLASHINFER_TRTLLM, TrtLlmBf16LoRAExperts
        logger.info_once("Using TRITON Unquantized MoE LoRA backend")
        return UnquantizedMoeBackend.TRITON, backend_to_kernel_cls(
            UnquantizedMoeBackend.TRITON
        )[0]

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = _get_priority_backends(moe_config)

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
        reason = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, None, None, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    runner_backend = moe_config.moe_backend
    # 'humming' is quantization-only; an unquantized layer (e.g. excluded via
    # modules_to_not_convert) falls through to auto instead of erroring.
    if runner_backend not in ["auto", "humming"]:
        requested_backend = map_unquantized_backend(runner_backend)
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == UnquantizedMoeBackend.TRITON
        ):
            requested_backend = UnquantizedMoeBackend.BATCHED_TRITON

        return _return_or_raise(requested_backend, moe_config, activation_format)

    # Handle explicit AITER FP8 configuration.
    if envs.is_set("VLLM_ROCM_USE_AITER") or envs.is_set("VLLM_ROCM_USE_AITER_MOE"):
        if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:
            if UnquantizedMoeBackend.AITER in AVAILABLE_BACKENDS:
                AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.AITER)
        else:
            backend = UnquantizedMoeBackend.AITER
            return _return_or_raise(backend, moe_config, activation_format)

    for backend in AVAILABLE_BACKENDS:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, moe_config, None, None, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls

            logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No Unquantized MoE backend supports the deployment configuration."
    )


def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    moe_config: FusedMoEConfig,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unquantized_backend == UnquantizedMoeBackend.AITER:
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(w13_weight, w2_weight)

    elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS:
        if moe_config.is_act_and_mul:
            # Swap halves to arrange as [w3; w1] (kernel expectation)
            # Non-gated MoE: w13 is a single projection, no need to swap.
            w13_weight = swap_w13_to_w31(w13_weight)

    elif unquantized_backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM:
        is_act_and_mul = moe_config.is_act_and_mul
        if not is_act_and_mul:
            # Kernel requires intermediate_size_per_partition % 128 == 0 (BlockMajorK
            # weight layout uses block_k=128). Pad along the intermediate dim when
            # the model + TP split don't satisfy the constraint.
            w13_weight, w2_weight, padded_intermediate = align_moe_weights_for_fi(
                w13_weight, w2_weight, is_act_and_mul, min_alignment=128
            )
            moe_config.intermediate_size_per_partition = padded_intermediate

        _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        w13_weight, w2_weight = convert_moe_weights_to_flashinfer_trtllm_block_layout(
            _cache_permute_indices,
            w13_weight,
            w2_weight,
            is_gated_act_gemm=is_act_and_mul,
        )

    if (
        unquantized_backend == UnquantizedMoeBackend.TRITON
        and current_platform.is_rocm()
        and envs.VLLM_ROCM_MOE_PADDING
    ):
        # Skip .contiguous(): it would undo the ROCm MoE weight padding.
        return w13_weight, w2_weight
    return w13_weight.contiguous(), w2_weight.contiguous()


def make_unquantized_moe_kernel(
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    backend: UnquantizedMoeBackend,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
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

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)
    logger.info_once("Using %s MoE backend", experts_cls.__name__)

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
    )

    return kernel


# ---------------------------------------------------------------------------
# Class-based view (first PR of the #37753 series; see oracle/base.py).
# Methods delegate to the module-level functions above so behaviour is
# bit-identical with pre-class code.
# ---------------------------------------------------------------------------


class UnquantizedMoEKernelOracle(MoEKernelOracle[UnquantizedMoeBackend]):
    """Class-based view of the unquantized MoE kernel oracle.

    Each method delegates to its module-level counterpart so that
    instantiating and calling this class is bit-identical to calling
    the standalone functions. Follow-up PRs may move logic from the
    module-level functions into these methods.
    """

    def backend_enum_cls(self) -> type[UnquantizedMoeBackend]:
        return UnquantizedMoeBackend

    def get_priority_backends(
        self, moe_config: FusedMoEConfig
    ) -> list[UnquantizedMoeBackend]:
        return _get_priority_backends(moe_config)

    def backend_to_kernel_cls(
        self, backend: UnquantizedMoeBackend
    ) -> list[type[mk.FusedMoEExperts]]:
        return backend_to_kernel_cls(backend)

    def map_backend(self, runner_backend: MoEBackend) -> UnquantizedMoeBackend:
        return map_unquantized_backend(runner_backend)

    def select_backend(
        self,
        moe_config: FusedMoEConfig,
        weight_key: "QuantKey | None" = None,
        activation_key: "QuantKey | None" = None,
    ) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
        assert weight_key is None and activation_key is None, (
            "Weights and activations will never be quantized for "
            "UnquantizedMoEKernelOracle"
        )
        return select_unquantized_moe_backend(moe_config)

    def convert_to_kernel_format(
        self,
        backend: UnquantizedMoeBackend,
        moe_config: FusedMoEConfig,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return convert_to_unquantized_kernel_format(
            backend, moe_config, w13_weight, w2_weight
        )

    def make_kernel(
        self,
        quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        backend: UnquantizedMoeBackend,
        experts_cls: type[mk.FusedMoEExperts],
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEKernel:
        return make_unquantized_moe_kernel(
            quant_config, moe_config, backend, experts_cls, routing_tables
        )
