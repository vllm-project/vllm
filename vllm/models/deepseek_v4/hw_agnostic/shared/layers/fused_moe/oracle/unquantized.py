# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from torch.nn import Module

import vllm.envs as envs
import vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.platforms import current_platform

# DSv4 hw-agnostic doesn't exercise the FlashInfer Cutlass / TRTLLM
# unquantized MoE kernels — those are NV-specific fast paths. The
# upstream copy imported ``swap_w13_to_w31`` and
# ``convert_moe_weights_to_flashinfer_trtllm_block_layout`` from the
# forbidden quantization-utils subtree; the corresponding branches in
# ``convert_to_unquantized_kernel_format`` raise NotImplementedError on
# this path. OOT vendor plugins re-add the FlashInfer paths via their
# own subclass.

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

    elif current_platform.is_xpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.XPU]
    elif current_platform.is_cpu():
        _AVAILABLE_BACKENDS = [UnquantizedMoeBackend.CPU]
    return _AVAILABLE_BACKENDS


def backend_to_kernel_cls(
    backend: UnquantizedMoeBackend,
) -> type[mk.FusedMoEExperts]:
    """The DSv4 hw-agnostic path takes ``UnquantizedMoeBackend.OOT`` from
    ``select_unquantized_moe_backend`` (because ``is_out_of_tree()`` is
    True), which causes its caller to skip ``backend_to_kernel_cls``
    entirely. Calling this function on the hw-agnostic path is a bug —
    the concrete experts modules
    (``vllm.model_executor.layers.fused_moe.experts.*``) live upstream
    and are HW-specific kernels that we deliberately did NOT vendor.
    OOT vendor plugins re-add their own kernel via subclassing.
    """
    raise NotImplementedError(
        f"backend_to_kernel_cls({backend.value!r}) is not vendored on the "
        "DSv4 hw-agnostic FusedMoE path. The OOT plugin's "
        "select_unquantized_moe_backend returns OOT, which short-circuits "
        "before this function is reached."
    )


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

    if moe_config.is_lora_enabled:
        return UnquantizedMoeBackend.TRITON, backend_to_kernel_cls(
            UnquantizedMoeBackend.TRITON
        )

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
        k_cls = backend_to_kernel_cls(backend)
        supported, reason = k_cls.is_supported_config(
            k_cls, config, None, None, activation_format
        )
        if supported:
            logger.info_once(_make_log_backend(backend))
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

    # Handle explicit AITER FP8 configuration.
    if envs.is_set("VLLM_ROCM_USE_AITER") or envs.is_set("VLLM_ROCM_USE_AITER_MOE"):
        if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:
            if UnquantizedMoeBackend.AITER in AVAILABLE_BACKENDS:
                AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.AITER)
        else:
            backend = UnquantizedMoeBackend.AITER
            return _return_or_raise(backend, moe_config, activation_format)

    for backend in AVAILABLE_BACKENDS:
        k_cls = backend_to_kernel_cls(backend)
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
    layer: Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unquantized_backend == UnquantizedMoeBackend.AITER:
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(w13_weight, w2_weight)

    elif unquantized_backend in (
        UnquantizedMoeBackend.FLASHINFER_CUTLASS,
        UnquantizedMoeBackend.FLASHINFER_TRTLLM,
    ):
        raise NotImplementedError(
            f"FlashInfer unquantized MoE kernels ({unquantized_backend}) "
            "are not vendored on the DSv4 hw-agnostic path. Use the upstream "
            "FusedMoE for these backends or register an OOT vendor subclass."
        )

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
