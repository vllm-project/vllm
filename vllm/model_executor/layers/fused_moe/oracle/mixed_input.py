# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)

logger = init_logger(__name__)


class MixedInputMoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    TRITON = "TRITON"


def backend_to_kernel_cls(
    backend: MixedInputMoEBackend,
) -> type[mk.FusedMoEPermuteExpertsUnpermute]:
    if backend == MixedInputMoEBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            MarlinExperts,
        )

        return MarlinExperts
    elif backend == MixedInputMoEBackend.BATCHED_MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            BatchedMarlinExperts,
        )

        return BatchedMarlinExperts
    elif backend == MixedInputMoEBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            TritonWNA16Experts,
        )

        return TritonWNA16Experts
    else:
        raise ValueError(f"Unknown MixedInput MoE backend: {backend.value}")


def select_mixed_input_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[MixedInputMoEBackend, type[mk.FusedMoEPermuteExpertsUnpermute] | None]:
    """
    Select the primary MixedInput MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = [
        MixedInputMoEBackend.MARLIN,
        MixedInputMoEBackend.BATCHED_MARLIN,
        MixedInputMoEBackend.TRITON,
    ]

    # NOTE(rob): this is kind of a hack. We need to peak into
    # the prepare-finalize selection to determine if we are using
    # the batched or standard expert format.
    use_batched = (
        config.moe_parallel_config.use_deepep_ll_kernels
        or config.moe_parallel_config.use_pplx_kernels
    )
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if use_batched
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: MixedInputMoEBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using '{backend.value}' MixedInput MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: MixedInputMoEBackend, reason: str | None) -> str:
        if reason:
            return (
                f"MixedInput MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"MixedInput MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

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

    raise NotImplementedError(
        "No MixedInput MoE backend supports the deployment configuration."
    )


def convert_to_mixed_input_moe_kernel_format(
    backend: MixedInputMoEBackend,
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor | None,
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
    return (
        w13,
        w13_scale,
        w13_scale_2,
        a13_scale,
        w2,
        w2_scale,
        w2_scale_2,
        a2_scale,
    )


def make_mixed_input_quant_config(
    backend: MixedInputMoEBackend,
    num_bits: int,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    group_size: int | None = None,
) -> FusedMoEQuantConfig | None:
    if num_bits == 4:
        return int4_w4a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            group_size=(group_size or 0),
        )
    elif num_bits == 8:
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            group_size=(group_size or 0),
        )
    else:
        raise ValueError(
            f"MixedInput MoE backend '{backend.value}' does not support "
            f"{num_bits}-bit quantization."
        )


def make_mixed_input_moe_kernel_for_mkm(
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    experts_cls: type[mk.FusedMoEPermuteExpertsUnpermute],
    prepare_finalize: mk.FusedMoEPrepareAndFinalize,
) -> mk.FusedMoEPermuteExpertsUnpermute:
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens_per_rank is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens_per_rank,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=quant_config,
        ) 

    logger.debug_once("Using %s", experts.__class__.__name__)
    return experts


def make_mixed_input_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEPermuteExpertsUnpermute],
) -> mk.FusedMoEModularKernel:
    # TODO(rob): unify after we merge tp and dp/ep.
    if (
        moe_config.moe_parallel_config.use_all2all_kernels
        and moe_config.moe_parallel_config.all2all_backend
        not in ["allgather_reducescatter", "naive"]
    ):
        raise ValueError(
            "NvFP4 Oracle should not create non-naive A2A P/F. "
            "This should happen via the ModularKernelMethod."
        )

    # Create Prepare/Finalize.
    prepare_finalize = MoEPrepareAndFinalizeNoEP(
        defer_input_quant=experts_cls.expects_unquantized_inputs(
            moe_config, moe_quant_config
        ),
    )

    # Create Experts.
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
        shared_experts=None,
        moe_parallel_config=moe_config.moe_parallel_config,
    )

    # TODO(rob): update inplace logic to be part of the kernel.
    return kernel
