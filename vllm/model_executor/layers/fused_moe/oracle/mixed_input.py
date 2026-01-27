# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.fused_moe import FusedMoE
from vllm import _custom_ops as ops

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
    layer: "FusedMoE",
    mixed_input_moe_backend: MixedInputMoEBackend,
    num_bits: int,
    packed_factor: int,
    group_size: int,
    num_groups_w13: int,
    num_groups_w2: int,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_g_idx: torch.Tensor | None = None,
    w2_g_idx: torch.Tensor | None = None,
    actorder: str | None = None,
    marlin_input_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,  # w13
    torch.Tensor,  # w13_scale
    torch.Tensor | None,  # w13_g_idx
    torch.Tensor | None,  # w13_g_idx_sort_idxs
    torch.Tensor | None,  # a13_gscale
    torch.Tensor,  # w2
    torch.Tensor,  # w2_scale
    torch.Tensor | None,  # w2_g_idx
    torch.Tensor | None,  # w2_g_idx_sort_idxs
    torch.Tensor | None,  # a2_gscale
]:
    if mixed_input_moe_backend in [
        MixedInputMoEBackend.MARLIN,
        MixedInputMoEBackend.BATCHED_MARLIN,
    ]:
        is_a_8bit = marlin_input_dtype is not None and marlin_input_dtype.itemsize == 1

        if marlin_input_dtype == torch.float8_e4m3fn:
            # NOTE: for non-zp quantization format only
            ops.marlin_int4_fp8_preprocess(w13, inplace=True)
            ops.marlin_int4_fp8_preprocess(w2, inplace=True)
            w13_scale = w13_scale * 512
            w2_scale = w2_scale * 512

        assert w13_g_idx is not None
        assert w2_g_idx is not None

        # GIDX - for activation re-ordering.
        if actorder == "group":
            num_experts = w13_g_idx.shape[0]
            w13_g_idx_sort_idxs = torch.empty_like(w13_g_idx)
            w2_g_idx_sort_idxs = torch.empty_like(w2_g_idx)
            w13_sorted_g_idx = torch.empty_like(w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(w2_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_idxs[e] = torch.argsort(w13_g_idx[e]).to(torch.int32)
                w2_g_idx_sort_idxs[e] = torch.argsort(w2_g_idx[e]).to(torch.int32)
                w13_sorted_g_idx[e] = w13_g_idx[e][w13_g_idx_sort_idxs[e]]
                w2_sorted_g_idx[e] = w2_g_idx[e][w2_g_idx_sort_idxs[e]]

            w13_g_idx = w13_sorted_g_idx
            w2_g_idx = w2_sorted_g_idx
        else:
            device = w13.device
            E = w13_g_idx.shape[0]
            w13_g_idx = torch.empty((E, 0), dtype=torch.int32, device=device)
            w2_g_idx = torch.empty((E, 0), dtype=torch.int32, device=device)
            w13_g_idx_sort_idxs = torch.empty((E, 0), dtype=torch.int32, device=device)
            w2_g_idx_sort_idxs = torch.empty((E, 0), dtype=torch.int32, device=device)

        # WEIGHTS - repack into MARLIN format.
        w13 = ops.gptq_marlin_moe_repack(
            w13,
            perm=w13_g_idx_sort_idxs,
            size_k=w13.shape[1] * packed_factor,
            size_n=w13.shape[2],
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )
        w2 = ops.gptq_marlin_moe_repack(
            w2,
            perm=w2_g_idx_sort_idxs,
            size_k=w2.shape[1] * packed_factor,
            size_n=w2.shape[2],
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )

        # SCALES - permute into MARLIN format.
        w13_scale = marlin_moe_permute_scales(
            s=w13_scale,
            size_k=w13_scale.shape[1],  # is this right?
            size_n=w13_scale.shape[2],
            group_size=group_size,
            is_a_8bit=is_a_8bit,
        )
        if marlin_input_dtype == torch.int8 and num_groups_w13 > 1:
            w13_scale, a13_gscale = marlin_act_int8_process_scales(w13_scale)

        w2_scale = marlin_moe_permute_scales(
            s=w2_scale,
            size_k=w2_scale.shape[1]
            * (group_size if group_size != -1 else packed_factor),
            size_n=w2_scale.shape[2],
            group_size=group_size,
            is_a_8bit=is_a_8bit,
        )
        if marlin_input_dtype == torch.int8 and num_groups_w2 > 1:
            w2_scale, a2_gscale = marlin_act_int8_process_scales(w2_scale)

        layer.workspace = marlin_make_workspace_new(device, 4)

    return (
        w13,
        w13_scale,
        w13_g_idx,
        w13_g_idx_sort_idxs,
        a13_gscale,
        w2,
        w2_scale,
        w2_g_idx,
        w2_g_idx_sort_idxs,
        a2_gscale,
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


def make_mixed_input_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEPermuteExpertsUnpermute],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEModularKernel:
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
    kernel = mk.FusedMoEModularKernel(
        prepare_finalize,
        experts,
        shared_experts=(
            shared_experts
            if moe_config.moe_parallel_config.use_all2all_kernels
            else None
        ),
        moe_parallel_config=moe_config.moe_parallel_config,
    )

    return kernel
