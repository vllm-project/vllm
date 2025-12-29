# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    CutlassExpertsFp8,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa: E501
    FlashInferAllGatherMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    AiterExperts,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
    register_moe_scaling_factors,
    rotate_flashinfer_fp8_moe_weights,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    prepare_moe_fp8_layer_for_marlin,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer_moe
from vllm.utils.import_utils import has_deep_gemm

logger = init_logger(__name__)


class Fp8MoeBackend(Enum):
    NONE = 0
    FLASHINFER_TRTLLM = 1
    FLASHINFER_CUTLASS = 2
    DEEPGEMM = 3
    MARLIN = 4
    TRITON = 5
    AITER = 6
    VLLM_CUTLASS = 7


def select_fp8_moe_backend(
    block_quant: bool,
    tp_size: int,
    with_lora_support: bool,
    allow_vllm_cutlass: bool = False,
) -> Fp8MoeBackend:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    # TODO(rob): update so that each mk expresses supported features.
    # TODO(rob): update so that we have priority order for each.

    if with_lora_support:
        return Fp8MoeBackend.TRITON

    def _make_log_backend(backend_name: str):
        return f"Using {backend_name} backend for FP8 MoE"

    # Prefer FlashInfer backends on supported GPUs; allow SM90 and SM100.
    if (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability(90)
        )
        and envs.VLLM_USE_FLASHINFER_MOE_FP8
        and has_flashinfer_moe()
    ):
        backend = get_flashinfer_moe_backend()
        if backend == FlashinferMoeBackend.TENSORRT_LLM:
            logger.info_once(_make_log_backend("FlashInfer TRTLLM"))
            return Fp8MoeBackend.FLASHINFER_TRTLLM
        else:
            if block_quant and current_platform.is_device_capability_family(100):
                raise ValueError(
                    "FlashInfer FP8 MoE throughput backend does not "
                    "support block quantization. Please use "
                    "VLLM_FLASHINFER_MOE_BACKEND=latency "
                    "instead."
                )
            logger.info_once(_make_log_backend("FlashInfer CUTLASS"))
            return Fp8MoeBackend.FLASHINFER_CUTLASS

    # weight-only path for older GPUs without native FP8
    use_marlin = (
        not current_platform.has_device_capability(89)
        or envs.VLLM_TEST_FORCE_FP8_MARLIN
    )
    if current_platform.is_rocm():
        use_marlin = False
    if use_marlin:
        logger.info_once(_make_log_backend("Marlin"))
        return Fp8MoeBackend.MARLIN

    # Determine if we should use DeepGEMM with block-quantized weights:
    # - If explicitly set by user, respect their choice
    # - If not explicitly set (default), disable when TP size is >= 8
    moe_use_deep_gemm = envs.VLLM_MOE_USE_DEEP_GEMM
    if not envs.is_set("VLLM_MOE_USE_DEEP_GEMM") and tp_size >= 8:
        moe_use_deep_gemm = False
        logger.info_once(
            "DeepGEMM MoE is disabled by default when TP size is >= 8. "
            "Set VLLM_MOE_USE_DEEP_GEMM=1 to enable it.",
            scope="local",
        )

    if envs.VLLM_USE_DEEP_GEMM and moe_use_deep_gemm and block_quant:
        if not has_deep_gemm():
            logger.warning_once(
                "DeepGEMM backend requested but not available.", scope="local"
            )
        elif is_deep_gemm_supported():
            logger.info_once(_make_log_backend("DeepGEMM"), scope="local")
            return Fp8MoeBackend.DEEPGEMM

    if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MOE:
        logger.info_once(_make_log_backend("ROCm AITER"), scope="local")
        return Fp8MoeBackend.AITER

    if allow_vllm_cutlass:
        logger.info_once(_make_log_backend("vLLM CUTLASS"), scope="local")
        return Fp8MoeBackend.VLLM_CUTLASS

    # default to Triton
    logger.info_once(_make_log_backend("Triton"))
    return Fp8MoeBackend.TRITON


def convert_weights_to_kernel_format(
    fp8_backend: Fp8MoeBackend,
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_quant = hasattr(layer, "weight_block_size")
    if fp8_backend == Fp8MoeBackend.DEEPGEMM:
        assert block_quant
        w13_weight, w13_weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=w13_weight,
            ws=w13_weight_scale,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
        )
        w2_weight, w2_weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=w2_weight,
            ws=w2_weight_scale,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
        )
    elif fp8_backend == Fp8MoeBackend.AITER:
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(w13_weight, w2_weight)
    elif fp8_backend == Fp8MoeBackend.MARLIN:
        workspace, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale = (
            prepare_moe_fp8_layer_for_marlin(
                layer,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                input_dtype=get_marlin_input_dtype(prefix=""),
            )
        )
        layer.workspace = workspace
    elif fp8_backend in [
        Fp8MoeBackend.FLASHINFER_CUTLASS,
        Fp8MoeBackend.FLASHINFER_TRTLLM,
    ]:
        w13_weight = swap_w13_to_w31(w13_weight)
        if block_quant:
            w13_weight_scale = swap_w13_to_w31(w13_weight_scale)
        else:
            # TODO(rob): this function is a hack that renames the scaling
            # factors in the Module. This is a hack we should clean up.
            register_moe_scaling_factors(layer)
            if fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM:
                rotate_flashinfer_fp8_moe_weights(w13_weight, w2_weight)
    elif fp8_backend == Fp8MoeBackend.AITER:
        w13_weight, w2_weight = rocm_aiter_ops.shuffle_weights(w13_weight, w2_weight)

    return w13_weight, w2_weight, w13_weight_scale, w2_weight_scale


def make_kernel(
    layer: torch.nn.Module,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    fp8_backend: Fp8MoeBackend,
) -> tuple[mk.FusedMoEModularKernel, bool]:
    # NOTE(rob): this is a WIP refactor. We are first migrating
    # all of the kernels in the TP case to use mk. Once this is
    # done, then we will initialzie the TP case and DP/EP case
    # via the same code path (i.e. via maybe_init_modular_kernel).
    # NOTE(rob): in progress migrating all into this format.
    use_inplace = True
    if fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
        kernel = mk.FusedMoEModularKernel(
            # TODO(rob): we can use the generic MoEPrepareAndFinalizeNoEP
            # with the changes to defer input quantization
            FlashInferAllGatherMoEPrepareAndFinalize(
                use_dp=(moe_config.dp_size > 1),
                use_deepseek_fp8_block_scale=moe_quant_config.is_block_quantized,
            ),
            FlashInferExperts(
                out_dtype=torch.get_default_dtype(),
                quant_config=moe_quant_config,
                ep_rank=moe_config.ep_rank,
                ep_size=moe_config.ep_size,
                tp_rank=moe_config.tp_rank,
                tp_size=moe_config.tp_size,
                use_dp=(moe_config.dp_size > 1),
                use_deepseek_fp8_block_scale=moe_quant_config.is_block_quantized,
            ),
        )
        use_inplace = False

    elif fp8_backend == Fp8MoeBackend.AITER:
        kernel = mk.FusedMoEModularKernel(
            # TODO: make defer_input_quant an attr of the AiterExperts
            MoEPrepareAndFinalizeNoEP(defer_input_quant=True),
            AiterExperts(quant_config=moe_quant_config),
        )
    elif fp8_backend == Fp8MoeBackend.MARLIN:
        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            MarlinExperts(quant_config=moe_quant_config),
        )
    elif fp8_backend == Fp8MoeBackend.VLLM_CUTLASS:
        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            CutlassExpertsFp8(
                out_dtype=layer.moe.in_dtype,
                e=layer.local_num_experts,
                n=layer.intermediate_size_per_partition,
                k=layer.hidden_size,
                device=layer.w13_weight.device,
                quant_config=moe_quant_config,
            ),
        )
    else:
        assert fp8_backend in [Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.TRITON]
        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            TritonOrDeepGemmExperts(
                quant_config=moe_quant_config,
                allow_deep_gemm=(fp8_backend == Fp8MoeBackend.DEEPGEMM),
            ),
        )
    return kernel, use_inplace
