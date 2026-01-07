# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    fp8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
    make_fp8_moe_alpha_scales_for_fi,
    prepare_fp8_moe_layer_for_fi,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    prepare_fp8_moe_layer_for_deepgemm,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_fp8_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_group_gemm_supported,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
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
    is_act_and_mul: bool = True,
    allow_vllm_cutlass: bool = False,
) -> Fp8MoeBackend:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    # TODO(rob): in a future PR, we will query each mk for
    # supported features and return the mk directly, just like
    # we do for the Attention Backend.

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
            if not is_act_and_mul:
                raise ValueError(
                    "FlashInfer TRTLLM FP8 MoE backend only supports "
                    "act_and_mul gate_up_project fusion. Please set "
                    "VLLM_USE_FLASHINFER_MOE_FP8=throughput to use the "
                    "FlashInfer CUTLASS backend instead."
                )
            return Fp8MoeBackend.FLASHINFER_TRTLLM
        else:
            if block_quant and current_platform.is_device_capability_family(100):
                raise ValueError(
                    "FlashInfer FP8 MoE throughput backend does not "
                    "support block quantization on SM100. Please use "
                    "VLLM_FLASHINFER_MOE_BACKEND=latency to use the "
                    "FlashInfer TRTLLM backend instead."
                )
            logger.info_once(_make_log_backend("FlashInfer CUTLASS"))
            return Fp8MoeBackend.FLASHINFER_CUTLASS

    # weight-only path for older GPUs without native FP8
    if (
        current_platform.is_cuda() and not current_platform.has_device_capability(89)
    ) or envs.VLLM_TEST_FORCE_FP8_MARLIN:
        logger.info_once(_make_log_backend("Marlin"), scope="local")
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

    use_deep_gemm = envs.VLLM_USE_DEEP_GEMM
    if not is_deep_gemm_supported():
        use_deep_gemm = False
        logger.info_once(
            "DeepGEMM is disabled because the platform does not support it.",
            scope="local",
        )

    if use_deep_gemm and moe_use_deep_gemm and block_quant:
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

    if allow_vllm_cutlass and not block_quant and cutlass_group_gemm_supported():
        logger.info_once(_make_log_backend("vLLM CUTLASS"), scope="local")
        return Fp8MoeBackend.VLLM_CUTLASS

    # default to Triton
    logger.info_once(_make_log_backend("Triton"), scope="local")
    return Fp8MoeBackend.TRITON


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
    if fp8_backend == Fp8MoeBackend.DEEPGEMM:
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

    return w13, w2, w13_scale, w2_scale


def make_fp8_moe_quant_config(
    fp8_backend: Fp8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig | None:
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
    # TRTLLM does not use Modular Kernel abstraction yet.
    if fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM:
        return None

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
        g1_alphas, g2_alphas = make_fp8_moe_alpha_scales_for_fi(
            w1_scale,
            a1_scale,
            w2_scale,
            a2_scale,
        )
        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            a1_gscale=(1.0 / a1_scale),
            a2_gscale=(1.0 / a2_scale),
            g1_alphas=g1_alphas,
            g2_alphas=g2_alphas,
        )
    # All other backends use normal config.
    return fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )


def make_fp8_moe_kernel(
    layer: torch.nn.Module,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    fp8_backend: Fp8MoeBackend,
) -> tuple[mk.FusedMoEModularKernel, bool]:
    # Delayed import is required since the oracle is imported
    # by CPU backends which cannot import all of these experts.
    # TODO: update the experts to make this not happen.
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )

    # NOTE(rob): this is a WIP refactor. We are first migrating
    # all of the kernels in the TP case to use mk. Once this is
    # done, then we will initialzie the TP case and DP/EP case
    # via the same code path (i.e. via maybe_init_modular_kernel).
    # NOTE(rob): in progress migrating all into this format.
    use_inplace = True
    if fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
            FlashInferExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(
                defer_input_quant=moe_quant_config.is_block_quantized
            ),
            FlashInferExperts(
                out_dtype=layer.orig_dtype,
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
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            AiterExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            # TODO: make defer_input_quant an attr of the AiterExperts
            MoEPrepareAndFinalizeNoEP(defer_input_quant=True),
            AiterExperts(quant_config=moe_quant_config),
        )
    elif fp8_backend == Fp8MoeBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            MarlinExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            MarlinExperts(quant_config=moe_quant_config),
        )
    elif fp8_backend == Fp8MoeBackend.VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.triton_cutlass_moe import (
            TritonOrCutlassExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            TritonOrCutlassExperts(
                out_dtype=moe_config.in_dtype,
                e=layer.local_num_experts,
                n=layer.intermediate_size_per_partition,
                k=layer.hidden_size,
                device=layer.w13_weight.device,
                quant_config=moe_quant_config,
            ),
        )
    elif fp8_backend == Fp8MoeBackend.DEEPGEMM:
        from vllm.model_executor.layers.fused_moe import (
            TritonOrDeepGemmExperts,
        )

        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            TritonOrDeepGemmExperts(quant_config=moe_quant_config),
        )
    else:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            TritonExperts,
        )

        assert fp8_backend == Fp8MoeBackend.TRITON
        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            TritonExperts(quant_config=moe_quant_config),
        )
    return kernel, use_inplace
