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
    nvfp4_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    CutlassExpertsFp4,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    is_flashinfer_fp4_cutedsl_moe_available,
    is_flashinfer_fp4_cutlass_moe_available,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported,
)

logger = init_logger(__name__)


class NvFp4MoeBackend(Enum):
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
    FLASHINFER_CUTEDSL = "FLashInfer CUTEDSL"
    VLLM_CUTLASS = "vLLM CUTASS"
    MARLIN = "vLLM MARLIN"


FLASHINFER_NVFP4_MOE_BACKENDS = [
    NvFp4MoeBackend.FLASHINFER_CUTLASS,
    NvFp4MoeBackend.FLASHINFER_TRTLLM,
    NvFp4MoeBackend.FLASHINFER_CUTEDSL,
]

fi_2_vllm_backend_map: dict[FlashinferMoeBackend, NvFp4MoeBackend] = {
    FlashinferMoeBackend.CUTLASS: NvFp4MoeBackend.FLASHINFER_CUTLASS,
    FlashinferMoeBackend.TENSORRT_LLM: NvFp4MoeBackend.FLASHINFER_TRTLLM,
    FlashinferMoeBackend.CUTEDSL: NvFp4MoeBackend.FLASHINFER_CUTEDSL,
}


def is_global_sf_supported_for_nvfp4_backend(backend: NvFp4MoeBackend) -> bool:
    return backend in [
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
    ]


def select_nvfp4_moe_backend() -> NvFp4MoeBackend:
    def _make_log_backend(backend: NvFp4MoeBackend):
        return f"Using {backend.value} backend for NvFp4 MoE"

    if cutlass_fp4_supported() and not envs.VLLM_TEST_FORCE_FP8_MARLIN:
        allow_flashinfer = (
            is_flashinfer_fp4_cutlass_moe_available()
            or is_flashinfer_fp4_cutedsl_moe_available()
        )
        if allow_flashinfer and envs.VLLM_USE_FLASHINFER_MOE_FP4:
            backend = fi_2_vllm_backend_map[get_flashinfer_moe_backend()]
        else:
            backend = NvFp4MoeBackend.VLLM_CUTLASS
    elif is_fp4_marlin_supported():
        backend = NvFp4MoeBackend.MARLIN
    else:
        raise ValueError("No NvFp4 kernel backend available for NvFp4 MoE.")

    # Log warning if FI backend requested but not available.
    if (
        backend not in FLASHINFER_NVFP4_MOE_BACKENDS
        and envs.VLLM_USE_FLASHINFER_MOE_FP4
    ):
        logger.warning_once(
            "Requested FlashInfer backend for NvFp4 MoE, but it's not available. "
            "Falling back to %s for NvFp4 MoE",
            backend.value,
            scope="local",
        )
    else:
        logger.info_once(_make_log_backend(backend), scope="local")
    return backend


def make_nvfp4_moe_kernel(
    backend: NvFp4MoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> mk.FusedMoEModularKernel | None:
    assert moe_config.dp_size == 1

    UNSUPPORTED_BACKENDS = [
        # TRTLLM does not use the modular kernl abstraction.
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        # CUTEDSL is used with BATCHED (masked) format only.
        # TODO: add here once we support dp/ep via the oracle.
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
    ]

    if backend in UNSUPPORTED_BACKENDS:
        return None

    elif backend == NvFp4MoeBackend.FLASHINFER_CUTLASS:
        return mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(defer_input_quant=True),
            FlashInferExperts(
                out_dtype=moe_config.in_dtype,
                quant_config=quant_config,
                ep_rank=moe_config.ep_rank,
                ep_size=moe_config.ep_size,
                tp_rank=moe_config.tp_rank,
                tp_size=moe_config.tp_size,
                use_dp=False,
                use_deepseek_fp8_block_scale=False,
            ),
        )

    elif backend == NvFp4MoeBackend.VLLM_CUTLASS:
        return mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(defer_input_quant=True),
            CutlassExpertsFp4(
                out_dtype=moe_config.in_dtype,
                # TODO(rob): see what impact this has on expert map?
                max_experts_per_worker=moe_config.num_experts,
                quant_config=quant_config,
            ),
        )

    elif backend == NvFp4MoeBackend.MARLIN:
        return mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            MarlinExperts(quant_config=quant_config),
        )

    else:
        return None


def make_nvfp4_moe_quant_config(
    backend: NvFp4MoeBackend,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    a2_scale: torch.Tensor,
) -> FusedMoEQuantConfig | None:
    UNSUPPORTED = [NvFp4MoeBackend.FLASHINFER_TRTLLM]
    if backend in UNSUPPORTED:
        return None

    elif backend == NvFp4MoeBackend.MARLIN:
        return nvfp4_w4a16_moe_quant_config(
            g1_alphas=w13_scale_2,
            g2_alphas=w2_scale_2,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        )

    g1_alphas = a13_scale * w13_scale_2
    g2_alphas = a2_scale * w2_scale_2
    return nvfp4_moe_quant_config(
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        a1_gscale=(1.0 / a13_scale),
        a2_gscale=(1.0 / a2_scale),
        w1_scale=w13_scale,
        w2_scale=w2_scale,
    )
