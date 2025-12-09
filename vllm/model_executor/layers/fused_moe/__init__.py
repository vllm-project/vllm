# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.utils import activation_without_mul
from vllm.triton_utils import HAS_TRITON

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoEConfig",
    "FusedMoEMethodBase",
    "UnquantizedFusedMoEMethod",
    "FusedMoeWeightScaleSupported",
    "FusedMoEPermuteExpertsUnpermute",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalize",
    "SharedFusedMoE",
    "activation_without_mul",
    "override_config",
    "get_config",
]

if HAS_TRITON:
    # import to register the custom ops
    from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        CutlassBatchedExpertsFp8,
        CutlassExpertsFp8,
        CutlassExpertsW4A8Fp8,
        cutlass_moe_fp4,
        cutlass_moe_fp8,
        cutlass_moe_w4a8_fp8,
    )
    from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
    from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
        BatchedTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        TritonExperts,
        fused_experts,
        fused_topk,
        get_config_file_name,
        grouped_topk,
    )
    from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts,
    )

    __all__ += [
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
        "cutlass_moe_fp8",
        "cutlass_moe_fp4",
        "cutlass_moe_w4a8_fp8",
        "CutlassExpertsFp8",
        "CutlassBatchedExpertsFp8",
        "CutlassExpertsW4A8Fp8",
        "TritonExperts",
        "BatchedTritonExperts",
        "DeepGemmExperts",
        "BatchedDeepGemmExperts",
        "TritonOrDeepGemmExperts",
    ]
else:
    # Some model classes directly use the custom ops. Add placeholders
    # to avoid import errors.
    def _raise_exception(method: str):
        raise NotImplementedError(f"{method} is not implemented as lack of triton.")

    fused_topk = lambda *args, **kwargs: _raise_exception("fused_topk")
    fused_experts = lambda *args, **kwargs: _raise_exception("fused_experts")
