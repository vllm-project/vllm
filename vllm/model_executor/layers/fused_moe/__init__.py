from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                        FusedMoEMethodBase)
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts, fused_moe, fused_topk, get_config_file_name,
        grouped_topk)

__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
    "fused_moe",
    "fused_topk",
    "fused_experts",
    "get_config_file_name",
    "grouped_topk",
]

if not HAS_TRITON:
    # need to do it like this other ruff complains
    __all__ = __all__[:2]
