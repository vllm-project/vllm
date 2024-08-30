from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.triton_utils import HAS_TRITON

__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
]

if HAS_TRITON:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts, fused_moe, fused_topk, get_config_file_name,
        grouped_topk)
    from vllm.model_executor.layers.fused_moe.fused_moe_marlin import (
        fused_moe_marlin, single_moe_marlin)

    __all__ += [
        "fused_moe_marlin",
        "single_moe_marlin",
        "fused_moe",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
    ]
