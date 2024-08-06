from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts, fused_moe, fused_topk, get_config_file_name,
        grouped_topk)
    from vllm.model_executor.layers.fused_moe.fused_moe_awq import (
        fused_experts_awq)
    from vllm.model_executor.layers.fused_moe.layer import (FusedMoE,
                                                            FusedMoEMethodBase)

    __all__ = [
        "FusedMoE",
        "FusedMoEMethodBase",
        "fused_experts_awq",
        "fused_moe",
        "fused_experts",
        "fused_topk",
        "get_config_file_name",
        "grouped_topk",
    ]
