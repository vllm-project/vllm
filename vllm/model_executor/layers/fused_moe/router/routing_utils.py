# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType


def resolve_fused_topk_routing_method(
    scoring_func: str,
    top_k: int,
    renormalize: bool,
) -> RoutingMethodType:
    if scoring_func == "sigmoid":
        return RoutingMethodType.Llama4 if top_k == 1 else RoutingMethodType.DeepSeekV3
    if scoring_func == "softmax":
        return (
            RoutingMethodType.RenormalizeNaive
            if renormalize
            else RoutingMethodType.Default
        )
    raise ValueError(f"Unsupported scoring function: {scoring_func}")
