# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which parallel configs the TRTLLM MoE-LoRA experts accept.

Both kernels read the token->slot mapping the naive DP+EP dispatch gathers,
so neither needs `dp_size == 1`.
"""

import pytest

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmBf16LoRAExperts,
    TrtLlmMxfp4LoRAExperts,
)


def _parallel_config(
    dp_size: int, all2all_backend: str, enable_eplb: bool = False
) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=dp_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend=all2all_backend,
        enable_eplb=enable_eplb,
    )


@pytest.mark.parametrize("experts_cls", [TrtLlmBf16LoRAExperts, TrtLlmMxfp4LoRAExperts])
def test_trtllm_lora_experts_support_dp_with_allgather_dispatch(experts_cls):
    """Both TRTLLM LoRA expert kernels take the naive DP+EP dispatch."""
    assert experts_cls._supports_parallel_config(
        _parallel_config(1, "allgather_reducescatter")
    )
    assert experts_cls._supports_parallel_config(
        _parallel_config(4, "allgather_reducescatter")
    )
    # The other all2all backends do not carry the token->slot mapping, and
    # EPLB moves experts out from under the per-expert LoRA weights.
    assert not experts_cls._supports_parallel_config(
        _parallel_config(4, "deepep_high_throughput")
    )
    assert not experts_cls._supports_parallel_config(
        _parallel_config(4, "allgather_reducescatter", enable_eplb=True)
    )
