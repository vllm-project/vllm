# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)


def make_ep_config() -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        dp_size=1,
        dp_rank=0,
        ep_size=2,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )


def test_expert_map_logging_is_meta_safe():
    with torch.device("meta"):
        manager = ExpertMapManager(
            max_num_batched_tokens=16,
            top_k=2,
            global_num_experts=8,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=make_ep_config(),
            placement_strategy="linear",
            enable_eplb=False,
        )

    assert manager.expert_map is not None
    assert manager.expert_map.is_meta
    assert manager.get_compressed_map_string() == "<meta expert_map shape=(8,)>"
