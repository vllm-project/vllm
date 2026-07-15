# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts


def _moe_config(*, enable_eplb: bool) -> FusedMoEConfig:
    parallel_config = SimpleNamespace(
        enable_eplb=enable_eplb,
        all2all_backend="deepep_low_latency",
        use_fi_nvl_two_sided_kernels=False,
    )
    return cast(
        FusedMoEConfig,
        SimpleNamespace(moe_parallel_config=parallel_config),
    )


def test_set_moe_config_updates_overlap_decision():
    shared_experts = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(shared_experts)
    shared_experts._moe_config = _moe_config(enable_eplb=True)
    assert shared_experts._disable_shared_experts_overlap

    new_moe_config = _moe_config(enable_eplb=False)
    shared_experts._set_moe_config(new_moe_config)

    assert shared_experts._moe_config is new_moe_config
    assert not shared_experts._disable_shared_experts_overlap
